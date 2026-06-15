#!/usr/bin/env python3
"""Stage 1 — Convert raw OpenSeeker / REDSearcher trajectories to single-step
skeleton samples for downstream LLM rewriting (Stage 2).

Output schema (one JSONL line per source assistant turn):

{
  "source":             "openseeker" | "redsearcher",
  "source_traj_id":     "<unique id, e.g. 'openseeker:line:123'>",
  "step_idx":           1,            // 1-based
  "total_steps":        23,           // total assistant turns in this traj
  "is_initial":         true,
  "is_final":           false,        // true on the <answer>-emitting turn
  "language":           "en",
  "task":               "<source question>",

  "source_think":       "<text inside <think>...</think>>",
  "source_raw_assistant": "<full original assistant text>",

  "mapped_tool_call":   {"name": "search", "arguments": {"query": ["..."]}},  // null on final
  "mapped_answer":      null | "<final answer string>",
  "source_observation_after": "<tool_response text of the next user turn>",

  "meta": {
    "ground_truth_answer":     "<source answer>",
    "trajectory_correctness":  "Correct" | "Incorrect" | null,
    "tool_used":               "search" | "visit" | "scholar" | "code_sandbox" | null
  }
}

Trajectories are dropped (with a reason) when:
  - they contain `google_maps` calls anywhere
  - any assistant turn has >1 tool_call (multi-tool not yet supported)
  - the source format is structurally malformed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from glob import glob
from pathlib import Path

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from lib.parse_source import (  # noqa: E402
    extract_answer,
    extract_think,
    extract_tool_calls,
    iter_openseeker_trajectories,
    iter_redsearcher_rows,
)
from lib.tool_mapping import (  # noqa: E402
    DROP_SOURCE_TOOLS,
    UnsupportedToolError,
    map_tool_call,
)


DEFAULT_OPENSEEKER_PATH = "/data/czj/WebDatasets/PolarSeeker/OpenSeeker-v1-Data/openseeker_v1_data.jsonl"
DEFAULT_REDSEARCHER_GLOB = "/data/czj/WebDatasets/Zchu/REDSearcher_SFT_10K/data/train-*.parquet"


# -----------------------------------------------------------------------------
# Per-trajectory walker
# -----------------------------------------------------------------------------


def _walk_assistant_turns(messages: list[dict]) -> list[tuple[int, dict, str]]:
    """Return [(message_index, assistant_msg, next_user_observation_text)].

    next_user_observation_text is the content of the user message immediately
    following this assistant turn, stripped to plain text. If the assistant turn
    is the last message, observation is "".
    """
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        obs = ""
        if i + 1 < len(messages):
            nxt = messages[i + 1]
            if isinstance(nxt, dict) and nxt.get("role") == "user":
                nc = nxt.get("content", "")
                obs = nc if isinstance(nc, str) else ""
        out.append((i, m, obs))
    return out


def _build_steps_for_trajectory(
    *,
    source: str,
    traj_id: str,
    task: str,
    ground_truth_answer: str,
    correctness: str | None,
    language: str,
    messages: list[dict],
    drop_reasons: Counter,
) -> list[dict] | None:
    """Convert one source trajectory into a list of skeleton step dicts, or
    return None to indicate the whole trajectory should be dropped."""

    turns = _walk_assistant_turns(messages)
    if not turns:
        drop_reasons["empty_trajectory"] += 1
        return None

    # First pass: scan all tool_calls in this traj to filter out:
    #   1. google_maps (no hybrid equivalent)
    #   2. multi-tool turns (>1 tool_call in one assistant message)
    raw_tool_calls_per_turn: list[list[tuple[dict, str]]] = []
    for _, assistant_msg, _ in turns:
        content = assistant_msg.get("content", "")
        if not isinstance(content, str):
            content = ""
        calls = extract_tool_calls(content)
        raw_tool_calls_per_turn.append(calls)
        for tc, _attached_code in calls:
            if tc.get("name") in DROP_SOURCE_TOOLS:
                drop_reasons["contains_google_maps"] += 1
                return None
        if len(calls) > 1:
            drop_reasons["multi_tool_call_turn"] += 1
            return None

    # Identify the final assistant turn (the one that has an <answer> tag).
    total_steps = len(turns)
    final_step_idx_1based = None
    for idx_0based, (_, msg, _) in enumerate(turns):
        content = msg.get("content", "") if isinstance(msg.get("content", ""), str) else ""
        if extract_answer(content):
            final_step_idx_1based = idx_0based + 1
            # Some sources have trailing junk after the answer turn; ignore later assistant turns
            turns = turns[: idx_0based + 1]
            raw_tool_calls_per_turn = raw_tool_calls_per_turn[: idx_0based + 1]
            total_steps = idx_0based + 1
            break

    out_steps: list[dict] = []
    for step_idx_1based, ((_, assistant_msg, obs_text), calls) in enumerate(
        zip(turns, raw_tool_calls_per_turn), start=1
    ):
        raw_text = assistant_msg.get("content", "")
        if not isinstance(raw_text, str):
            raw_text = ""

        think_text = extract_think(raw_text)
        is_final = (final_step_idx_1based is not None and step_idx_1based == final_step_idx_1based)

        mapped_tool_call: dict | None = None
        mapped_answer: str | None = None
        tool_used: str | None = None

        if is_final:
            answer = extract_answer(raw_text)
            mapped_answer = answer if (answer and answer.strip()) else ""
            if not mapped_answer:
                drop_reasons["empty_answer_on_final"] += 1
                return None
        else:
            if not calls:
                # Non-final assistant turn without a tool_call: malformed source.
                drop_reasons["nonfinal_turn_without_tool_call"] += 1
                return None
            tc_raw, attached_code = calls[0]
            try:
                mapped_tool_call = map_tool_call(
                    tc_raw.get("name", ""),
                    tc_raw.get("arguments", {}) or {},
                    attached_code=attached_code,
                )
                tool_used = mapped_tool_call["name"]
            except UnsupportedToolError as exc:
                drop_reasons[f"unsupported_tool:{exc}"] += 1
                return None

        out_steps.append(
            {
                "source": source,
                "source_traj_id": traj_id,
                "step_idx": step_idx_1based,
                "total_steps": total_steps,
                "is_initial": step_idx_1based == 1,
                "is_final": is_final,
                "language": language,
                "task": task,
                "source_think": think_text,
                "source_raw_assistant": raw_text,
                "mapped_tool_call": mapped_tool_call,
                "mapped_answer": mapped_answer,
                "source_observation_after": obs_text,
                "meta": {
                    "ground_truth_answer": ground_truth_answer,
                    "trajectory_correctness": correctness,
                    "tool_used": tool_used,
                },
            }
        )

    return out_steps


# -----------------------------------------------------------------------------
# Source-specific drivers
# -----------------------------------------------------------------------------


def run_openseeker(
    src_path: str,
    out_path: str,
    *,
    limit: int | None,
    only_correct: bool,
) -> dict:
    stats = Counter()
    drop_reasons = Counter()
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    with open(out_path_p, "w", encoding="utf-8") as fout:
        for line_idx, traj in enumerate(iter_openseeker_trajectories(src_path)):
            if limit is not None and stats["trajs_emitted"] >= limit:
                break
            stats["trajs_in"] += 1
            correctness = traj.get("trajectory correctness")
            if only_correct and correctness != "Correct":
                drop_reasons["filtered_incorrect"] += 1
                continue
            messages = traj.get("trajectory") or []
            if isinstance(messages, str):
                # defensive — earlier inspection showed it's already a list
                try:
                    messages = json.loads(messages)
                except Exception:
                    drop_reasons["unparsable_trajectory_string"] += 1
                    continue
            task = traj.get("question", "") or ""
            gt_answer = traj.get("answer", "") or ""
            traj_id = f"openseeker:line:{line_idx}"
            steps = _build_steps_for_trajectory(
                source="openseeker",
                traj_id=traj_id,
                task=task,
                ground_truth_answer=gt_answer,
                correctness=correctness,
                language="en",
                messages=messages,
                drop_reasons=drop_reasons,
            )
            if steps is None:
                stats["trajs_dropped"] += 1
                continue
            for step in steps:
                fout.write(json.dumps(step, ensure_ascii=False) + "\n")
                stats["steps_emitted"] += 1
            stats["trajs_emitted"] += 1
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["drop_reasons"] = dict(drop_reasons)
    return dict(stats)


def run_redsearcher(
    parquet_glob: str,
    out_path: str,
    *,
    limit: int | None,
) -> dict:
    stats = Counter()
    drop_reasons = Counter()
    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    seen_idx = 0
    with open(out_path_p, "w", encoding="utf-8") as fout:
        for row in iter_redsearcher_rows(parquet_glob):
            seen_idx += 1
            if limit is not None and stats["trajs_emitted"] >= limit:
                break
            stats["trajs_in"] += 1
            meta = row.get("meta") or {}
            task = meta.get("question", "") or ""
            gt_answer = meta.get("answer", "") or ""
            language = meta.get("language", "") or "en"
            messages = row.get("messages") or []
            traj_id = f"redsearcher:row:{seen_idx}"
            steps = _build_steps_for_trajectory(
                source="redsearcher",
                traj_id=traj_id,
                task=task,
                ground_truth_answer=gt_answer,
                correctness=None,  # REDSearcher has no per-traj correctness label
                language=language,
                messages=messages,
                drop_reasons=drop_reasons,
            )
            if steps is None:
                stats["trajs_dropped"] += 1
                continue
            for step in steps:
                fout.write(json.dumps(step, ensure_ascii=False) + "\n")
                stats["steps_emitted"] += 1
            stats["trajs_emitted"] += 1
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    stats["drop_reasons"] = dict(drop_reasons)
    return dict(stats)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Stage 1 skeleton builder for OpenSeeker + REDSearcher → hybrid."
    )
    ap.add_argument(
        "--source",
        choices=["openseeker", "redsearcher", "both"],
        default="both",
    )
    ap.add_argument("--openseeker-path", default=DEFAULT_OPENSEEKER_PATH)
    ap.add_argument("--redsearcher-glob", default=DEFAULT_REDSEARCHER_GLOB)
    ap.add_argument(
        "--out-dir",
        default="/data/czj/HybridDataConversion/data",
        help="Where to write the skeleton .jsonl files.",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max trajectories to emit per source (useful for dry runs).",
    )
    ap.add_argument(
        "--openseeker-only-correct",
        action="store_true",
        default=False,
        help="Drop OpenSeeker trajectories with trajectory_correctness != 'Correct'.",
    )
    ap.add_argument("--tag", default="", help="Optional suffix in output filenames.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""

    if args.source in ("openseeker", "both"):
        out_path = out_dir / f"openseeker_skeleton{suffix}.jsonl"
        print(f"[openseeker] -> {out_path}")
        stats = run_openseeker(
            args.openseeker_path,
            str(out_path),
            limit=args.limit,
            only_correct=args.openseeker_only_correct,
        )
        print(json.dumps({"openseeker": stats}, ensure_ascii=False, indent=2))

    if args.source in ("redsearcher", "both"):
        out_path = out_dir / f"redsearcher_skeleton{suffix}.jsonl"
        print(f"[redsearcher] -> {out_path}")
        stats = run_redsearcher(
            args.redsearcher_glob,
            str(out_path),
            limit=args.limit,
        )
        print(json.dumps({"redsearcher": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
