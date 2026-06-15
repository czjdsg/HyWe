#!/usr/bin/env python3
"""Build HybridWebAgent hybrid_routed Search-side SFT data from Search rewrites.

Input:
  - Stage-2 Search rewrite JSONL, append-only and possibly containing reruns.
  - Original Search skeleton JSONL files, used only to recover
    source_observation_after for the next step's "Previous Observation".

Output:
  - One JSONL row per HybridWebAgent decision-model call.
  - Only the latest successful final attempt for each trajectory is emitted.
  - All rows are routed Search-side: agent_mode=hybrid_routed, active_mode=S.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
HWA_ROOT = Path("/data/czj/HybridWebAgent")
if str(HWA_ROOT) not in sys.path:
    sys.path.insert(0, str(HWA_ROOT))

from hybrid_prompts import get_system_prompt, get_user_prompt_template  # noqa: E402
from hybrid_utils import compact_observation_text, sanitize_surrogates  # noqa: E402


AUDIT_KEYS = (
    "date_evidence",
    "time_dependent",
    "time_dependent_reason",
    "inferred_date",
    "step_quality",
    "step_quality_reason",
)


def json_dumps(obj: Any, *, indent: int | None = None) -> str:
    return json.dumps(sanitize_surrogates(obj), ensure_ascii=False, indent=indent)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                yield line_no, json.loads(line)
            except Exception:
                continue


def get_tid(rec: dict[str, Any]) -> str:
    return str(rec.get("source_traj_id") or "")


def get_step_idx(rec: dict[str, Any]) -> int:
    try:
        return int(rec.get("step_idx") or 0)
    except Exception:
        return 0


def final_current_date(rec: dict[str, Any]) -> str:
    rw = rec.get("rewrite") or {}
    meta = rec.get("meta") or {}
    return (
        str(rw.get("inferred_date") or "").strip()
        or str(meta.get("time_insensitive_inferred_date") or "").strip()
        or "2026-05-01"
    )


def find_latest_success_intervals(search_path: Path, max_trajectories: int = 0):
    """Return intervals for each trajectory's latest successful final attempt."""
    attempt_start_by_tid: dict[str, int] = {}
    latest: dict[str, dict[str, Any]] = {}
    bad_json = 0
    line_count = 0

    with search_path.open(encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line_count = line_no
            try:
                rec = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            tid = get_tid(rec)
            if not tid:
                continue
            if get_step_idx(rec) == 1:
                attempt_start_by_tid[tid] = line_no
            rw = rec.get("rewrite")
            if rec.get("is_final") and rw is not None:
                start_line = attempt_start_by_tid.get(tid, line_no)
                latest[tid] = {
                    "tid": tid,
                    "start_line": start_line,
                    "end_line": line_no,
                    "current_date": final_current_date(rec),
                    "source": rec.get("source"),
                    "total_steps": rec.get("total_steps"),
                    "final_step_quality": (rw.get("step_quality") or "").strip().lower(),
                }

    intervals = sorted(latest.values(), key=lambda x: (x["start_line"], x["end_line"]))
    if max_trajectories and max_trajectories > 0:
        intervals = intervals[:max_trajectories]
    return intervals, {"search_lines": line_count, "bad_json_lines": bad_json}


def line_interval_index(intervals: list[dict[str, Any]]):
    return [(int(x["start_line"]), int(x["end_line"]), x["tid"]) for x in intervals]


def collect_needed_observation_keys(search_path: Path, intervals: list[dict[str, Any]]):
    indexed = line_interval_index(intervals)
    needed: set[str] = set()
    expected_steps = 0
    idx = 0
    for line_no, rec in iter_jsonl(search_path):
        while idx < len(indexed) and line_no > indexed[idx][1]:
            idx += 1
        if idx >= len(indexed):
            break
        start, _end, tid = indexed[idx]
        if line_no < start:
            continue
        step_idx = get_step_idx(rec)
        expected_steps += 1
        if step_idx > 1:
            needed.add(f"{tid}\t{step_idx - 1}")
    return needed, expected_steps


def open_obs_db(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute(
        "CREATE TABLE obs (tid TEXT NOT NULL, step_idx INTEGER NOT NULL, observation TEXT NOT NULL, "
        "PRIMARY KEY (tid, step_idx))"
    )
    return con


def build_observation_db(
    skeleton_paths: list[Path],
    needed: set[str],
    db_path: Path,
    progress_every: int = 100000,
):
    con = open_obs_db(db_path)
    found = 0
    scanned = 0
    batch: list[tuple[str, int, str]] = []
    t0 = time.time()

    for path in skeleton_paths:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                scanned += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                tid = get_tid(rec)
                step_idx = get_step_idx(rec)
                key = f"{tid}\t{step_idx}"
                if key not in needed:
                    continue
                batch.append((tid, step_idx, rec.get("source_observation_after") or ""))
                found += 1
                if len(batch) >= 1000:
                    con.executemany("INSERT OR REPLACE INTO obs VALUES (?, ?, ?)", batch)
                    batch.clear()
                if progress_every and found and found % progress_every == 0:
                    elapsed = max(1e-6, time.time() - t0)
                    print(
                        f"[obs] found={found}/{len(needed)} scanned={scanned} "
                        f"rate={found / elapsed:.1f}/s",
                        flush=True,
                    )
                if found == len(needed):
                    break
        if found == len(needed):
            break

    if batch:
        con.executemany("INSERT OR REPLACE INTO obs VALUES (?, ?, ?)", batch)
    con.commit()
    return con, {"needed_observations": len(needed), "found_observations": found, "skeleton_rows_scanned": scanned}


def load_observations_for_tid(con: sqlite3.Connection, tid: str) -> dict[int, str]:
    cur = con.execute("SELECT step_idx, observation FROM obs WHERE tid = ?", (tid,))
    return {int(step): obs for step, obs in cur.fetchall()}


def format_recent_actions(records: list[dict[str, Any]], max_chars: int = 0) -> str:
    action_records = [record for record in records if record.get("tool_call")][-10:]
    if not action_records:
        return "No previous tool actions."
    entries = []
    for record in action_records:
        action = json_dumps(record.get("tool_call", {}))
        description = (record.get("action_description") or "").strip()
        step_id = record.get("step", "?")
        if description:
            entries.append(f"Step {step_id}, S: {action} \u2014 {description}")
        else:
            entries.append(f"Step {step_id}, S: {action}")
    return compact_observation_text("\n\n".join(entries), max_chars=max_chars)


def audit_subset(rewrite: dict[str, Any]) -> dict[str, Any]:
    return {k: rewrite[k] for k in AUDIT_KEYS if k in rewrite}


def target_for_record(rec: dict[str, Any], *, is_first: bool) -> dict[str, Any]:
    rw = rec.get("rewrite") or {}
    if is_first:
        return {
            "task_analysis": rw.get("task_analysis") or "",
            "remaining_subgoals": rw.get("remaining_subgoals") or [],
            "next_action_rationale": rw.get("next_action_rationale") or "",
            "tool_call": rec.get("tool_call"),
            "action_description": rw.get("action_description") or "",
        }

    if rec.get("is_final"):
        return {
            "this_turn": rw.get("this_turn") or "",
            "cumulative_progress": rw.get("cumulative_progress") or "",
            "remaining_subgoals": rw.get("remaining_subgoals") or [],
            "next_action_rationale": (
                rw.get("next_action_rationale")
                or "Use the accumulated evidence to provide the final answer."
            ),
            "tool_call": None,
            "action_description": None,
            "answer": rw.get("answer") or rec.get("answer") or "",
        }

    answer = rw.get("answer") or None
    return {
        "this_turn": rw.get("this_turn") or "",
        "cumulative_progress": rw.get("cumulative_progress") or "",
        "remaining_subgoals": rw.get("remaining_subgoals") or [],
        "next_action_rationale": rw.get("next_action_rationale") or "",
        "tool_call": None if answer else rec.get("tool_call"),
        "action_description": None if answer else (rw.get("action_description") or ""),
        "answer": answer,
    }


def build_prompt(
    *,
    rec: dict[str, Any],
    prompt_kind: str,
    current_date: str,
    last_report: str,
    last_observation: str,
    recent_actions: str,
    max_observation_chars: int,
) -> str:
    payload = {
        "current_date": current_date,
        "task": rec.get("task") or "",
        "task_with_web": rec.get("task") or "",
        "initial_url": "",
        "current_url": "",
        "current_title": "",
        "viewport_width": 1080,
        "viewport_height": 1440,
        "last_report": last_report,
        "last_observation": compact_observation_text(last_observation, max_chars=max_observation_chars),
        "recent_actions": recent_actions,
        "tabs_text": "",
    }
    text = get_user_prompt_template("hybrid_routed", prompt_kind, "S").format(**payload)
    return sanitize_surrogates(text)


def write_training_rows(
    *,
    search_path: Path,
    out_path: Path,
    obs_con: sqlite3.Connection,
    intervals: list[dict[str, Any]],
    max_observation_chars: int,
    max_action_history_chars: int,
    include_debug_fields: bool,
    progress_every: int,
):
    indexed = line_interval_index(intervals)
    info_by_tid = {x["tid"]: x for x in intervals}
    system_prompt = get_system_prompt("hybrid_routed", "S")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "rows_written": 0,
        "trajectories_written": 0,
        "missing_observations": 0,
        "prompt_kind_counts": Counter(),
        "step_quality_counts": Counter(),
        "final_step_quality_counts": Counter(),
        "answer_rows": 0,
    }

    idx = 0
    active_tid = None
    current_date = "2026-05-01"
    obs_by_step: dict[int, str] = {}
    last_report = "No previous report."
    recent_records: list[dict[str, Any]] = []
    first_in_interval = True
    t0 = time.time()

    with out_path.open("w", encoding="utf-8", buffering=1024 * 1024) as fout:
        for line_no, rec in iter_jsonl(search_path):
            while idx < len(indexed) and line_no > indexed[idx][1]:
                idx += 1
                active_tid = None
            if idx >= len(indexed):
                break
            start, _end, tid = indexed[idx]
            if line_no < start:
                continue

            if active_tid != tid:
                active_tid = tid
                first_in_interval = True
                current_date = str(info_by_tid[tid]["current_date"])
                obs_by_step = load_observations_for_tid(obs_con, tid)
                last_report = "No previous report."
                recent_records = []
                stats["trajectories_written"] += 1

            step_idx = get_step_idx(rec)
            prompt_kind = "initial" if first_in_interval else "turn"
            last_observation = ""
            if not first_in_interval:
                last_observation = obs_by_step.get(step_idx - 1, "")
                if not last_observation:
                    stats["missing_observations"] += 1
            recent_actions = format_recent_actions(recent_records, max_chars=max_action_history_chars)
            prompt_text = build_prompt(
                rec=rec,
                prompt_kind=prompt_kind,
                current_date=current_date,
                last_report=last_report,
                last_observation=last_observation,
                recent_actions=recent_actions,
                max_observation_chars=max_observation_chars,
            )
            target = target_for_record(rec, is_first=first_in_interval)
            target = sanitize_surrogates(target)
            raw_response = json_dumps(target, indent=2)

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": prompt_text}]},
                {"role": "assistant", "content": [{"type": "text", "text": raw_response}]},
            ]

            row = {
                "source": rec.get("source"),
                "source_traj_id": tid,
                "step_idx": step_idx,
                "total_steps": rec.get("total_steps"),
                "is_final": bool(rec.get("is_final")),
                "language": rec.get("language"),
                "task": rec.get("task"),
                "agent_mode": "hybrid_routed",
                "active_mode": "S",
                "prompt_kind": prompt_kind,
                "current_date": current_date,
                "messages": messages,
                "report_parsed": target,
                "tool_call": target.get("tool_call"),
                "answer": target.get("answer"),
                "action_description": target.get("action_description"),
                "meta": {
                    "selected_attempt_start_line": start,
                    "selected_attempt_end_line": indexed[idx][1],
                    "rewrite_audit": audit_subset(rec.get("rewrite") or {}),
                    "source_meta": rec.get("meta") or {},
                },
            }
            if include_debug_fields:
                row["system_prompt"] = system_prompt
                row["prompt_text"] = prompt_text
                row["raw_response"] = raw_response

            fout.write(json_dumps(row) + "\n")

            rw = rec.get("rewrite") or {}
            quality = (rw.get("step_quality") or "").strip().lower() or "<missing>"
            stats["step_quality_counts"][quality] += 1
            stats["prompt_kind_counts"][prompt_kind] += 1
            if rec.get("is_final"):
                stats["final_step_quality_counts"][quality] += 1
            if target.get("answer"):
                stats["answer_rows"] += 1
            stats["rows_written"] += 1

            last_report = raw_response
            if target.get("tool_call"):
                recent_records.append(
                    {
                        "step": step_idx,
                        "tool_call": target.get("tool_call"),
                        "action_description": target.get("action_description") or "",
                        "active_mode_before": "S",
                    }
                )
            first_in_interval = False

            if progress_every and stats["rows_written"] % progress_every == 0:
                elapsed = max(1e-6, time.time() - t0)
                print(
                    f"[write] rows={stats['rows_written']} trajs={stats['trajectories_written']} "
                    f"rate={stats['rows_written'] / elapsed:.1f}/s",
                    flush=True,
                )

    stats["prompt_kind_counts"] = dict(stats["prompt_kind_counts"])
    stats["step_quality_counts"] = dict(stats["step_quality_counts"])
    stats["final_step_quality_counts"] = dict(stats["final_step_quality_counts"])
    return stats


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--search-jsonl",
        default=str(ROOT / "data/hybrid_search_deepseek_v4_flash_full.jsonl"),
        help="Append-only Search stage-2 rewrite JSONL.",
    )
    ap.add_argument(
        "--skeletons",
        default=",".join(
            [
                str(ROOT / "data/openseeker_skeleton_correct.jsonl"),
                str(ROOT / "data/redsearcher_skeleton_correct.jsonl"),
            ]
        ),
        help="Comma-separated skeleton JSONL files.",
    )
    ap.add_argument(
        "--out-path",
        default=str(ROOT / "data/hybrid_search_deepseek_v4_flash_hybrid_routed_s_train.jsonl"),
    )
    ap.add_argument(
        "--obs-cache",
        default=str(ROOT / "data/.hybrid_search_routed_s_obs_cache.sqlite"),
    )
    ap.add_argument("--max-trajectories", type=int, default=0)
    ap.add_argument("--max-observation-chars", type=int, default=0)
    ap.add_argument("--max-action-history-chars", type=int, default=0)
    ap.add_argument("--progress-every", type=int, default=50000)
    ap.add_argument("--no-debug-fields", action="store_true")
    ap.add_argument("--keep-obs-cache", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    search_path = Path(args.search_jsonl)
    skeleton_paths = [Path(x.strip()) for x in args.skeletons.split(",") if x.strip()]
    out_path = Path(args.out_path)
    obs_cache = Path(args.obs_cache)

    t0 = time.time()
    print(f"[scan] search={search_path}", flush=True)
    intervals, scan_stats = find_latest_success_intervals(search_path, args.max_trajectories)
    print(
        f"[scan] selected_trajs={len(intervals)} search_lines={scan_stats['search_lines']} "
        f"bad_json={scan_stats['bad_json_lines']}",
        flush=True,
    )

    needed, expected_steps = collect_needed_observation_keys(search_path, intervals)
    print(f"[plan] expected_rows={expected_steps} needed_observations={len(needed)}", flush=True)

    print(f"[obs] cache={obs_cache}", flush=True)
    obs_con, obs_stats = build_observation_db(skeleton_paths, needed, obs_cache, args.progress_every)

    print(f"[write] out={out_path}", flush=True)
    write_stats = write_training_rows(
        search_path=search_path,
        out_path=out_path,
        obs_con=obs_con,
        intervals=intervals,
        max_observation_chars=args.max_observation_chars,
        max_action_history_chars=args.max_action_history_chars,
        include_debug_fields=not args.no_debug_fields,
        progress_every=args.progress_every,
    )
    obs_con.close()

    if not args.keep_obs_cache:
        try:
            obs_cache.unlink()
        except FileNotFoundError:
            pass

    stats = {
        "out_path": str(out_path),
        "search_jsonl": str(search_path),
        "skeletons": [str(p) for p in skeleton_paths],
        "max_trajectories": args.max_trajectories,
        "max_observation_chars": args.max_observation_chars,
        "max_action_history_chars": args.max_action_history_chars,
        "include_debug_fields": not args.no_debug_fields,
        "elapsed_sec": round(time.time() - t0, 3),
        **scan_stats,
        **obs_stats,
        **write_stats,
        "out_size_bytes": out_path.stat().st_size if out_path.exists() else 0,
    }
    stats_path = out_path.with_suffix(out_path.suffix + ".stats.json")
    stats_path.write_text(json_dumps(stats, indent=2), encoding="utf-8")
    print("[done]", json_dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
