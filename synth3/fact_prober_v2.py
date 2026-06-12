#!/usr/bin/env python3
"""v7 Stage B (production-grade): probe each O-side fact with the same
search-mode agent that powers hybrid_routed rollout.

Architecture (vs the original fact_prober.py which rolled its own 5-step
mini-agent):
  1. CLASSIFY        same code-side filter as before (chrome / captcha / visual)
  2. QUESTION-FORM   per fact, deepseek-v4-pro turns the value into a
                     no-leak question
  3. SEARCH-AGENT    one batched invocation of HybridWebAgent's
                     `--agent-mode search` over the whole probe worklist
                     (32 workers, browser-less, loop detection + reports)
  4. JUDGE           per fact, deepseek-v4-pro compares the agent's
                     prediction to the live fact value -> tag + harvest

The fact_tags.json output schema is intentionally identical to v1's, so
Stage C / viz consume it without changes. The `s_transcript` field now
carries the production records (step, tool, action_description, report,
observation, current_url) rather than v1's home-grown trace.

Usage:
  source ../env.local.sh
  python3 fact_prober_v2.py --run /tmp/synth7_base \\
      [--conc 16] [--max-probes 12] [--limit-sites 0]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ----- reuse classifier + LLM helpers + Q-form + judge from v1 --------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fact_prober import (  # noqa: E402
    classify_fact, richness, call_llm, extract_json,
    question_form, judge,
    FLASH_BASE, FLASH_MODEL, API_KEY,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
HYBRID_AGENT = Path(os.environ.get(
    "HYBRID_AGENT_ENTRY",
    str(_REPO_ROOT / "inference" / "run_hybrid_web_agent.py"),
))
# search-agent endpoint defaults — overridable via env, same names as v1
SAGENT_BASE = os.environ.get("PROBE_AGENT_BASE", "http://localhost:8000/v1")
SAGENT_MODEL = os.environ.get("PROBE_AGENT_MODEL", "qwen3.7-max")
SAGENT_KEY = os.environ.get("PROBE_AGENT_KEY", API_KEY)
SUMMARY_BASE = os.environ.get(
    "SUMMARY_BASE_URL", "http://localhost:8000/v1")
SUMMARY_MODEL = os.environ.get("SUMMARY_MODEL", "deepseek-v4-flash")

# probe budget: search-mode is more efficient than v1's manual loop,
# but we still want a small ceiling — most facts settle in 2-4 turns.
PROBE_MAX_ITER = int(os.environ.get("PROBE_MAX_ITER", "8"))


# ---------------- worklist build ----------------
def build_worklist(run: Path, max_probes: int, limit_sites: int):
    """Returns site_meta dict + flat list of work dicts identical in shape
    to v1's, plus a sites list for output traversal."""
    site_dirs = sorted(d for d in run.iterdir()
                       if d.is_dir() and (d / "result.json").is_file())
    if limit_sites:
        site_dirs = site_dirs[:limit_sites]
    site_meta = {}
    work = []
    for sd in site_dirs:
        res = json.loads((sd / "result.json").read_text())
        facts = ((res.get("kb") or {}).get("facts")) or {}
        domain = sd.name
        entry_url = (res.get("site") or {}).get("entry_url") or ""
        rows = []
        for k, v in facts.items():
            cls = classify_fact(k, v.get("value") or "")
            rows.append({
                "domain": domain, "entry_url": entry_url,
                "key": k, "value": v.get("value") or "",
                "step": v.get("step"), "mode": v.get("mode"),
                "probe": cls, "tag": None,
            })
        probe_rows = [r for r in rows if r["probe"] == "probe"]
        probe_rows.sort(key=lambda r: -richness(r["value"]))
        for r in probe_rows[max_probes:]:
            r["probe"] = "skip_capped"
        for r in rows:
            if r["probe"] == "auto_o":
                r["tag"] = "O_EXCLUSIVE"
                r["reason"] = ("visual fact — search cannot retrieve "
                               "rendered imagery")
                r["harvested"] = []
        site_meta[domain] = {"rows": rows, "entry_url": entry_url}
        work.extend(r for r in rows if r["probe"] == "probe")
    return site_meta, work, site_dirs


# ---------------- question-form pass (parallel) ----------------
def qform_pass(work: list[dict], conc: int) -> None:
    """In place: populate w['question'] for each probe work item."""
    def _one(w):
        try:
            return w, question_form(w["domain"], w["entry_url"],
                                    w["key"], w["value"]), None
        except Exception as exc:
            return w, None, f"qform: {type(exc).__name__}: {exc}"
    n_done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(_one, w) for w in work]
        for fut in cf.as_completed(futs):
            w, q, err = fut.result()
            if q:
                w["question"] = q
            else:
                w["question"] = None
                w["probe"] = "error"
                w["tag"] = "O_EXCLUSIVE"  # conservative: can't probe → O-only
                w["reason"] = err
            n_done += 1
            if n_done % 25 == 0 or n_done == len(work):
                print(f"  [qform] {n_done}/{len(work)}  "
                      f"rate={n_done/max(time.time()-t0,0.01):.1f}/s", flush=True)


# ---------------- search-agent invocation (single batched call) ----------------
def safe_task_id(domain: str, key: str) -> str:
    """Match run_hybrid's safe_task_id idea — short alphanumerics + dashes."""
    raw = f"{domain}__{key}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw)[:100]


def run_search_agent(work: list[dict], out_dir: Path, conc: int) -> dict[str, dict]:
    """Spawn run_hybrid_web_agent.py once on the whole probe worklist.
    Returns task_id -> result_json dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / "input.jsonl"
    out_path = out_dir / "results.jsonl"
    log_path = out_dir / "search_agent.log"
    # If a previous invocation already produced all per-task result.json
    # under _probe_out, skip the subprocess (idempotent resume).
    if input_path.is_file():
        try:
            cached_ids = set()
            for line in input_path.open():
                cached_ids.add(json.loads(line)["task_id"])
            if cached_ids and all((out_dir / tid / "result.json").is_file()
                                  for tid in cached_ids):
                print(f"  cached: {len(cached_ids)} probe results already on disk, skipping search agent", flush=True)
                results = {}
                for tid in cached_ids:
                    try:
                        results[tid] = json.loads((out_dir / tid / "result.json").read_text())
                    except Exception:
                        pass
                id_to_idx = {tid: i for i, tid in enumerate(sorted(cached_ids))}
                return results, id_to_idx
        except Exception:
            pass

    # task_id <-> work index map (so we can stitch results back even if some
    # tasks fail to write a result.json)
    id_to_idx: dict[str, int] = {}
    with input_path.open("w") as f:
        for i, w in enumerate(work):
            if not w.get("question"):
                continue
            tid = safe_task_id(w["domain"], w["key"])
            id_to_idx[tid] = i
            f.write(json.dumps({
                "task_id": tid,
                "question": w["question"],
                "web": "",                       # search mode → no start URL
                "dataset": "synth7_probe",
                "split": "probe",
                "_probe_meta": {"domain": w["domain"], "key": w["key"]},
            }, ensure_ascii=False) + "\n")

    cmd = [
        sys.executable, str(HYBRID_AGENT),
        "--input-path", str(input_path),
        "--output-dir", str(out_dir),
        "--output-path", str(out_path),
        "--agent-mode", "search",
        "--max-workers", str(conc),
        "--max-task-attempts", "1",
        "--max-iter", str(PROBE_MAX_ITER),
        "--max-tokens", "12000",
        "--base-url", SAGENT_BASE,
        "--model", SAGENT_MODEL,
        "--api-key", SAGENT_KEY,
        "--api-timeout", "300",
        "--router-base-url", SAGENT_BASE,
        "--router-model", SAGENT_MODEL,
        "--router-api-key", SAGENT_KEY,
        "--router-timeout", "300",
        "--router-max-tokens", "4096",
        "--summary-base-url", SUMMARY_BASE,
        "--summary-model", SUMMARY_MODEL,
        "--summary-api-key", SAGENT_KEY,
        "--summary-timeout", "300",
        "--visit-summary-max-tokens", "4000",
        "--no-visit-images",
        "--visit-max-images", "0",
        "--code-sandbox-base-url", os.environ.get(
            "PROBE_SANDBOX_BASE_URL", "http://127.0.0.1:8080"),
    ]
    _probe_tok = os.environ.get("PROBE_TOKENIZER_PATH", "")
    if _probe_tok:
        cmd += ["--tokenizer-path", _probe_tok,
                "--summary-tokenizer-path", _probe_tok]
    # Inject local pool env keys so the search agent's serper/jina tools work.
    # HARD requirement: both serper and jina MUST go through the local pools
    # at 127.0.0.1:8765 / :8766 — never the public endpoints (rate-limited /
    # billable / unreliable). Read env.local.sh from the HybridWebAgent dir
    # and refuse to launch if the local pool URLs are not present.
    child_env = os.environ.copy()
    # env.local.sh lives at repo root (next to inference/, evaluation/, ...).
    env_sh = _REPO_ROOT / "env.local.sh"
    if env_sh.is_file():
        for line in env_sh.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*"
                         r"['\"]?([^'\"\n]+)['\"]?\s*$", line)
            if m:
                # Override any inherited value — env.local.sh is the source
                # of truth so a stale public URL from a parent shell cannot
                # leak into the child.
                child_env[m.group(1)] = m.group(2)
    required = {
        "SERPER_BASE_URL": "http://127.0.0.1:8765",
        "JINA_READER_BASE_URL": "http://127.0.0.1:8766",
        "SERPER_KEY_ID": None,
        "JINA_API_KEY": None,
    }
    bad = []
    for k, expected in required.items():
        v = child_env.get(k)
        if not v:
            bad.append(f"{k} missing")
        elif expected and not v.startswith("http://127.0.0.1:"):
            bad.append(f"{k}={v!r} is not the local pool")
    if bad:
        raise RuntimeError(
            "search agent cannot run: local Serper/Jina pools required. "
            f"Issues: {bad}. Fix env.local.sh at {env_sh}.")
    print(f"  pools: serper={child_env['SERPER_BASE_URL']} "
          f"jina={child_env['JINA_READER_BASE_URL']}", flush=True)
    print(f"  launching search agent on {len(id_to_idx)} probes  "
          f"(max-iter={PROBE_MAX_ITER}, workers={conc})", flush=True)
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                cwd=str(HYBRID_AGENT.parent),
                                env=child_env)
        proc.wait()
    print(f"  search agent exit={proc.returncode}  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

    # Read each per-task result.json (more complete than results.jsonl)
    results: dict[str, dict] = {}
    for tid in id_to_idx:
        rp = out_dir / tid / "result.json"
        if rp.is_file():
            try:
                results[tid] = json.loads(rp.read_text())
            except Exception:
                pass
    print(f"  collected {len(results)}/{len(id_to_idx)} results", flush=True)
    return results, id_to_idx


# ---------------- transcript extraction ----------------
def _trim(s, n):
    if not s: return s
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def extract_probe(result: dict) -> dict:
    """Map a run_hybrid result.json to the {answer, found, steps, transcript}
    shape that judge() expects, with v2-shaped trace entries.

    Per-step trace entry:
      {step, tool, action, args, action_description, current_url,
       cumulative_progress, next_action_rationale, observation_head, prompt_head}
    """
    records = result.get("records") or []
    answer = (result.get("prediction") or "").strip()
    found = bool(answer)
    transcript = []
    for r in records:
        rp = r.get("report_parsed") or {}
        if not isinstance(rp, dict):
            rp = {}
        tc = r.get("tool_call") or {}
        if not isinstance(tc, dict):
            tc = {}
        rp_tc = rp.get("tool_call") or {}
        if not isinstance(rp_tc, dict):
            rp_tc = {}
        # tool name lives in tool_call.name or tool_call.function.name
        tool = tc.get("name")
        if not tool and isinstance(tc.get("function"), dict):
            tool = tc["function"].get("name")
        if not tool:
            tool = rp_tc.get("name")
        args = tc.get("arguments") or rp_tc.get("arguments")
        transcript.append({
            "step": r.get("step"),
            "tool": tool,
            "args": args,
            "action_description": r.get("action_description")
                                  or rp.get("action_description"),
            "current_url": r.get("current_url"),
            "cumulative_progress": _trim(rp.get("cumulative_progress"), 800),
            "next_action_rationale": _trim(rp.get("next_action_rationale"), 400),
            "observation_head": _trim(r.get("observation"), 1500),
        })
    return {
        "answer": answer,
        "found": found,
        "steps": len(records),
        "transcript": transcript,
        "termination": result.get("termination"),
    }


# ---------------- judge pass (parallel) ----------------
def judge_pass(work: list[dict], results: dict[str, dict],
               id_to_idx: dict[str, int], conc: int) -> None:
    """In place: populate w['tag','reason','harvested', plus probe extras]."""
    def _one(w):
        tid = safe_task_id(w["domain"], w["key"])
        res = results.get(tid)
        if not res:
            return w, None, "search agent returned no result"
        probe = extract_probe(res)
        try:
            v = judge(w["domain"], w["question"], w["value"], probe)
        except Exception as exc:
            return w, probe, f"judge: {type(exc).__name__}: {exc}"
        return w, {**probe, **v}, None

    n_done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(_one, w) for w in work
                if w.get("probe") == "probe" and w.get("question")]
        for fut in cf.as_completed(futs):
            w, merged, err = fut.result()
            if err and not merged:
                w["probe"] = "error"
                w["tag"] = "O_EXCLUSIVE"
                w["reason"] = err
                w["harvested"] = []
            else:
                w["probe"] = "probed"
                w["s_answer"] = merged.get("answer") or ""
                w["s_found"] = merged.get("found", False)
                w["s_steps"] = merged.get("steps", 0)
                w["s_transcript"] = merged.get("transcript") or []
                w["s_termination"] = merged.get("termination")
                w["tag"] = merged.get("tag") or "O_EXCLUSIVE"
                w["reason"] = merged.get("reason") or ""
                w["harvested"] = merged.get("harvested") or []
                if err:
                    w["reason"] = (w["reason"] + " | " + err).strip(" |")
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t0
                print(f"  [judge] {n_done}  rate={n_done/max(el,0.01):.1f}/s", flush=True)


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--conc", type=int, default=16,
                    help="parallel workers for the search agent batch + "
                         "the Q-form/judge LLM threadpools")
    ap.add_argument("--max-probes", type=int, default=12)
    ap.add_argument("--limit-sites", type=int, default=0)
    ap.add_argument("--keep-search-out", action="store_true",
                    help="keep the intermediate per-fact result dirs "
                         "(_probe_out/) instead of leaving only fact_tags.json")
    args = ap.parse_args()

    run = Path(args.run)
    site_meta, work, site_dirs = build_worklist(
        run, args.max_probes, args.limit_sites)
    n_facts_total = sum(len(m["rows"]) for m in site_meta.values())
    print(f"sites={len(site_dirs)}  total_facts={n_facts_total}  "
          f"to_probe={len(work)}", flush=True)
    t0 = time.time()

    if not work:
        # No probeable facts — still fall through to stage 4 so every site
        # gets its fact_tags.json (auto_o rows are already tagged O_EXCLUSIVE
        # in build_worklist and can power single-side tasks downstream).
        print("nothing to probe — writing tags only.", flush=True)
    else:
        print(f"[stage 1] Q-form pass", flush=True)
        qform_pass(work, args.conc)

        print(f"[stage 2] search agent (production mode)", flush=True)
        out_dir = run / "_probe_out"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        results, id_to_idx = run_search_agent(work, out_dir, args.conc)

        print(f"[stage 3] judge", flush=True)
        judge_pass(work, results, id_to_idx, args.conc)

    print(f"[stage 4] write fact_tags.json (per site)", flush=True)
    idx_path = run / "fact_tags_index.jsonl"
    with idx_path.open("w") as idx_f:
        for domain, meta in site_meta.items():
            rows = meta["rows"]
            counts = {
                "total": len(rows),
                "probed": sum(1 for r in rows if r.get("probe") == "probed"),
                "s_reachable": sum(1 for r in rows if r.get("tag") == "S_REACHABLE"),
                "o_exclusive": sum(1 for r in rows if r.get("tag") == "O_EXCLUSIVE"),
                "auto_o": sum(1 for r in rows if r.get("probe") == "auto_o"),
                "skipped": sum(1 for r in rows
                               if (r.get("probe") or "").startswith("skip")),
                "errors": sum(1 for r in rows if r.get("probe") == "error"),
                "harvested": sum(len(r.get("harvested") or []) for r in rows),
            }
            out = {"domain": domain, "facts": rows, "counts": counts}
            (run / domain / "fact_tags.json").write_text(
                json.dumps(out, ensure_ascii=False, indent=1))
            idx_f.write(json.dumps(
                {"domain": domain, **counts}, ensure_ascii=False) + "\n")
            print(f"  {domain}: {counts}", flush=True)

    if work and not args.keep_search_out:
        shutil.rmtree(out_dir, ignore_errors=True)

    print(f"DONE in {time.time()-t0:.0f}s -> {idx_path}", flush=True)


if __name__ == "__main__":
    main()
