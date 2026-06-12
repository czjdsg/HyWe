"""Synth3 driver: spin up an ExplorerAgent per candidate site, dump candidates."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explorer_runtime import ExplorerRuntime
from explorer_agent import ExplorerAgent
# v8 decoupled operator/extractor agent — opt-in via SYNTH3_AGENT=v8 or
# --agent v8 flag below. v7 remains the default and is fully preserved.
try:
    from explorer_agent_v8 import ExplorerAgentV8
except Exception:
    ExplorerAgentV8 = None
import os as _os
_AGENT_ENV = (_os.environ.get("SYNTH3_AGENT") or "").strip().lower()


def _load_pool(path: str) -> list[dict]:
    obj = json.load(open(path))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "all_usable" in obj:
        return obj["all_usable"]
    raise ValueError(f"unexpected pool format in {path}")


async def run_one(site: dict, out_root: str, *, max_steps: int = 18,
                  use_vision: bool = True,
                  phase_budgets: dict | None = None,
                  min_phases_before_record: int = 3,
                  explore_only: bool = False,
                  agent_version: str | None = None) -> dict:
    domain = site.get("domain", "unknown").replace("/", "_")
    task_dir = os.path.join(out_root, domain)
    os.makedirs(task_dir, exist_ok=True)
    entry_url = site.get("entry_url") or ("https://" + site.get("domain", ""))
    runtime = ExplorerRuntime(entry_url=entry_url, task_dir=task_dir)
    av = (agent_version or _AGENT_ENV or "v7").lower()
    if av == "v8" and ExplorerAgentV8 is not None:
        Agent = ExplorerAgentV8
    else:
        Agent = ExplorerAgent
    agent = Agent(site=site, runtime=runtime,
                  max_steps=max_steps, use_vision=use_vision,
                  phase_budgets=phase_budgets,
                  min_phases_before_record=min_phases_before_record,
                  explore_only=explore_only)
    t0 = time.time()
    result = await agent.run()
    result["elapsed_s"] = round(time.time() - t0, 1)
    # persist
    with open(os.path.join(task_dir, "result.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True,
                    help="hybrid_pool.json or compact list path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 means all; otherwise only this many sites")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-vision", action="store_true")
    ap.add_argument("--filter-domain", default="",
                    help="if set, only run sites with this domain string")
    ap.add_argument("--budget-o", type=int, default=0,
                    help="per-phase budget for O (operation) phases; 0=default")
    ap.add_argument("--budget-s", type=int, default=0,
                    help="per-phase budget for S (search) phases; 0=default")
    ap.add_argument("--min-phases", type=int, default=3,
                    help="minimum phases before terminal RECORD")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip sites whose result.json already exists in --out")
    ap.add_argument("--explore-only", action="store_true",
                    help="v7 Stage A: single OPS phase, stop on budget or "
                         "saturation; no STAR/RECORD (facts tagged offline)")
    ap.add_argument("--shard", type=lambda s: tuple(int(x) for x in s.split("/")),
                    default=None,
                    help='take only shard idx/N of the pool, e.g. "1/2"')
    ap.add_argument("--agent", choices=("v7", "v8"), default=None,
                    help="v7 = single agent (legacy); v8 = decoupled "
                         "operator + fact-extractor (sonnet on facts). "
                         "Env SYNTH3_AGENT also accepted.")
    args = ap.parse_args()
    phase_budgets: dict = {}
    if args.budget_o > 0:
        phase_budgets["O"] = args.budget_o
    if args.budget_s > 0:
        phase_budgets["S"] = args.budget_s

    pool = _load_pool(args.pool)
    if args.filter_domain:
        pool = [p for p in pool if args.filter_domain in p.get("domain", "")]
    if args.skip_existing:
        # Skip sites whose result.json already exists in the output dir.
        before = len(pool)
        pool = [p for p in pool if not os.path.isfile(
            os.path.join(args.out, p.get("domain", "unknown").replace("/", "_"), "result.json"))]
        print(f"skip-existing: dropped {before - len(pool)} already-finished sites", flush=True)
    if args.shard:
        shard_idx, shard_n = args.shard
        pool = [p for i, p in enumerate(pool) if i % shard_n == shard_idx]
        print(f"shard {shard_idx}/{shard_n}: {len(pool)} sites", flush=True)
    if args.limit > 0:
        pool = pool[: args.limit]

    print(f"Driver: {len(pool)} sites, concurrency={args.concurrency}", flush=True)
    os.makedirs(args.out, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    results: list[dict] = []

    async def runner(site):
        async with sem:
            try:
                return await run_one(
                    site, args.out,
                    max_steps=args.max_steps,
                    use_vision=not args.no_vision,
                    phase_budgets=phase_budgets,
                    min_phases_before_record=args.min_phases,
                    explore_only=args.explore_only,
                    agent_version=args.agent,
                )
            except Exception as exc:
                import traceback
                tb = traceback.format_exc()
                print(f"[FATAL] {site.get('domain','?')}:\n{tb}", flush=True)
                return {"site": site, "fatal": True,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": tb,
                        "candidates": [], "history": []}

    tasks = [runner(s) for s in pool]
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        print(
            f"[done] {r.get('site',{}).get('domain','?'):28s} "
            f"steps={r.get('steps_used','?')} "
            f"candidates={len(r.get('candidates',[]))} "
            f"elapsed={r.get('elapsed_s','?')}s",
            flush=True,
        )

    # aggregate
    agg_path = os.path.join(args.out, "synth3_candidates.jsonl")
    with open(agg_path, "w") as f:
        for r in results:
            for c in r.get("candidates", []):
                row = {
                    "domain": r.get("site", {}).get("domain"),
                    "category": r.get("site", {}).get("category"),
                    "entry_url": r.get("site", {}).get("entry_url"),
                    **c,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Aggregated -> {agg_path}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
