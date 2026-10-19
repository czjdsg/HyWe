#!/usr/bin/env python3
"""HyWeCollector Stage 1 driver: explore -> value-filter -> search probe, per site.

For every site in the pool, independently and concurrently:

  A  explore       an operation-based explorer (explorer_agent_v8: operator +
                   fact extractor, both vision models) drives a real browser on
                   the site and records answer-shaped facts into
                   <out>/<domain>/result.json.
  VF value filter  (optional, --value-filter) one LLM pass per site keeps only
                   concrete user-value facts (prices, counts, readings ...) and
                   drops scaffolding / UI narration / near-duplicates.
  B  probe         fact_prober_v2 turns each fact into a no-leak question,
                   runs the search-based agent (run_hybrid_web_agent.py
                   --agent-mode search) under a fixed budget, and labels the
                   fact S_REACHABLE / O_EXCLUSIVE -> <out>/<domain>/fact_tags.json
                   (the search transcript is kept).

Afterwards run refilter_pool.py and heuristic_label.py over <out> to obtain the
final stable fact pool used to abstract the task prototypes.

Per-site hard timeouts (PIPE_T_A / PIPE_T_B, seconds) keep one hung site from
stalling the run; on an A timeout the facts already checkpointed are salvaged.

Progress lines on stdout:
  PIPE <domain> A_done facts=N elapsed=S
  PIPE <domain> B_done S=n O=n harvested=n
  PIPE <domain> SKIP <stage> <reason>
  PIPE_STATS done=N/Total inflight A=.. B=..
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path

COLLECTOR = Path(__file__).resolve().parent
sys.path.insert(0, str(COLLECTOR))

from synth_driver import run_one  # noqa: E402  (stage A explorer, explore_only path)
import value_filter               # noqa: E402  (A->B user-value fact filter)

T_A = int(os.environ.get("PIPE_T_A", 30 * 60))
T_B = int(os.environ.get("PIPE_T_B", 90 * 60))

_jsonl_lock = threading.Lock()


def log(msg: str):
    print(msg, flush=True)


def append_jsonl(path: Path, rows: list[dict]):
    with _jsonl_lock:
        with path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


async def run_proc(cmd: list[str], timeout: int, log_path: Path) -> int:
    """Run a subprocess with hard timeout; stream output to its own log."""
    with log_path.open("ab") as lf:
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=lf, stderr=asyncio.subprocess.STDOUT)
        try:
            return await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            return -9


class Stage1:
    def __init__(self, pool: list[dict], out: Path, args):
        self.pool = pool
        self.out = out
        self.args = args
        self.sem_a = asyncio.Semaphore(args.sem_a)
        self.sem_b = asyncio.Semaphore(args.sem_b)
        self.inflight = {"A": 0, "B": 0}
        self.done = 0
        self.state_jsonl = out / "pipeline_state.jsonl"
        (out / "shims").mkdir(parents=True, exist_ok=True)
        (out / "_logs").mkdir(parents=True, exist_ok=True)

    def _state(self, domain: str, stage: str, **kw):
        append_jsonl(self.state_jsonl,
                     [{"domain": domain, "stage": stage,
                       "ts": int(time.time()), **kw}])

    # ---------------- stage A ----------------
    async def stage_a(self, site: dict) -> dict | None:
        domain = site.get("domain", "?").replace("/", "_")
        site_dir = self.out / domain
        if self.args.skip_existing and (site_dir / "result.json").is_file():
            res = json.loads((site_dir / "result.json").read_text())
            n = len(((res.get("kb") or {}).get("facts")) or {})
            log(f"PIPE {domain} A_skip facts={n} (existing)")
            return res
        t0 = time.time()
        async with self.sem_a:
            self.inflight["A"] += 1
            try:
                res = await asyncio.wait_for(
                    run_one(site, str(self.out),
                            max_steps=self.args.max_steps,
                            use_vision=True,
                            phase_budgets={},
                            min_phases_before_record=3,
                            explore_only=True,
                            agent_version=self.args.agent),
                    timeout=T_A)
            except asyncio.TimeoutError:
                # Per-site wall-clock cap hit. Salvage the facts already extracted
                # (result.json is checkpointed every step) so a slow-but-productive
                # site still goes to B instead of being thrown away.
                ck = site_dir / "result.json"
                if ck.is_file():
                    try:
                        res = json.loads(ck.read_text())
                        n = len(((res.get("kb") or {}).get("facts")) or {})
                    except Exception:
                        res, n = None, 0
                    if res is not None and n > 0:
                        log(f"PIPE {domain} A_timeout_salvage facts={n} ({T_A}s cap)")
                        self._state(domain, "A", ok=True, facts=n)
                        return res
                log(f"PIPE {domain} SKIP A timeout({T_A}s, 0 facts)")
                self._state(domain, "A", ok=False, err="timeout")
                return None
            except Exception as exc:
                log(f"PIPE {domain} SKIP A {type(exc).__name__}: {str(exc)[:90]}")
                self._state(domain, "A", ok=False, err=str(exc)[:200])
                return None
            finally:
                self.inflight["A"] -= 1
        n = len(((res.get("kb") or {}).get("facts")) or {})
        log(f"PIPE {domain} A_done facts={n} elapsed={time.time()-t0:.0f}s")
        self._state(domain, "A", ok=True, facts=n)
        return res

    # ---------------- stage B ----------------
    async def stage_b(self, domain: str) -> dict | None:
        ft = self.out / domain / "fact_tags.json"
        if self.args.skip_existing and ft.is_file():
            try:
                d = json.loads(ft.read_text())
                c = d.get("counts") or {}
                if not c.get("errors"):
                    log(f"PIPE {domain} B_skip (cached) "
                        f"S={c.get('s_reachable',0)} O={c.get('o_exclusive',0)}")
                    return d
            except Exception:
                pass
        site_dir = self.out / domain
        # fact_prober_v2 walks every site directory under --run, so give it a
        # one-site view through a symlink shim.
        shim = self.out / "shims" / domain
        shim.mkdir(parents=True, exist_ok=True)
        link = shim / domain
        if not link.exists():
            link.symlink_to(site_dir.resolve())
        cmd = [sys.executable, str(COLLECTOR / "fact_prober_v2.py"),
               "--run", str(shim), "--conc", str(self.args.b_conc)]
        async with self.sem_b:
            self.inflight["B"] += 1
            try:
                rc = await run_proc(cmd, T_B,
                                    self.out / "_logs" / f"{domain}_B.log")
            finally:
                self.inflight["B"] -= 1
        if rc != 0 or not ft.is_file():
            log(f"PIPE {domain} SKIP B rc={rc}")
            self._state(domain, "B", ok=False, rc=rc)
            return None
        d = json.loads(ft.read_text())
        c = d.get("counts") or {}
        log(f"PIPE {domain} B_done S={c.get('s_reachable',0)} "
            f"O={c.get('o_exclusive',0)} harvested={c.get('harvested',0)}")
        self._state(domain, "B", ok=True, **{k: c.get(k) for k in
                    ("s_reachable", "o_exclusive", "harvested") if k in c})
        return d

    # ---------------- per-site chain ----------------
    async def site_flow(self, site: dict):
        domain = site.get("domain", "?").replace("/", "_")
        try:
            res = await self.stage_a(site)
            if res is not None:
                if self.args.value_filter:
                    n_in, n_kept = await asyncio.to_thread(
                        value_filter.apply_to_kb, str(self.out / domain))
                    if n_kept >= 0:
                        log(f"PIPE {domain} VF kept {n_kept}/{n_in}")
                        res = json.loads((self.out / domain / "result.json").read_text())
                facts = ((res.get("kb") or {}).get("facts")) or {}
                if facts:
                    await self.stage_b(domain)
                else:
                    log(f"PIPE {domain} SKIP B no facts from A")
        except Exception as exc:
            log(f"PIPE {domain} SKIP flow {type(exc).__name__}: {str(exc)[:120]}")
        finally:
            self.done += 1
            log(f"PIPE_STATS done={self.done}/{len(self.pool)} "
                f"inflight A={self.inflight['A']} B={self.inflight['B']}")

    async def run(self):
        log(f"STAGE1 start: {len(self.pool)} sites  "
            f"semA={self.args.sem_a} semB={self.args.sem_b} bConc={self.args.b_conc}")
        await asyncio.gather(*(self.site_flow(s) for s in self.pool))
        log("STAGE1_ALL_DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=str(COLLECTOR / "data" / "insta_screened.pool.json"),
                    help='site pool JSON: {"all_usable": [{domain, entry_url, category, ...}]} or a list')
    ap.add_argument("--out", required=True, help="run directory (one sub-directory per site)")
    ap.add_argument("--max-steps", type=int, default=40, help="explorer step budget per site")
    ap.add_argument("--sem-a", type=int, default=16, help="concurrent explorer sites")
    ap.add_argument("--sem-b", type=int, default=8, help="concurrent probe sites")
    ap.add_argument("--b-conc", type=int, default=4, help="probe workers per site")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--value-filter", action="store_true",
                    help="between A and B keep only user-value facts (drop enum/nav/UI-state/near-dup)")
    ap.add_argument("--agent", choices=("v7", "v8"), default="v8",
                    help="v8 = decoupled operator + fact extractor (used for the paper); v7 = single agent")
    args = ap.parse_args()

    os.environ["SYNTH3_AGENT"] = args.agent
    obj = json.load(open(args.pool))
    pool = obj if isinstance(obj, list) else obj["all_usable"]
    if args.limit:
        pool = pool[:args.limit]
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(Stage1(pool, out, args).run())


if __name__ == "__main__":
    main()
