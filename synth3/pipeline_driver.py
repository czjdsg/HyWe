#!/usr/bin/env python3
"""Per-site streaming pipeline: A explore -> B probe -> C synth -> R rollout.

Unlike the stage-barrier chain (all sites A, then all B, ...), every site
flows through the FULL pipeline independently the moment its previous stage
finishes. A site goes from live exploration to guided rollout within ~30-60
minutes, so the facts the tasks are anchored on are still fresh when the
rollout agent hits the live site (critical for top-N / trending / price
facts that drift within hours).

Stage workers (steady-state port budget ~48 per endpoint):
  A explore : in-process coroutines, SEM_A=20   (kimi-k2.6 vision @ :8002 + 20 browsers)
  B probe   : per-site fact_prober_v2 subprocess, SEM_B=8, each --conc 4
              (search qwen3.7-max + qform/judge v4-pro @ :8000 = ~32 lanes)
  C synth   : in-process threads (task_synth.synth_one), SEM_C=6 (opus @ :8002)
  R rollout : per-site run_hybrid_web_agent subprocess, SEM_R=8, each
              --max-workers 3 (O kimi @ :8002 ~24 lanes + S qwen3.7-max @ :8000)

Per-site hard timeouts so one hung site can never stall the run
(lesson from quandoo.it hanging Stage A for 9 hours):
  A 90min / B 40min / C 30min / R 150min.

Progress protocol on stdout (one line per event, Monitor-friendly):
  PIPE <domain> A_done facts=N elapsed=S
  PIPE <domain> B_done S=n O=n harvested=n
  PIPE <domain> C_done tasks=N kinds=...
  PIPE <domain> R_done finished=N answer=N
  PIPE <domain> SKIP <stage> <reason>
  PIPE_STATS done=N/Total inflight A=.. B=.. C=.. R=..
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

SYNTH3 = Path(__file__).resolve().parent
sys.path.insert(0, str(SYNTH3))

from synth_driver import run_one  # noqa: E402  (explorer, explore_only path)
import task_synth as ts_mod       # noqa: E402  (synth_one + opus infinite retry)
from explorer_runtime import coord_scale_for_model  # noqa: E402

REPO_ROOT = SYNTH3.parent
HYBRID_AGENT = os.environ.get(
    "HYBRID_AGENT_ENTRY",
    str(REPO_ROOT / "inference" / "run_hybrid_web_agent.py"),
)
# Placeholder API key for local vLLM-style endpoints that ignore the value.
# Override with PIPE_API_KEY when talking to a real auth-checked endpoint.
KEY = os.environ.get("PIPE_API_KEY", "EMPTY")
# Local tokenizer path passed to run_hybrid_web_agent --tokenizer-path /
# --summary-tokenizer-path. Defaults to empty (skip); set when running with
# a local checkpoint that needs the tokenizer.
PIPE_TOKENIZER_PATH = os.environ.get("PIPE_TOKENIZER_PATH", "")
# R-stage operator model (the one that emits screen coordinates).
R_OPERATION_MODEL = os.environ.get("R_OPERATION_MODEL", "qwen3.7-plus")

T_A = int(os.environ.get("PIPE_T_A", 90 * 60))
T_B = int(os.environ.get("PIPE_T_B", 40 * 60))
T_C = int(os.environ.get("PIPE_T_C", 30 * 60))
T_R = int(os.environ.get("PIPE_T_R", 150 * 60))

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


def build_rollout_rows(domain: str, site: dict, tasks: list[dict]) -> list[dict]:
    """Same field layout task_synth's aggregator writes (rollout-ready)."""
    rows = []
    entry_url = (site or {}).get("entry_url", "")
    for i, t in enumerate(tasks):
        start_mode = t.get("start_mode") or "O"
        rows.append({
            "task_id": f"synth7-{domain}-{i}",
            "subset": (site or {}).get("category", "synth7"),
            "question": t["question"],
            "web": entry_url if start_mode == "O" else "",
            "dataset": "synth7",
            "split": "test",
            "record_id": f"synth7-{domain}-{i}",
            "answer": "",
            "start_mode": start_mode,
            "route_plan": json.dumps(t.get("route_plan_typed") or [],
                                     ensure_ascii=False),
            "_signal": json.dumps({
                "difficulty": t.get("difficulty"),
                "shape": t.get("shape"),
                "n_crossings": t.get("n_crossings"),
                "score": t.get("score"),
                "o_fact_keys": t.get("o_fact_keys"),
                "s_fact_keys": t.get("s_fact_keys"),
                "answer_sketch": t.get("answer_sketch"),
            }, ensure_ascii=False),
        })
    return rows


class Pipeline:
    def __init__(self, pool: list[dict], out: Path, args):
        self.pool = pool
        self.out = out
        self.args = args
        self.sem_a = asyncio.Semaphore(args.sem_a)
        self.sem_b = asyncio.Semaphore(args.sem_b)
        self.sem_c = asyncio.Semaphore(args.sem_c)
        self.sem_r = asyncio.Semaphore(args.sem_r)
        self.inflight = {"A": 0, "B": 0, "C": 0, "R": 0}
        self.done = 0
        self.tasks_jsonl = out / "synth7_tasks.jsonl"
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
                            explore_only=True),
                    timeout=T_A)
            except asyncio.TimeoutError:
                log(f"PIPE {domain} SKIP A timeout({T_A}s)")
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
        site_dir = self.out / domain
        shim = self.out / "shims" / domain
        shim.mkdir(parents=True, exist_ok=True)
        link = shim / domain
        if not link.exists():
            link.symlink_to(site_dir)
        cmd = ["python3", str(SYNTH3 / "fact_prober_v2.py"),
               "--run", str(shim), "--conc", str(self.args.b_conc)]
        async with self.sem_b:
            self.inflight["B"] += 1
            try:
                rc = await run_proc(cmd, T_B,
                                    self.out / "_logs" / f"{domain}_B.log")
            finally:
                self.inflight["B"] -= 1
        ft = site_dir / "fact_tags.json"
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

    async def stage_b_or_skip(self, domain: str) -> dict | None:
        ft = self.out / domain / "fact_tags.json"
        if self.args.skip_existing and ft.is_file():
            try:
                d = json.loads(ft.read_text())
                c = d.get("counts") or {}
                log(f"PIPE {domain} B_skip (cached) "
                    f"S={c.get('s_reachable',0)} O={c.get('o_exclusive',0)}")
                return d
            except Exception:
                pass
        return await self.stage_b(domain)

    async def stage_c_or_skip(self, domain: str) -> list[dict]:
        site_dir = self.out / domain
        tp = site_dir / "task.json"
        if self.args.skip_existing and tp.is_file():
            try:
                out = json.loads(tp.read_text())
                kind = out.get("kind")
                tasks = out.get("tasks") or []
                log(f"PIPE {domain} C_skip (cached) tasks={len(tasks)} kind={kind}")
                if not tasks:
                    return []
                rows = build_rollout_rows(domain, out.get("site") or {}, tasks)
                append_jsonl(self.tasks_jsonl, rows)
                return rows
            except Exception:
                pass
        return await self.stage_c(domain)

    # ---------------- stage C ----------------
    async def stage_c(self, domain: str) -> list[dict]:
        site_dir = self.out / domain
        async with self.sem_c:
            self.inflight["C"] += 1
            try:
                fut = asyncio.to_thread(ts_mod.synth_one, site_dir,
                                        self.args.c_retries,
                                        self.args.n_tasks)
                out = await asyncio.wait_for(fut, timeout=T_C)
            except asyncio.TimeoutError:
                log(f"PIPE {domain} SKIP C timeout({T_C}s)")
                self._state(domain, "C", ok=False, err="timeout")
                return []
            except Exception as exc:
                log(f"PIPE {domain} SKIP C {type(exc).__name__}: {str(exc)[:90]}")
                self._state(domain, "C", ok=False, err=str(exc)[:200])
                return []
            finally:
                self.inflight["C"] -= 1
        kind = out.get("kind")
        tasks = out.get("tasks") or []
        log(f"PIPE {domain} C_done tasks={len(tasks)} kind={kind}")
        self._state(domain, "C", ok=True, kind=kind, n_tasks=len(tasks))
        if not tasks:
            return []
        rows = build_rollout_rows(domain, out.get("site") or {}, tasks)
        append_jsonl(self.tasks_jsonl, rows)
        return rows

    # ---------------- stage R ----------------
    async def stage_r(self, domain: str, rows: list[dict]):
        roll = self.out / domain / "rollout"
        roll.mkdir(parents=True, exist_ok=True)
        inp = roll / "input.jsonl"
        with inp.open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Endpoint URLs are deployment-specific. Override the four PIPE_*_URL
        # vars and PIPE_*_MODEL vars for your own model serving topology.
        base_url_main = os.environ.get("PIPE_BASE_URL", "http://localhost:8000/v1")
        base_url_vision = os.environ.get("PIPE_VISION_BASE_URL", base_url_main)
        router_model = os.environ.get("PIPE_ROUTER_MODEL", "kimi-k2.6")
        search_model = os.environ.get("PIPE_SEARCH_MODEL", "qwen3.7-max")
        summary_model = os.environ.get("PIPE_SUMMARY_MODEL", "deepseek-v4-flash")
        proxy_url = os.environ.get("PIPE_PROXY_URL", "")
        sandbox_url = os.environ.get("PIPE_SANDBOX_BASE_URL", "http://127.0.0.1:8080")
        cmd = ["python3", HYBRID_AGENT,
               "--input-path", str(inp),
               "--output-dir", str(roll),
               "--output-path", str(roll / "results.jsonl"),
               "--agent-mode", "hybrid_guided_routed",
               "--max-workers", str(self.args.r_workers),
               "--max-task-attempts", "1",
               "--max-iter", "40",
               "--headless",
               "--window-width", "1280", "--window-height", "720",
               "--coord-scale", str(coord_scale_for_model(R_OPERATION_MODEL)),
               "--base-url", base_url_main,
               "--model", R_OPERATION_MODEL,
               "--api-key", KEY,
               "--api-timeout", "240",
               "--routed-search-model", search_model,
               "--routed-search-base-url", base_url_main,
               "--routed-operation-model", R_OPERATION_MODEL,
               "--routed-operation-base-url", base_url_main,
               "--router-model", router_model,
               "--router-base-url", base_url_vision,
               "--router-api-key", KEY,
               "--router-timeout", "240",
               "--router-max-tokens", "4096",
               "--summary-base-url", base_url_vision,
               "--summary-model", summary_model,
               "--summary-api-key", KEY,
               "--summary-timeout", "120",
               "--visit-summary-max-tokens", "4000",
               "--no-visit-images",
               "--visit-max-images", "0",
               "--max-tokens", "16000",
               "--code-sandbox-base-url", sandbox_url]
        if PIPE_TOKENIZER_PATH:
            cmd += ["--tokenizer-path", PIPE_TOKENIZER_PATH,
                    "--summary-tokenizer-path", PIPE_TOKENIZER_PATH]
        if proxy_url:
            cmd += ["--proxy-url", proxy_url]
        async with self.sem_r:
            self.inflight["R"] += 1
            try:
                rc = await run_proc(cmd, T_R,
                                    self.out / "_logs" / f"{domain}_R.log")
            finally:
                self.inflight["R"] -= 1
        res_p = roll / "results.jsonl"
        n_fin = n_ans = 0
        if res_p.is_file():
            for line in res_p.open():
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                n_fin += 1
                if r.get("termination") == "answer":
                    n_ans += 1
        log(f"PIPE {domain} R_done finished={n_fin}/{len(rows)} "
            f"answer={n_ans} rc={rc}")
        self._state(domain, "R", ok=rc == 0, finished=n_fin, answer=n_ans)

    # ---------------- per-site chain ----------------
    async def site_flow(self, site: dict):
        domain = site.get("domain", "?").replace("/", "_")
        try:
            res = await self.stage_a(site)
            if res is not None:
                facts = ((res.get("kb") or {}).get("facts")) or {}
                if facts:
                    tagged = await self.stage_b_or_skip(domain)
                    if tagged is not None:
                        rows = await self.stage_c_or_skip(domain)
                        if rows:
                            await self.stage_r(domain, rows)
                else:
                    log(f"PIPE {domain} SKIP B no facts from A")
        except Exception as exc:
            log(f"PIPE {domain} SKIP flow {type(exc).__name__}: {str(exc)[:120]}")
        finally:
            self.done += 1
            log(f"PIPE_STATS done={self.done}/{len(self.pool)} "
                f"inflight A={self.inflight['A']} B={self.inflight['B']} "
                f"C={self.inflight['C']} R={self.inflight['R']}")

    async def run(self):
        log(f"PIPELINE start: {len(self.pool)} sites  "
            f"semA={self.args.sem_a} semB={self.args.sem_b} "
            f"semC={self.args.sem_c} semR={self.args.sem_r}")
        await asyncio.gather(*(self.site_flow(s) for s in self.pool))
        log("PIPELINE_ALL_DONE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--n-tasks", type=int, default=3)
    ap.add_argument("--c-retries", type=int, default=2)
    ap.add_argument("--sem-a", type=int, default=20)
    ap.add_argument("--sem-b", type=int, default=8)
    ap.add_argument("--b-conc", type=int, default=4)
    ap.add_argument("--sem-c", type=int, default=6)
    ap.add_argument("--sem-r", type=int, default=8)
    ap.add_argument("--r-workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--agent", choices=("v7", "v8"), default=None,
                    help="v7 (single agent) or v8 (decoupled op+extractor); "
                         "also accepts env SYNTH3_AGENT")
    ap.add_argument("--n-tasks-cap", type=int, default=None,
                    help="override --n-tasks at runtime (default 3)")
    args = ap.parse_args()

    if args.agent:
        os.environ["SYNTH3_AGENT"] = args.agent
    if args.n_tasks_cap is not None:
        args.n_tasks = args.n_tasks_cap

    obj = json.load(open(args.pool))
    pool = obj if isinstance(obj, list) else obj["all_usable"]
    if args.limit:
        pool = pool[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    asyncio.run(Pipeline(pool, out, args).run())


if __name__ == "__main__":
    main()
