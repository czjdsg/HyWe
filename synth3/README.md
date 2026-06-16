# synth3 — Data generation pipeline

`synth3/` builds **(task, reference answer, rollout trajectory)** triples by
running an exploration / probing / synthesis / rollout pipeline over a
candidate-site pool.

The pipeline is the one that produced the agent's hybrid training data: it
explores each site live, harvests verifiable facts, turns them into tasks
with reference answers, then re-runs the agent on those tasks to generate
trajectories.

## Pipeline at a glance

```
Stage A  Explore       explorer_agent[_v8].py  +  explorer_runtime.py
         (Playwright + vision LLM walks each site, records facts)
              │
              ▼
Stage B  Probe         fact_prober_v2.py  (search-agent verifies each fact,
                                            harvests cross-source claims)
              │
              ▼
Stage C  Synth         task_synth.py  (Opus turns facts into tasks +
                                       reference answers)
              │
              ▼
Stage R  Rollout       run_hybrid_web_agent.py  (run hybrid_guided_routed
                                                 on the synthesized tasks)
```

Driver scripts:

- `pipeline_driver.py` — full A→B→C→R per-site streaming pipeline (every
  site flows through all four stages as soon as its previous stage
  finishes — no whole-batch barriers).
- `synth_driver.py` — A-only (explore + dump candidates), useful for
  eyeballing the explorer in isolation.
- `rollout_watcher.py` — long-running watcher that polls a synth3 output
  dir and launches R-stage rollouts as new sites finish A→B→C.

## File map

| File | Role |
| --- | --- |
| `pipeline_driver.py` | Full per-site streaming A→B→C→R pipeline with stage semaphores and per-stage hard timeouts. |
| `synth_driver.py`    | Explore-only driver. Spins one `ExplorerAgent` per site. |
| `rollout_watcher.py` | Periodic batch-launcher for R-stage rollouts. |
| `explorer_agent.py`  | v7 explorer (operator + recorder via single Claude). |
| `explorer_agent_v8.py` | v8 explorer (decoupled operator / fact extractor). |
| `explorer_runtime.py` | Slim wrapper around `HybridPlaywrightComputer` + `SearchTool` + `VisitTool` for the explorer loop. |
| `fact_prober.py`     | v1 fact prober — manual per-fact search loop. |
| `fact_prober_v2.py`  | v2 fact prober — invokes `run_hybrid_web_agent --agent-mode search` per site. |
| `task_synth.py`      | Synthesizes tasks + reference answers from probed facts. |
| `qa_sampler.py`      | Random-samples synthesized tasks + rollouts for human QA review. |
| `qa_loop.sh`         | Periodic QA loop (calls `qa_sampler.py` every 30 min). |
| `build_pipe_viz.py`  | Builds the HTML viz under `viz/pipe_index.html`. |
| `viz_refresh.sh`     | Periodic viz refresh loop. |
| `viz/`               | HTML viewer + a tiny demo trajectory (`google.com_flights/`). |
| `scripts/v8_full1170_launch.sh`   | End-to-end launch for the full v8 pipeline. |
| `scripts/v8_qwen_explore_launch.sh` | Explore-only launch with qwen on both roles. |

## Inputs

The pipeline reads a **site pool JSON** with one record per candidate site:

```json
[
  { "domain": "google.com_flights",
    "entry_url": "https://www.google.com/travel/flights",
    "category": "travel",
    "notes": "..." },
  ...
]
```

Pass it via `--pool /path/to/pool.json`. Format can also be
`{"all_usable": [...]}` — the schema used by the bundled pool. `scripts/v8_*_launch.sh`
defaults to `benchmarks/data/hybrid_pool/hybrid_pool_combined.json` which
**ships with the repo** (1170 sites across 24 categories — adjudicated +
probe-validated). Override with `POOL=/path/to/another.json` to point at
a different pool.

## Configuration — env vars

Synth3 is configured almost entirely through environment variables so the
same scripts work for every model topology. Anything not listed has a sane
default in the corresponding `*.py`.

### Endpoints

| Var | Default | Used by |
| --- | --- | --- |
| `EXPLORER_API_BASE` | `http://localhost:8000/v1` | Stage A operator |
| `EXPLORER_API_KEY`  | `EMPTY` | Stage A operator |
| `EXPLORER_MODEL`    | `qwen3.7-plus` | Stage A operator |
| `FACT_API_BASE`     | `EXPLORER_API_BASE` | Stage A fact extractor |
| `FACT_MODEL`        | `kimi-k2.6` | Stage A fact extractor |
| `PROBE_AGENT_BASE`  | `http://localhost:8000/v1` | Stage B search agent |
| `PROBE_AGENT_MODEL` | `qwen3.7-max` | Stage B search agent |
| `SUMMARY_BASE_URL`  | `http://localhost:8000/v1` | Visit summary (B and explorer) |
| `SUMMARY_MODEL`     | `deepseek-v4-flash` | Visit summary |
| `SYNTH_BASE`        | `http://localhost:8000/v1` | Stage C synthesis |
| `SYNTH_MODEL`       | `qwen3.7-max` | Stage C synthesis |
| `PIPE_BASE_URL`     | `http://localhost:8000/v1` | Stage R operator endpoint |
| `PIPE_VISION_BASE_URL` | `PIPE_BASE_URL` | Stage R router + summary endpoint |
| `PIPE_ROUTER_MODEL` | `kimi-k2.6` | Stage R router |
| `PIPE_SEARCH_MODEL` | `qwen3.7-max` | Stage R search side |
| `PIPE_SUMMARY_MODEL` | `deepseek-v4-flash` | Stage R summary |
| `R_OPERATION_MODEL` | `qwen3.7-plus` | Stage R operator model |

### Pools (shared with inference framework)

| Var | Default |
| --- | --- |
| `SERPER_BASE_URL` | `http://127.0.0.1:8765` (local serper pool) |
| `JINA_READER_BASE_URL` | `http://127.0.0.1:8766` (local jina pool) |
| `SERPER_KEY_ID`, `JINA_API_KEY` | (set in `env.local.sh`) |

`fact_prober_v2.py` **refuses to launch** if the local serper/jina pool
URLs are not in the child env — this is a hard requirement so the search
agent never accidentally hits public endpoints during data generation.

### Paths

| Var | Default | Purpose |
| --- | --- | --- |
| `HYBRID_AGENT_ENTRY` | `<repo>/inference/run_hybrid_web_agent.py` | Stage R / B agent entry script |
| `PIPE_TOKENIZER_PATH` | `""` (skip) | Local tokenizer for the inference framework's token counting |
| `PROBE_TOKENIZER_PATH` | `""` | Same, for B-stage subprocess |
| `ROLLOUT_TOKENIZER_PATH` | `""` | Same, for `rollout_watcher.py` |
| `PIPE_PROXY_URL` | `""` | HTTP proxy for the R-stage browser |
| `PROXY_URL` | `""` | HTTP proxy for the A-stage browser |

### Per-stage timeouts

| Var | Default | Purpose |
| --- | --- | --- |
| `PIPE_T_A` | 5400 (90 min) | Stage A hard cap per site |
| `PIPE_T_B` | 2400 (40 min) | Stage B |
| `PIPE_T_C` | 1800 (30 min) | Stage C |
| `PIPE_T_R` | 9000 (150 min) | Stage R |

## Reference experiment topology

The production v8 run used a 2-endpoint, 5-model split. The exact config
sits in `scripts/v8_experiment.env`:

| Stage / role           | Model                  | Endpoint |
| ---------------------- | ---------------------- | -------- |
| A operator (vision)    | `qwen3.7-plus`         | :8000 |
| A fact extractor       | `kimi-k2.6` (vision)   | :8002 |
| B search probe         | `qwen3.7-max`          | :8000 |
| C task synth           | `qwen3.7-max`          | :8002 |
| R operator (vision)    | `qwen3.7-plus`         | :8000 |
| R router               | `kimi-k2.6`            | :8002 |
| R search side          | `qwen3.7-max`          | :8000 |
| Visit summary (A + R)  | `deepseek-v4-flash`    | :8002 |

The launch scripts auto-source `scripts/v8_experiment.env` if it exists; set
`EXPERIMENT_ENV=/path/to/file.env` to point at a different topology, or
`EXPERIMENT_ENV=NONE` to fall back to the per-stage defaults documented
under "Configuration — env vars" below.

The bundled `v8_experiment.env` writes to `47.82.159.100:8000` and
`47.82.159.100:8002`. Edit those URLs before running on a different host.

## Running

End-to-end launch (override paths as needed):

```bash
POOL=/path/to/site_pool.json \
OUT=runs/synth3_v8_full1170 \
bash synth3/scripts/v8_full1170_launch.sh
```

A-only (just explore):

```bash
POOL=/path/to/site_pool.json \
OUT=runs/synth3_v8_explore \
bash synth3/scripts/v8_qwen_explore_launch.sh
```

Watcher for incremental rollout while the pipeline is still running:

```bash
WATCHER_BATCH_ROOT=runs/synth3_rollouts \
WATCHER_SYNTH_OUT_DIR=runs/synth3_v8_full1170 \
python3 synth3/rollout_watcher.py --period 900 --min-batch 8 --max-batch 64
```

Periodic QA sample (random-pick + LLM critique):

```bash
QA_RUN_DIR=runs/synth3_v8_full1170 \
bash synth3/qa_loop.sh
```

Build / refresh the viz:

```bash
VIZ_RUN=runs/synth3_v8_full1170 \
VIZ_OUT=runs/synth3_v8_full1170_viz \
bash synth3/viz_refresh.sh
```

## Outputs

Per site: `<OUT>/<domain>/`

- `result.json` — A-stage explorer output (facts, trace, candidates)
- `fact_tags.json` — B-stage probed/judged facts with `s_transcript`
- `tasks.jsonl` — C-stage synthesized tasks + reference answers
- `rollout/` — R-stage `run_hybrid_web_agent` per-task folders + `results.jsonl`

Aggregates at `<OUT>/`:

- `fact_tags_index.jsonl`
- `tasks_index.jsonl`
- `_logs/<domain>/{a,b,c,r}.log`

## Dependencies

Same as the top-level `requirements.txt`. The viz uses no extra deps beyond
the standard library + the existing inference / search tooling.

## Notes

- The shipped `viz/google.com_flights/` is a tiny **demo trajectory** (9 PNG
  screenshots) so the HTML viewer renders something out of the box. Real
  runs write into `<OUT>` not `viz/`.
- `env.local.sh` at the repo root is gitignored — populate it with your
  Jina + Serper keys before running B-stage.
- `pipeline_driver.py` is the production entry point. The other drivers
  are subset launchers.
