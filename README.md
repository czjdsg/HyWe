# HybridWebAgent

End-to-end pipeline for **hybrid web tasks** — a single agent that mixes
browser operation, web search, and webpage reading — together with the
reference-answer evaluation framework used to score it.

This repository bundles seven things:

| Component | Path | Purpose |
| --- | --- | --- |
| Inference framework | `inference/` | The agent itself: Playwright runtime, IterResearch-style controller, search / scholar / visit / code-sandbox tools, multiple ablation modes. |
| Evaluation framework | `evaluation/` | Reference-answer judges (Type 1) for both **operation** benchmarks (WebVoyager, Online-Mind2Web) and **search / deep-research** benchmarks (BrowseComp, FRAMES, GAIA, WebWalkerQA, HLE, …). |
| Benchmark data | `benchmarks/data/` | Curated JSONL splits for the operation + search benchmarks above. |
| Data-generation pipeline | `synth3/` | A→B→C→R pipeline that produces (task, reference answer, rollout) triples from a site pool. See `synth3/README.md`. |
| Local API-key pools | `pools/jina_pool/`, `pools/serper_pool/` | Local key-rotation proxies for Jina Reader and Serper. Ship **empty** by design — add your own keys after cloning. |
| Local code sandbox | `tools/local_sandbox/` | Pure-Python HTTP server that speaks the same `/run_code` JSON contract as the vemlp `code-sandbox` Docker image. Use this on hosts where Docker is unavailable (Autodl, locked-down VMs). Supports python/bash/nodejs/cpp/go/typescript. |
| Vendored clash | `vendor/clash/` | Static linux/amd64 clash binary + USA-topology config + GeoIP DB. Used as the outbound HTTP/SOCKS proxy for the browser. Listens on 8990/8991/9090. |
| Inference preference | `configs/inference.env` | Production-tuned env wiring: `BASE_URL`, `MODEL`, `SUMMARY_MODEL`, key-pool URLs, sandbox URL, proxy URL — `source` this file before launching the agent. |

The agent itself is described in detail in `inference/README.original.md` and
`inference/ARCHITECTURE.md` / `inference/WORKFLOW_OVERVIEW.md`. The evaluation
framework's design is documented in `evaluation/README.original.md` and
`evaluation/docs/STATEFUL_*.md`.

## Quick start

```bash
git clone git@github.com:czjdsg/HybridWebAgent.git
cd HybridWebAgent
bash scripts/install.sh         # see INSTALL.md for what this does
bash scripts/start_all.sh       # jina_pool + serper_pool + local_sandbox + clash
python pools/jina_pool/manage_keys.py  add YOUR_JINA_KEY
python pools/serper_pool/manage_keys.py add YOUR_SERPER_KEY
source configs/inference.env    # production-tuned BASE_URL / MODEL / SUMMARY_*
```

End-to-end smoke test:

```bash
python inference/run_hybrid_web_agent.py \
  --input-path benchmarks/data/operation/web_webvoyager_643_final.jsonl \
  --base-url $BASE_URL --model $MODEL --api-key $API_KEY \
  --summary-base-url $SUMMARY_BASE_URL --summary-model $SUMMARY_MODEL \
  --code-sandbox-base-url $CODE_SANDBOX_BASE_URL \
  --proxy-url $TASK_PROXY_URL \
  --headless \
  --output-dir runs/smoke
```

Then score it:

```bash
python evaluation/scripts/evaluate_reference_answers.py \
  --run-dir runs/smoke \
  --output-dir runs/smoke/eval \
  --base-url $BASE_URL --model $MODEL --api-key $API_KEY
```

(All inference defaults — coord-scale, temperature, max-iter, window size,
timeouts, visit-images — are documented in `inference/DEFAULTS.md`.)

## Repository layout

```
HybridWebAgent/
├── README.md                   # this file
├── INSTALL.md                  # full install instructions
├── requirements.txt            # pip dependencies
├── scripts/                    # install + pool management helpers
│   ├── install.sh
│   ├── start_pools.sh
│   └── stop_pools.sh
├── inference/                  # agent + runtime + tools
│   ├── run_hybrid_web_agent.py   # CLI entrypoint
│   ├── hybrid_agent.py           # main controller loop
│   ├── hybrid_runtime.py         # Playwright runtime
│   ├── hybrid_prompts.py         # prompts + tool schemas (~96K)
│   ├── hybrid_search.py          # Serper-backed `search`
│   ├── hybrid_scholar.py         # Google Scholar (Serper-Scholar)
│   ├── hybrid_visit.py           # Jina-Reader-backed `visit`
│   ├── hybrid_code_sandbox.py    # code execution tool
│   ├── hybrid_utils.py
│   ├── visualize_*.py            # trajectory viewers
│   ├── ARCHITECTURE.md / WORKFLOW_OVERVIEW.md
│   └── requirements.txt
├── evaluation/                 # answer-eval framework
│   ├── scripts/
│   │   ├── evaluate_reference_answers.py   # Type-1 reference-answer judge
│   │   ├── evaluate_classified_web_tasks.py
│   │   ├── classify_task_validation_type.py
│   │   ├── classify_defer_answer_points.py
│   │   ├── rewrite_insta_criteria.py
│   │   └── ...
│   ├── prompts/                # all judge prompts
│   └── docs/                   # STATEFUL_DEFER / STATEFUL_OPERATION / ANNOTATION_*
├── benchmarks/data/
│   ├── operation/              # WebVoyager-643, Online-Mind2Web-256
│   ├── search/                 # BrowseComp / FRAMES / GAIA / WebWalkerQA / HLE / DeepSearch / browsecomp_zh
│   └── annotations/            # sidecar annotation JSONs (for routes beyond static_answer)
├── synth3/                     # data-generation pipeline (explore → probe → synth → rollout)
│   ├── pipeline_driver.py        # full A→B→C→R streaming driver
│   ├── synth_driver.py           # explore-only
│   ├── rollout_watcher.py        # incremental R-stage launcher
│   ├── explorer_agent[_v8].py    # A-stage agents
│   ├── fact_prober[_v2].py       # B-stage probers
│   ├── task_synth.py             # C-stage synthesizer
│   ├── qa_sampler.py / qa_loop.sh # periodic LLM QA review
│   ├── build_pipe_viz.py / viz/  # HTML viewer (ships with a demo trajectory)
│   ├── scripts/v8_*_launch.sh    # production launchers
│   └── README.md                 # pipeline + env-var reference
├── pools/
│   ├── jina_pool/              # Jina Reader key-rotation proxy (empty DB)
│   └── serper_pool/            # Serper key-rotation proxy (empty DB)
└── vendor/
    └── mobile_agent_browser_use/   # vendored SoM / clean_html (Mobile-Agent-v3.5)
```

## Benchmarks shipped here

All datasets in this repository carry **reference answers** and are evaluated
by the **Type-1 route** (`evaluate_reference_answers.py` against
`prompts/judge_reference_answer.txt` / `judge_static_answer.txt`).

### Operation (browser-driven tasks)

| File | Tasks | Source |
| --- | --: | --- |
| `benchmarks/data/operation/web_webvoyager_643_final.jsonl` | 643 | WebVoyager (full) |
| `benchmarks/data/operation/web_online_mind2web_256_nohuman_final.jsonl` | 256 | Online-Mind2Web (no-human-verification subset) |

### Search / deep-research (Type-1 answer benchmarks)

| File | Tasks | Source |
| --- | --: | --- |
| `benchmarks/data/search/dr_browsecomp_1266.jsonl` | 1 266 | BrowseComp (test) |
| `benchmarks/data/search/dr_browsecomp_zh_289.jsonl` | 289 | BrowseComp-ZH |
| `benchmarks/data/search/dr_deepsearch_100.jsonl` | 100 | DeepSearch |
| `benchmarks/data/search/dr_frames_824.jsonl` | 824 | FRAMES |
| `benchmarks/data/search/dr_gaia_test_301.jsonl` | 301 | GAIA test |
| `benchmarks/data/search/dr_gaia_validation_165.jsonl` | 165 | GAIA validation |
| `benchmarks/data/search/dr_hle_test_2500.jsonl` | 2 500 | HLE test |
| `benchmarks/data/search/dr_webwalkerqa_main_680.jsonl` | 680 | WebWalkerQA (main) |

Per-record schema (canonical):

```json
{
  "task_id": "BrowseComp-test-1",
  "question": "...",
  "answer": "1988-96",         // reference answer (Type 1)
  "web": "",                   // optional start URL (mostly for operation tasks)
  "dataset": "BrowseComp",
  "split": "test",
  "extra": {...},
  "raw": {...}
}
```

Operation records additionally carry a `web` start URL. Search records carry
an empty `web` and let the agent decide where to navigate.

## Pools

The local **Jina Reader** and **Serper** key-rotation proxies ship with no
keys. Both default to an empty SQLite DB next to their package (no DB is
committed; first run creates one).

```bash
# 1. Start both pools (default 127.0.0.1:8766 / 127.0.0.1:8765)
bash scripts/start_pools.sh

# 2. Add your keys (DB is created on first add)
python pools/jina_pool/manage_keys.py add  YOUR_JINA_KEY_1 YOUR_JINA_KEY_2 --label-prefix mine
python pools/serper_pool/manage_keys.py add YOUR_SERPER_KEY_1 --label main-1

# 3. Verify
curl http://127.0.0.1:8766/healthz
curl http://127.0.0.1:8765/healthz
```

Tell the agent to use them:

```bash
export JINA_READER_BASE_URL=http://127.0.0.1:8766
export SERPER_BASE_URL=http://127.0.0.1:8765
```

(These two env vars are the only contract the agent has with the pools — you
can also point them at any other compatible upstream.)

Pool internals (rotation, cooldown, exhausted-key handling) are documented in
`pools/jina_pool/README.original.md` and `pools/serper_pool/README.original.md`.

## Installation

See **[INSTALL.md](INSTALL.md)** for the full one-shot setup on a fresh
machine, including model-endpoint expectations.

## License

Research code. Re-uses third-party prompts and SoM grounding code from
[Mobile-Agent-v3.5](https://github.com/X-PLUG/MobileAgent) (vendored under
`vendor/mobile_agent_browser_use/`).
