# Install — HybridWebAgent

This document walks a coding agent (or a human) through a clean install on a
fresh Linux machine. Tested on Ubuntu 22.04 with Python 3.10 / 3.11.

## 0. Prerequisites

| Item | Version | Why |
| --- | --- | --- |
| Python | 3.10+ | All scripts use modern type syntax (`str \| None`). |
| pip   | recent | dependency install |
| git   | any | clone the repo |
| Chromium-running OS | Linux x86_64 | Playwright headless Chromium |
| tmux  | optional | `*_ctl.sh start` prefers tmux; falls back to nohup |

System packages that Playwright Chromium expects (Debian / Ubuntu):

```bash
sudo apt-get update
sudo apt-get install -y \
  python3 python3-venv python3-pip git curl tmux \
  libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
  libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
  libgbm1 libpango-1.0-0 libcairo2 libasound2
```

(If `apt` is not available, the equivalent packages exist on every major
distro — just `playwright install --with-deps chromium` will tell you what's
missing.)

## 1. Clone the repo

```bash
git clone git@github.com:czjdsg/HybridWebAgent.git
cd HybridWebAgent
```

## 2. Create a virtualenv and install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` covers everything the inference framework, the evaluation
framework, and the two pool servers need.

## 3. Install Playwright Chromium

```bash
playwright install chromium
# or, if you also want all OS-level deps:
playwright install --with-deps chromium
```

## 4. Empty key pools — confirmed working out of the box

The two key-pool servers ship with **no keys** and **no DB** committed. They
create an empty DB on first use.

```bash
bash scripts/start_pools.sh        # starts both pools in tmux (or nohup)
curl http://127.0.0.1:8766/healthz # jina
curl http://127.0.0.1:8765/healthz # serper
```

You can also start them individually:

```bash
bash pools/jina_pool/jina_pool_ctl.sh start
bash pools/serper_pool/serper_pool_ctl.sh start
```

### Adding your keys

```bash
# Jina Reader (https://jina.ai)
python pools/jina_pool/manage_keys.py add JINA_KEY_1 JINA_KEY_2 --label-prefix mine
python pools/jina_pool/manage_keys.py list
python pools/jina_pool/manage_keys.py stats

# Serper (https://serper.dev)
python pools/serper_pool/manage_keys.py add SERPER_KEY_1 --label main-1
python pools/serper_pool/manage_keys.py list
```

Both managers also support `import` from a text file (one key per line, or
`label,key`).

Once keys are in, the agent uses them via these env vars (already the default
in `scripts/start_pools.sh`):

```bash
export JINA_READER_BASE_URL=http://127.0.0.1:8766
export SERPER_BASE_URL=http://127.0.0.1:8765
```

## 5. Configure your model endpoint

The agent and judge talk to any **OpenAI-compatible** chat-completions
endpoint (vLLM, SGLang, llama.cpp, the official OpenAI API, …). If you
already have an endpoint running, skip to "production preference" below;
otherwise step **5a** stands up one locally.

### 5a. (Optional) Self-host Qwen3.6-35B-A3B via vllm

The repo ships a one-shot vllm setup for Qwen3.6-35B-A3B (the open-source
MoE model the production stack uses for `qwen3.6-plus`). Skip this if you
plan to point `BASE_URL` at a remote endpoint.

```bash
# 1. Install the serve-only extras (vllm + matching transformers/torch)
pip install -r requirements-serve.txt

# 2. Download weights (~72 GB) via the hf-mirror CDN.
#    Override the destination with QWEN36_DIR=/big/disk bash scripts/download_qwen36.sh
bash scripts/download_qwen36.sh

# 3. Serve on http://0.0.0.0:8000/v1 with served-model-name=qwen3.6-plus.
#    tensor-parallel auto-detected from CUDA_VISIBLE_DEVICES (or all GPUs).
bash scripts/serve_qwen36.sh
```

The model's HF arch is `qwen3_5_moe` (256 experts / 8 active, 40 layers,
262K native ctx). `vllm serve` loads its config independently of
`transformers.AutoConfig`, so the served-name + max-model-len defaults
work out of the box. Override anything with env vars:

```bash
QWEN36_DIR=/data/models/Qwen3.6-35B-A3B \
QWEN36_NAME=qwen3.6-plus \
QWEN36_PORT=8000 \
QWEN36_TP=4 \
QWEN36_MAX_LEN=131072 \
QWEN36_GPU_UTIL=0.92 \
bash scripts/serve_qwen36.sh
```

Once the endpoint reports `Application startup complete`, point the agent
at it by setting `BASE_URL=http://127.0.0.1:8000/v1` and
`MODEL=qwen3.6-plus` before sourcing `configs/inference.env` (or just
edit those two lines in the file).

### Production preference (existing endpoint)


The production preference is pre-baked in **`configs/inference.env`**:

```bash
source configs/inference.env
```

That sets:

| Var | Value |
| --- | --- |
| `BASE_URL` | `http://47.82.159.100:8000/v1` |
| `MODEL` | `qwen3.7-plus` (operator, vision) |
| `SEARCH_MODEL` | `qwen3.7-max` (routed search side) |
| `ROUTER_MODEL` | `kimi-k2.6` (router) |
| `SUMMARY_BASE_URL` | same as `BASE_URL` (single endpoint per user preference) |
| `SUMMARY_MODEL` | `deepseek-v4-flash` |
| `CODE_SANDBOX_BASE_URL` | `http://127.0.0.1:8080` (the local sandbox) |
| `SERPER_BASE_URL` | `http://127.0.0.1:8765` |
| `JINA_READER_BASE_URL` | `http://127.0.0.1:8766` |
| `TASK_PROXY_URL` | `""` (direct connection by default; set to e.g. `http://127.0.0.1:8990` if you deploy your own HTTP proxy) |
| `API_KEY` | `EMPTY` (vLLM placeholder; override for real auth) |

Each variable is `${VAR:-default}` so you can override any single one before
or after sourcing.

The remaining defaults (`--max-iter`, `--temperature`, `--coord-scale`,
timeouts, …) are baked into argparse — see `inference/DEFAULTS.md`.

## 5b. Local code sandbox (replaces the Docker container)

On hosts where Docker is unavailable (Autodl, locked-down VMs), the agent
talks to **`tools/local_sandbox/server.py`** instead — a pure-Python HTTP
server speaking the same `/run_code` JSON contract as the vemlp
`code-sandbox` image.

```bash
bash scripts/start_sandbox.sh        # starts on 127.0.0.1:8080
curl http://127.0.0.1:8080/healthz   # {"status":"ok",...}
```

Supported languages: `python`, `python3`, `bash`, `nodejs`, `cpp`, `go`,
`typescript`. Missing toolchains report a clear error in the response. Each
invocation runs in a fresh temp dir under `/tmp/hwa_sandbox/` with CPU
(`SANDBOX_CPU_SECONDS=30`) and memory (`SANDBOX_MEM_BYTES=2GB`) rlimits.

> This is a development sandbox, not a multi-tenant production sandbox:
> there is no chroot, no cgroups, no network namespace. Use on trusted
> hosts only.

## 5c. Outbound proxy (optional, BYO)

If the host can reach the model endpoint + Jina + Serper directly, skip
this step — the browser will go direct.

If you need a proxy for browser traffic (e.g. region-walled sites in
synth3 / certain WebVoyager tasks), deploy your own HTTP proxy and point
the agent at it:

```bash
export TASK_PROXY_URL=http://127.0.0.1:8990   # whatever your proxy listens on
```

`configs/inference.env` defaults this to empty, so a missing proxy is the
explicit choice rather than a silent fall-through.

## 5d. Start everything at once

```bash
bash scripts/start_all.sh            # pools + sandbox
bash scripts/stop_all.sh             # inverse
```

## 6. Smoke test — operation benchmark

The agent reads every row of `--input-path`; slice the dataset for a 3-task
smoke:

```bash
head -3 benchmarks/data/operation/web_webvoyager_643_final.jsonl \
  > /tmp/wv3.jsonl

python inference/run_hybrid_web_agent.py \
  --input-path /tmp/wv3.jsonl \
  --base-url $BASE_URL --model $MODEL --api-key $API_KEY \
  --summary-base-url $SUMMARY_BASE_URL --summary-model $SUMMARY_MODEL \
  --code-sandbox-base-url $CODE_SANDBOX_BASE_URL \
  --headless \
  --output-dir runs/smoke_webvoyager
```

Each task writes a per-task folder under `runs/smoke_webvoyager/<task_id>/`
with screenshots, the trajectory, and `result.json`. The run also appends a
flat JSONL summary to `runs/smoke_webvoyager/summary.jsonl`.

## 7. Smoke test — search benchmark

```bash
head -3 benchmarks/data/search/dr_browsecomp_1266.jsonl > /tmp/bc3.jsonl

python inference/run_hybrid_web_agent.py \
  --input-path /tmp/bc3.jsonl \
  --agent-mode hybrid \
  --base-url $BASE_URL --model $MODEL --api-key $API_KEY \
  --summary-base-url $SUMMARY_BASE_URL --summary-model $SUMMARY_MODEL \
  --code-sandbox-base-url $CODE_SANDBOX_BASE_URL \
  --headless \
  --output-dir runs/smoke_browsecomp
```

## 8. Score it — reference-answer evaluation (Type 1)

```bash
python evaluation/scripts/evaluate_reference_answers.py \
  --run-dir runs/smoke_webvoyager \
  --output-dir runs/smoke_webvoyager/eval \
  --base-url $BASE_URL --model $MODEL --api-key $API_KEY
```

The judge writes:

- one JSONL row per task with the predicted label and reason
- a summary JSON with counts and accuracy

## 9. (Optional) The five-route classified evaluator

For benchmarks where the operation answer is not a simple reference string
(target URLs, reveal scripts, defer key points, operation page states), use:

```bash
python evaluation/scripts/evaluate_classified_web_tasks.py \
  --run-dir runs/smoke_webvoyager \
  --annotations benchmarks/data/annotations/web_online_mind2web_256_nohuman_annotations.json \
  --base-url $BASE_URL --model $MODEL \
  --output-dir runs/smoke_webvoyager/eval_classified
```

For this repository's shipped benchmarks (WebVoyager, Online-Mind2Web,
all `dr_*`), the recommended path is the simpler `evaluate_reference_answers.py`
flow above. The classified evaluator is included because the codebase also
supports `target_urls` / `reveal_script` / `defer_framework` /
`operation_key_points` routes for downstream work.

## 10. (Optional) Data-generation pipeline (`synth3/`)

`synth3/` produces (task, reference answer, rollout) training triples by
running an A→B→C→R pipeline over a candidate-site pool. Full details and
the env-var reference live in `synth3/README.md`.

Minimum to run end-to-end on the new machine:

1. Provide a candidate-site pool JSON at
   `benchmarks/data/hybrid_pool/hybrid_pool_combined.json` (or override
   `POOL=...`). Schema:
   ```json
   [
     {"domain": "google.com_flights",
      "entry_url": "https://www.google.com/travel/flights",
      "category": "travel"}
   ]
   ```
2. Make sure the local pools are up and `env.local.sh` exports both
   `SERPER_KEY_ID` and `JINA_API_KEY` (step 4 above).
3. Launch the full pipeline:
   ```bash
   POOL=/path/to/pool.json \
   OUT=runs/synth3_v8 \
   bash synth3/scripts/v8_full1170_launch.sh
   tail -f runs/synth3_v8.log
   ```
4. Optional: incremental rollout watcher and periodic QA loop:
   ```bash
   WATCHER_BATCH_ROOT=runs/synth3_rollouts \
   WATCHER_SYNTH_OUT_DIR=runs/synth3_v8 \
   python3 synth3/rollout_watcher.py --period 900 --min-batch 8 --max-batch 64 &

   QA_RUN_DIR=runs/synth3_v8 bash synth3/qa_loop.sh &
   ```

`synth3` is fully optional — only needed when you want to regenerate
training data, not to run the inference framework or the evaluation
framework.

## 11. Stopping the pools

```bash
bash scripts/stop_pools.sh
# or individually:
bash pools/jina_pool/jina_pool_ctl.sh stop
bash pools/serper_pool/serper_pool_ctl.sh stop
```

## Troubleshooting

- **`ImportError: browser.playwright.clean_html`** — make sure you run from
  the repo root. The vendored MobileAgent SoM lives at
  `vendor/mobile_agent_browser_use/` and is auto-added to `sys.path`. Override
  with `export HYBRID_BROWSER_USE_ROOT=/your/path` if you have an external
  checkout.
- **`playwright._impl._errors.Error: Executable doesn't exist`** — run
  `playwright install chromium` inside the venv.
- **Pool `401 / 403` after first key** — the key was rejected as invalid; see
  `pools/<name>/logs/<name>_pool.log`.
- **`429 / 402 / quota exhausted` on every key** — out of credit; add more
  keys with `manage_keys.py add` and the proxy will rotate.
- **No `tmux`** — both `*_ctl.sh` scripts fall back to `nohup`. You'll see a
  pid file in the pool directory.

## Optional CLI proxy

If your local machine needs an HTTP proxy to reach `r.jina.ai` and
`google.serper.dev`, point the pools at it:

```bash
export JINA_POOL_PROXY_URL=http://127.0.0.1:8990
export SERPER_POOL_PROXY_URL=http://127.0.0.1:8990
bash scripts/stop_pools.sh && bash scripts/start_pools.sh
```

(The agent itself does not need a proxy when the pools are local; localhost
traffic stays direct.)
