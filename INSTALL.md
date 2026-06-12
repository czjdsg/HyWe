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
endpoint (vLLM, SGLang, llama.cpp, the official OpenAI API, …). At minimum:

```bash
export API_KEY=sk-...                   # or empty for local vLLM
export BASE_URL=http://127.0.0.1:8001/v1
export MODEL=kimi-k2.5
```

The inference framework also supports a **separate summary-model endpoint**
for `visit` page summarization (image-capable model recommended):

```bash
export SUMMARY_API_KEY=...
export SUMMARY_BASE_URL=http://127.0.0.1:8002/v1
export SUMMARY_MODEL=Qwen3.5-27B
export SUMMARY_TOKENIZER_PATH=/path/to/local/qwen-checkpoint  # for token counting
```

These are picked up by `inference/run_hybrid_web_agent.py` via the CLI flags
`--base-url`, `--model`, `--summary-base-url`, etc.

## 6. Smoke test — operation benchmark

```bash
python inference/run_hybrid_web_agent.py \
  --input-path benchmarks/data/operation/web_webvoyager_643_final.jsonl \
  --max-tasks 3 \
  --base-url $BASE_URL --model $MODEL \
  --summary-base-url $SUMMARY_BASE_URL --summary-model $SUMMARY_MODEL \
  --headless \
  --output-dir runs/smoke_webvoyager
```

Each task writes a per-task folder under `runs/smoke_webvoyager/<task_id>/`
with screenshots, the trajectory, and `result.json`. The run also appends a
flat JSONL summary to `runs/smoke_webvoyager/summary.jsonl`.

## 7. Smoke test — search benchmark

```bash
python inference/run_hybrid_web_agent.py \
  --input-path benchmarks/data/search/dr_browsecomp_1266.jsonl \
  --max-tasks 3 \
  --agent-mode hybrid \
  --base-url $BASE_URL --model $MODEL \
  --summary-base-url $SUMMARY_BASE_URL --summary-model $SUMMARY_MODEL \
  --headless \
  --output-dir runs/smoke_browsecomp
```

## 8. Score it — reference-answer evaluation (Type 1)

```bash
python evaluation/scripts/evaluate_reference_answers.py \
  --run-dir runs/smoke_webvoyager \
  --output-dir runs/smoke_webvoyager/eval \
  --base-url $BASE_URL --model $MODEL
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

## 10. Stopping the pools

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
