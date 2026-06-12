# HybridWebAgent

`HybridWebAgent` is a small experimental project under `/data/czj/HybridWebAgent` that combines:

- `Mobile-Agent-v3.5/browser_use` for browser execution and SoM-style visual grounding
- `IterResearch` for report-driven context reconstruction and forced final-answer turns

It is meant as a cleaner base for mixed web tasks:

- pure website-operation tasks
- site-bounded information extraction
- hybrid tasks that require both navigation and factual answering

## Design

This project keeps the lower execution loop close to `Mobile-Agent`:

- Playwright browser session
- screenshot + SoM element labeling
- click / fill / scroll / select / go_back / goto actions
- browser-independent `visit` via Jina Reader style fetching
- external web search for `search`

And replaces the upper decision loop with an `IterResearch`-style controller:

- every turn must output `<report>...</report>`
- then either `<tool_call>...</tool_call>` or `<answer>...</answer>`
- the controller asks the model to re-evaluate the current bottleneck at every step:
  - missing information
  - missing interaction
  - enough information to answer
- next-turn context is rebuilt from:
  - task
  - last report
  - last action
  - last observation
  - current URL / title
  - current screenshot
  - current SoM text

So the agent is not forced to replay the entire raw history every turn.

## Files

- [run_hybrid_web_agent.py](/data/czj/HybridWebAgent/run_hybrid_web_agent.py): CLI entrypoint
- [hybrid_agent.py](/data/czj/HybridWebAgent/hybrid_agent.py): main agent loop
- [hybrid_runtime.py](/data/czj/HybridWebAgent/hybrid_runtime.py): Playwright runtime borrowing the Mobile-Agent execution style
- [hybrid_prompts.py](/data/czj/HybridWebAgent/hybrid_prompts.py): bottleneck-based prompts and tool schemas
- [hybrid_utils.py](/data/czj/HybridWebAgent/hybrid_utils.py): URL extraction and JSONL helpers
- [hybrid_search.py](/data/czj/HybridWebAgent/hybrid_search.py): search tool using Serper
- [hybrid_visit.py](/data/czj/HybridWebAgent/hybrid_visit.py): reader-style visit tool based on the IterResearch pattern

## Dependencies

The project imports SoM generation code directly from:

- `/data/czj/MobileAgent/Mobile-Agent-v3.5/browser_use/browser/playwright/som.py`
- `/data/czj/MobileAgent/Mobile-Agent-v3.5/browser_use/browser/playwright/clean_html.py`

You still need the browser-side dependencies used there.

Minimal Python packages:

- `openai`
- `playwright`
- `pillow`
- `requests`
- `beautifulsoup4`
- `transformers`

Then install Chromium once:

```bash
playwright install chromium
```

## Single-task Usage

```bash
python /data/czj/HybridWebAgent/run_hybrid_web_agent.py \
  --task "Find the monthly price of SSL Complete in solid-state-logic.com." \
  --web "https://store.solidstatelogic.com/" \
  --base-url "http://localhost:8001/v1" \
  --model "Qwen3-14B" \
  --headless
```

The script reads the API key from `API_KEY` by default. You can also pass `--api-key`.

By default, `visit` summaries use the same OpenAI-compatible endpoint and model as the main agent. You can override them independently with:

- `--summary-base-url`
- `--summary-model`
- `--summary-api-key`
- `--summary-tokenizer-path`

You can load local tool credentials with:

```bash
source /data/czj/HybridWebAgent/env.local.sh
```

Task-time network traffic now defaults to proxying through `http://127.0.0.1:8990`, while `localhost` and `127.0.0.1` stay in the direct-connect list so the local summary model keeps working.

You can override this with:

- `--proxy-url`
- `--no-proxy-hosts`

If you want `search` to work, set one of:

```bash
export SERPER_KEY_ID='your_serper_key'
```

If you want `visit` to use authenticated Jina Reader requests, set:

```bash
export JINA_API_KEY='your_jina_key'
```

`visit` now requests Jina output with:

- generated alt text
- image summaries
- preserved inline base64 image URLs

When inline image data URLs are present, the agent strips the raw base64 from the page text, keeps a bounded number of image attachments, and sends them to the visit-summary model call as multimodal inputs.

For long pages, `visit` now uses incremental chunk summarization:

- split the original page text into chunks of up to `--visit-max-webpage-tokens` tokens, default `32000`
- summarize each chunk with the current accumulated extraction
- carry the updated extraction into the next chunk summary
- keep `--visit-max-webpage-chars` disabled by default (`0`); set it only when an extra per-chunk character cap is needed

You can set the tokenizer with `--tokenizer-path` or `TOKENIZER_PATH`.

If you want the main agent and the visit-summary model to be different, a typical setup is:

```bash
python /data/czj/HybridWebAgent/run_hybrid_web_agent.py \
  --task "..." \
  --base-url "http://localhost:8001/v1" \
  --model "Qwen3-14B" \
  --summary-base-url "http://localhost:8002/v1" \
  --summary-model "Qwen3.5-27B" \
  --summary-tokenizer-path "/data/czj/models/Qwen/Qwen3.5-27B" \
  --headless
```

The local model path itself is used as tokenizer/config input. The actual `--summary-model` value should match the served model name of your summary endpoint.

## Dataset Usage

```bash
python /data/czj/HybridWebAgent/run_hybrid_web_agent.py \
  --input-path /path/to/tasks.jsonl \
  --base-url "http://localhost:8001/v1" \
  --model "Qwen3-14B" \
  --headless \
  --output-dir /data/czj/HybridWebAgent/results/demo
```

Each task gets:

- a per-task folder with screenshots and `result.json`
- an appended JSONL summary file

## Current Scope

This is a first integration pass, not a finished benchmark framework.

Implemented:

- report-driven context reconstruction
- browser actions with grounded labels
- `search` plus Jina-backed `visit` for missing-information bottlenecks
- multimodal visit summaries using Jina inline base64 images
- separate summary-model endpoint support for `visit`
- bottleneck-driven tool choice instead of a fixed task-type prior
- final-step forced answer mode

Not implemented yet:

- benchmark-specific login scripts
- parallel multi-task execution
- HTML viewer for trajectories
- separate planner / executor models
