# `run_hybrid_web_agent.py` — default values

Quick reference for every CLI flag whose default the user is likely to want
to know without reading argparse help. Anything not listed here keeps the
inline default in the argparse declaration; this file only covers what
matters at runtime.

## Sampling / output

| Flag | Default | Notes |
| --- | --- | --- |
| `--max-tokens` | 16384 | per-turn LLM output cap; large so `<report>` + `<tool_call>` always fit |
| `--router-max-tokens` | 4096 | router output cap (`hybrid_routed` / `hybrid_guided_routed`) |
| `--temperature` | **0.0** | deterministic by default |
| `--top-p` | 0.95 | |
| `--top-k` | 20 | |
| `--repetition-penalty` | 1.0 | |
| `--seed` | 1234 | |

## Loop control

| Flag | Default | Notes |
| --- | --- | --- |
| `--max-iter` | **50** | hard cap on decision turns per task |
| `--max-format-retries` | 3 | recover from bad-formatted model output |
| `--max-task-attempts` | 1 | task-level outer retry |
| `--max-workers` | 1 | task-level concurrency |

## Context truncation (input side)

| Flag | Default | Notes |
| --- | --- | --- |
| `--max-observation-chars` | 0 | 0 = **no truncation** (full SoM into prompt) |
| `--max-action-history-chars` | 0 | 0 = **no truncation** |
| `--code-sandbox-max-output-chars` | 0 | 0 = no truncation of code stdout |
| `--code-sandbox-max-file-bytes` | 25 000 000 | 25 MB single-file cap |

## `visit` tool (Jina Reader)

| Flag | Default | Notes |
| --- | --- | --- |
| `--visit-max-webpage-tokens` | 32 000 | tokens per chunk; long pages → incremental summarization |
| `--visit-max-webpage-chars` | 0 | optional per-chunk char cap; 0 = off |
| `--use-visit-images` / `--no-visit-images` | **`--no-visit-images`** | default: **no images** sent to the summary model |
| `--visit-max-images` | 4 | only used when `--use-visit-images` is on |
| `--visit-max-image-url-chars` | 2 000 000 | single base64 data URL cap |
| `--visit-summary-max-tokens` | **4000** | summary-model output cap per chunk |

## Browser viewport / coordinates

| Flag | Default | Notes |
| --- | --- | --- |
| `--window-width × --window-height` | **1280 × 720** | viewport size |
| `--coord-scale` | **auto** (from `--model`) | resolved via `coord_scale_for_model`: |
|  |  | • `kimi*` → 1 (relative 0..1 floats) |
|  |  | • `qwen*` → 1000 (0..1000 logical grid) |
|  |  | • `claude` / `sonnet` / `opus` / `gpt-4v` / `gpt-4o` → 0 (raw pixels) |
|  |  | • anything else → 1 |
|  |  | Set `--coord-scale N` or env `COORD_SCALE=N` to override. |

## Timeouts (seconds)

All four timeouts default to **900 s (15 min)** so long-context generations
don't get cut off:

| Flag | Default |
| --- | --- |
| `--api-timeout` | 900 |
| `--router-timeout` | 900 |
| `--summary-timeout` | 900 |
| `--code-sandbox-request-timeout` | 900 |

## Other useful flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--agent-mode` | `hybrid` (env `AGENT_MODE`) | one of `hybrid`, `hybrid_routed`, `hybrid_guided_routed`, `operation`, `search` |
| `--headless` | off | enable for benchmark / synth runs |
| `--output-dir` | `./results` | per-task subfolders + `summary.jsonl` |
| `--code-sandbox-base-url` | `http://127.0.0.1:8080` (env `CODE_SANDBOX_BASE_URL`) | code-sandbox HTTP endpoint |

## Tuning notes

- `--max-iter 50` is the production data-generation default. Bump to 100 for
  rollout where you want long unrolled trajectories (this is what
  `rollout_watcher.py` does).
- `--visit-summary-max-tokens 4000` is large because the summary model
  consumes a whole webpage chunk — too small a cap silently truncates
  important evidence.
- `--temperature 0.0` is the right default when you want reproducible runs.
  Bump to `0.2` for diversity (eval) or `0.6+` for synth sampling.
- Long timeouts (900 s) match real-world long-context generation; on a
  fast endpoint feel free to lower to 240 s for snappier failures.
