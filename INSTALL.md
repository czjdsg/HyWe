# Installation and configuration

Use Python 3.10 or newer. Install the core dependencies and Chromium as shown
in the [README](README.md). `requirements-dev.txt` adds pytest. Optional local
model serving has a separate `requirements-serve.txt` because vLLM has its own
CUDA/PyTorch requirements. Use a vLLM build supporting the checkpoint's
`qwen3_5_moe` architecture; core inference only needs an API endpoint.

The README's [local deployment and inference parameter table](README.md#local-deployment-and-inference-parameters)
lists the sampling preset, context budget, browser settings and task limits.
The launcher uses temperature 1.0, top-p 0.95, top-k 20, repetition penalty
1.05 and 8,192 output tokens per turn. These are request parameters supplied
by `scripts/run_hywe.sh`.

## Policy endpoint

Copy `configs/hywe.env.example` to the ignored `configs/hywe.env`, edit it, and
source it in the same shell as the launcher. Set `BASE_URL` including `/v1`,
`MODEL` to the endpoint's served model name, and `API_KEY`. The model must
support text, images, and raw assistant text generation. The harness parses
`<think>`, `<tool_call>` and `<answer>` from text; it does not depend on an
endpoint's server-side tool execution.

For remote endpoints with request-size limits, `HYBRID_IMAGE_TRANSPORT=url`
publishes immutable image files into `HYBRID_IMAGE_PUBLISH_DIR`. You must serve
that directory at `HYBRID_IMAGE_URL_BASE` so the model provider can fetch them.
Inline base64 is the default and requires no image host.

`HYBRID_TOKENIZER_PATH` accepts a Hugging Face model ID or local tokenizer
directory. Accurate context accounting uses the tokenizer and chat template;
point it at a local copy for disconnected runs. `SUMMARY_*` selects the
goal-conditioned `visit` summarizer. The example reuses the policy endpoint;
archived paper runs used a separate `deepseek-v4-flash` summarizer.

## Web tools

- `JINA_API_KEY`: Jina Search when enabled, Jina Reader for `visit`, and Jina
  embeddings for collection's semantic pairing.
- `SERPER_API_KEY`: Serper `search` and `scholar`. `SERPER_KEY_ID` and
  `SERPER_KEY` are accepted compatibility aliases; leave them unset when using
  `SERPER_API_KEY`.
- `HYBRID_SEARCH_BACKEND`: `serper`, `jina`, or `auto`. `jina` uses Jina only;
  `auto` permits Serper fallback. See `inference/hybrid_search.py` for behavior.
- `TASK_PROXY_URL`: optional browser HTTP proxy; empty means direct browser
  access. SDK/web requests also honor their documented environment settings.
- `CODE_SANDBOX_BASE_URL`: a SandboxFusion-compatible `/run_code` service.

Example provider health check:

```bash
curl "$BASE_URL/models" -H "Authorization: Bearer $API_KEY"
```

Set your provider quota/concurrency limits before collecting large batches.
Start with `LIMIT=5`, `COUNT=20`, and `MAX_WORKERS=1` while verifying settings.
All model roles in `configs/collector.env.example` default to `kimi-k3`.

## Validation and failures

`python scripts/check_release.py` checks Python/shell syntax, benchmark
integrity, all archived HyWeBench metrics, and required files without paid
network calls. `python scripts/smoke_test.py` runs a local scripted model,
browser, mode switch, screenshot transport, and checklist judge.

If Chromium cannot launch, install its OS libraries with
`playwright install --with-deps chromium`. If screenshots fail to upload, use
base64 or verify your configured image host. If context accounting downloads
fail, set a local `HYBRID_TOKENIZER_PATH`.

The generic runner's retry cleanup only removes unfinished `result.json`
checkpoints (`running`, `task_error`, or unreadable files). Task failures such
as CAPTCHA, step limits, and invalid actions remain in the record. Judge
outputs preserve missing/invalid counts; pass `TASKS=<benchmark.jsonl>` to the
operation judge launcher for the complete benchmark denominator.
