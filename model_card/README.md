---
license: apache-2.0
base_model: Qwen/Qwen3.6-35B-A3B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- web-agent
- tool-use
- browser-agent
- react
datasets:
- yzxjb/HyWe-Hybrid-SFT-Data
- yzxjb/HyWe-Search-SFT-Data
---

# HyWe-35B

HyWe-35B is the fine-tuned agent from **HyWe: Unifying Search and Operation
for Real-World Hybrid Web Tasks**. It combines text-based search and rendered
browser operation in one tool-using agent, using `switch` to change modes.
The checkpoint is fine-tuned from Qwen3.6-35B-A3B and is stored in BF16 across
16 safetensors shards (approximately 70.2 GB including configuration files).

💻 [Code, setup, and evaluation](https://github.com/czjdsg/HyWe) ·
🤗 [HyWeBench](https://huggingface.co/datasets/yzxjb/HyWeBench)

## Use with the HyWe framework

```bash
vllm serve yzxjb/HyWe-35B --served-model-name HyWe-35B \
  --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 --max-model-len 224000
```

Use a vLLM build supporting the checkpoint's `qwen3_5_moe` architecture and
adjust GPU count/concurrency to available memory. Set `MODEL=HyWe-35B` and
`BASE_URL=http://127.0.0.1:8000/v1` in the code repository's example config,
then run `scripts/run_hywe.sh`. The framework supplies tool schemas, executes
calls, switches the S/O prompt, and manages context folding.

Model outputs use `<think>...</think>` followed by
`<tool_call>{"name": "...", "arguments": {...}}</tool_call>` or
`<answer>...</answer>`. Browser screenshots have a 1280×720 viewport;
coordinates use a 0–1000 grid. A standard chat UI without the HyWe tools and
context renderer does not reproduce the benchmark setup.

## Data and training

- 🤗 [HyWe-Hybrid-SFT-Data](https://huggingface.co/datasets/yzxjb/HyWe-Hybrid-SFT-Data):
  3,735 hybrid trajectories, 8,055 rendered segments and 45,045 original screenshots.
- 🤗 [HyWe-Search-SFT-Data](https://huggingface.co/datasets/yzxjb/HyWe-Search-SFT-Data):
  the 128K multi-turn conversion of 7,000 XYZ-Aquila-SFT trajectories.

Training used full language-model fine-tuning with the vision encoder and
projector frozen, a 131,072-token sequence length, BF16, an effective batch
size of 32, peak learning rate 2e-5, and one configured epoch. Portable saved
settings and their provenance limits are documented in `training/` in the
code release. Exact GPU type, elapsed training time and source sampling
proportions were not recorded in the saved checkpoint arguments.

## Evaluation

| HyWeBench metric | Result (%) |
| --- | ---: |
| All checklist items achieved | 47.0 |
| Information items achieved | 82.2 |
| Operation items achieved | 68.8 |

These results aggregate archived judgments for the 100-task benchmark, using
the released framework and Kimi-K3 checklist judge. The code repository
includes all 14 comparison/ablation verdict sets and a script to recompute
their rates. New live-site runs can differ because websites, search results
and model services change. The recorded runs used Jina Search; the paper's
appendix describes Serper, a discrepancy documented in the release.

## Limitations

The agent may produce incorrect answers or actions and can fail on CAPTCHA,
authentication, changing pages, and long-horizon tasks. Local scripted smoke
tests validate integration, not learned task-solving performance. Use the
release's benchmark protocols when comparing models.

The checkpoint is distributed under Apache-2.0, consistent with its base model.
