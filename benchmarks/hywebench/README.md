# HyWeBench

[![Code and evaluator](../../assets/badges/code.svg)](https://github.com/czjdsg/HyWe)
[![Hugging Face dataset](../../assets/badges/benchmark.svg)](https://huggingface.co/datasets/yzxjb/HyWeBench)
[![HyWe-35B model](../../assets/badges/model.svg)](https://huggingface.co/yzxjb/HyWe-35B)

**HyWeBench** evaluates tasks that combine web search and browser operation.
It contains **100 human-written tasks** with fixed outcome checklists:
**309 information items** and **247 operation items**. The checklist is shared
across evaluated systems rather than regenerated for each trajectory.

| Property | Released split |
| --- | --- |
| Split | `test` |
| Tasks | 100 |
| Starting mode | 68 search / 32 operation |
| Switching pattern | 68 A3 / 32 A4 |
| Information checklist items | 309 |
| Operation checklist items | 247 |
| Language | English |
| License | Apache-2.0 |

The HF task file `data/test.jsonl` is byte-for-byte identical to
`benchmarks/hywebench/hywebench_100.jsonl` in the code repository.

## Download

🤗 **[Dataset page](https://huggingface.co/datasets/yzxjb/HyWeBench)** ·
📄 **[Direct JSONL download](https://huggingface.co/datasets/yzxjb/HyWeBench/resolve/main/data/test.jsonl)**

```python
from datasets import load_dataset

tasks = load_dataset("yzxjb/HyWeBench", split="test")
print(len(tasks))  # 100
```

For the existing JSONL runner:

```bash
hf download yzxjb/HyWeBench data/test.jsonl --repo-type dataset --local-dir data/hywebench
```

## Schema

| Field | Meaning |
| --- | --- |
| `task_id` | Stable task identifier |
| `question` | Public task instruction |
| `web` | Initial browser URL; empty means start in search mode |
| `domain` | Task's associated site domain |
| `pattern` | Designed switching pattern (`A3`: search → operation; `A4`: operation → search) |
| `start_mode` | `S` for search or `O` for browser operation |
| `checklist` | Fixed items, each with `id`, `goal`, `kind` and `site` |
| `dataset` | `HyWeBench` |

The policy receives the public question and initial URL. Keep the checklist as
evaluation metadata; it is not part of the policy prompt. Preserve the empty
`web` field on search-start tasks instead of constructing a URL from `domain`.

## Run and evaluate

In a checkout of the 💻 **[HyWe code repository](https://github.com/czjdsg/HyWe)**,
configure the policy endpoint and judge as described in the
📘 **[evaluation guide](https://github.com/czjdsg/HyWe/blob/main/evaluation/README.md)**:

```bash
source configs/hywe.env
bash scripts/run_hywe.sh benchmarks/hywebench/hywebench_100.jsonl runs/hywebench operation
bash evaluation/hywebench/judge.sh runs/hywebench
```

When using the downloaded HF JSONL instead of the included copy, pass its path
to the runner and set `TASKS` to the same path when invoking `judge.sh`.

The judge evaluates each checklist item against the recorded text and browser
evidence. Task success requires every item; information and operation rates
are micro-averages over the corresponding items. Missing judgments contribute
zero achieved items in the full benchmark denominator.

## Archived results

| Model | Task success (%) | Information (%) | Operation (%) |
| --- | ---: | ---: | ---: |
| Qwen3.6-35B-A3B base | 28.0 | 74.8 | 43.7 |
| HyWe-35B | 47.0 | 82.2 | 68.8 |

These are historical checklist judgments. All 14 comparison and ablation
verdict sets, the scoring script, and the protocol are available in the
📊 **[evaluation directory](https://github.com/czjdsg/HyWe/tree/main/evaluation/hywebench)**.
Live pages and search results can change, so a new run may differ.

## Related resources

🤗 **[HyWe-35B](https://huggingface.co/yzxjb/HyWe-35B)** ·
🤗 **[Hybrid SFT data](https://huggingface.co/datasets/yzxjb/HyWe-Hybrid-SFT-Data)** ·
🤗 **[Search SFT data](https://huggingface.co/datasets/yzxjb/HyWe-Search-SFT-Data)**
