---
pretty_name: HyWe Hybrid SFT Data
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- web-agent
- browser-agent
- web-search
- multimodal
- tool-use
- multi-turn
- supervised-fine-tuning
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---

# HyWe-Hybrid-SFT-Data

Multi-turn training data for web agents that combine **search-based reasoning**
with **visual browser operation**. Each training record represents one segment
of an interaction, with the task and preceding history as context and the
segment's assistant turns as supervision targets.

## At a glance

| Property | Value |
| --- | ---: |
| Split | `train` |
| Training samples | **8,055** |
| Complete trajectories | **3,735** |
| Interaction steps | **85,590** |
| Search / operation steps | 31,145 / 54,445 |
| Search / operation samples | 3,968 / 4,087 |
| Unique screenshots | **45,045** |
| Screenshot resolution | **1280 × 720**, PNG |
| Coordinate grid | **0–1000** |
| Language | English |
| License | Apache-2.0 |

## Training files

```text
data/train.jsonl          One JSON record per training sample
images_shards/*.tar       Screenshot archives
meta.json                Dataset statistics
```

Extract the screenshot archives at the dataset root. Images then reside at
`images/<prefix>/<hash>.png`, matching the relative paths in the messages.

## Record format

| Field | Description |
| --- | --- |
| `task_id` | Trajectory identifier shared by its segments |
| `segment` | Zero-based segment index |
| `segment_of` | Number of segments in the trajectory |
| `side` | `S` for search or `O` for browser operation |
| `seg_steps` | Number of steps in this segment |
| `ends_with` | Segment-ending action, such as `switch` or `answer` |
| `switch_form` | Designed task pattern (`F1`–`F4`) |
| `batch` | Collection-batch identifier |
| `messages` | Multi-turn conversation with `system`, `user` and `assistant` roles |

Search observations are text. Browser observations include image parts using
the `image_url` message format. User-message content can be either a string
or a list of text and image parts; image paths resolve against the dataset root.

Assistant messages use `<think>...</think>` followed by a JSON tool call inside
`<tool_call>...</tool_call>`, or a final answer inside `<answer>...</answer>`.
The `switch` tool marks a transition between search and browser operation.
Later segments include preceding interaction history as context.

Browser coordinates map to the screenshot as
`x_pixel = x × 1280 / 1000` and `y_pixel = y × 720 / 1000`.
Keep the recorded dates, observations and segment boundaries when preparing
training batches.

## Download and read

```bash
hf download yzxjb/HyWe-Hybrid-SFT-Data --repo-type dataset --local-dir hywe-hybrid-data
for shard in hywe-hybrid-data/images_shards/*.tar; do
  tar -xf "$shard" -C hywe-hybrid-data
done
```

Read the JSONL as a stream:

```python
import json
from pathlib import Path

root = Path("hywe-hybrid-data")
with (root / "data/train.jsonl").open(encoding="utf-8") as handle:
    sample = json.loads(next(handle))

print(sample["task_id"], sample["side"])
messages = sample["messages"]
```

💻 [Training and message-format tools](https://github.com/czjdsg/HyWe/tree/main/training)

## License

Apache-2.0.
