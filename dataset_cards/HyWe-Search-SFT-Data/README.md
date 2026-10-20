---
pretty_name: HyWe Search SFT Data
license: apache-2.0
language:
- en
task_categories:
- text-generation
tags:
- web-agent
- web-search
- reasoning
- tool-use
- multi-turn
- supervised-fine-tuning
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: train_aquila_v2_multiturn_128k.jsonl
---

# HyWe-Search-SFT-Data

Text-only, multi-turn training data for search agents. Conversations combine
reasoning, web search, page reading, code execution and final answers in a
consistent tool-use format.

## At a glance

| Property | Value |
| --- | --- |
| Split | `train` |
| Training samples | **10,012** |
| Modality | Text |
| Record format | JSONL containing multi-turn `messages` |
| Context format | 128K |
| Context-fold watermarks | 131,072 / 65,536 tokens |
| Tool names | `search`, `visit`, `code_sandbox` |
| Language | English |
| License | Apache-2.0 |

## Training file

`train_aquila_v2_multiturn_128k.jsonl` contains one conversation per line.
The file is approximately 3.35 GB; stream records when inspecting it.

## Conversation format

Each record contains a `messages` list. Every message has a `role` and text
`content`:

```text
system       Search-agent instructions and tool definitions
user         Task, plus earlier interaction history when applicable
assistant    <think>...</think> and <tool_call>{...}</tool_call>
user         Tool observation
assistant    Next reasoning and tool call
...          Additional interaction turns
assistant    <think>...</think> and <answer>...</answer>
```

The assistant uses the same tool names and argument structure throughout:

| Tool | Arguments |
| --- | --- |
| `search` | `query`: a list of search queries |
| `visit` | `url`, `goal` |
| `code_sandbox` | `language`, `code` |

Long interactions are split at context boundaries. Earlier history is included
in the next record, with older observations folded and recent context retained.
The final part contains the answer. Records can be used for supervised
fine-tuning on their assistant turns.

## Download and read

```bash
hf download yzxjb/HyWe-Search-SFT-Data train_aquila_v2_multiturn_128k.jsonl \
  --repo-type dataset --local-dir hywe-search-data
```

```python
import json
from pathlib import Path

path = Path("hywe-search-data/train_aquila_v2_multiturn_128k.jsonl")
with path.open(encoding="utf-8") as handle:
    sample = json.loads(next(handle))

messages = sample["messages"]
print(len(messages), messages[0]["role"])
```

💻 [Training and message-format tools](https://github.com/czjdsg/HyWe/tree/main/training)

## License

Apache-2.0.
