# Evaluation

The generic runner is `bash scripts/run_hywe.sh TASKS OUT PROFILE` after
sourcing `configs/hywe.env`. `PROFILE=operation` uses 100 steps;
`PROFILE=search` uses 300. Starting mode follows each record's `web` field.

## HyWeBench

`benchmarks/hywebench/hywebench_100.jsonl` contains 100 public tasks with fixed
human-written checklists: 309 information items and 247 operation items.
Checklists are not put in the policy prompt. S-start tasks have an empty `web`;
do not construct a start URL from `domain` because that changes their start mode.

```bash
bash scripts/run_hywe.sh benchmarks/hywebench/hywebench_100.jsonl runs/hywebench operation
bash evaluation/hywebench/judge.sh runs/hywebench
python evaluation/hywebench/score.py evaluation/hywebench/paper_verdicts/*.jsonl
```

The judge consumes the fixed checklist directly; it does not ask the LLM to
split each model's trajectory into different subgoals. Screenshots show action
coordinates. Information items require correct answers, operation items require
recorded end-state evidence. The judge reports per-item evidence and blocking
separately. Full success requires all items; Info./Op. are micro-averages over
the corresponding items. Missing judgments count as zero achieved items in
the full 100-task denominator. The last successful verdict for a task wins.

`judge.sh` sets `HJ_OP_SEARCH_RULE=0`, matching the archived paper judgment
configuration. The stricter rule rejecting operation items supported only by
search steps was added later and is available by setting it to 1. Fabricated
screenshot evidence on image-less trajectories is retried, then rejected on
the last judgment pass. Archived verdicts are sufficient to reproduce the
reported success/subgoal tables; full historic screenshots for every baseline
are not included in this code release.

## Online-Mind2Web and WebVoyager

```bash
python benchmarks/prepare_benchmarks.py online_mind2web
bash scripts/run_hywe.sh benchmarks/data/online_mind2web_300.jsonl runs/m2w operation
TASKS=benchmarks/data/online_mind2web_300.jsonl bash evaluation/online_mind2web/judge.sh runs/m2w

bash scripts/run_hywe.sh benchmarks/webvoyager/webvoyager_643.jsonl runs/wv operation
TASKS=benchmarks/webvoyager/webvoyager_643.jsonl bash evaluation/webvoyager/judge.sh runs/wv
```

Set `OPENAI_API_KEY` and optional `OPENAI_BASE_URL` for the operation judges.
`M2W_JUDGE_MODEL` defaults to `o4-mini`; `WV_JUDGE_MODEL` defaults to `gpt-4o`.
The policy's `MODEL` does not override either judge. Online-Mind2Web uses
adapted WebJudge with evidence-relevance threshold 3 and a larger completion
budget for reasoning models. WebVoyager uses the last 15 observations.
Both adapters treat textual tool returns as evidence alongside screenshots.

Pass `TASKS` for the complete benchmark denominator. Otherwise the launcher
scores over all results in the run directory. Missing/invalid judgments are
reported and count as zero success rather than shrinking the denominator.
Judge outputs resume at completed tasks; changing judge settings requires a
new output directory. Browser/site failures remain in the run artifacts.

## Deep search

```bash
python benchmarks/prepare_benchmarks.py xbench
bash scripts/run_hywe.sh benchmarks/data/xbench_deepsearch_2510_100.jsonl runs/xbench search
bash evaluation/search/judge.sh runs/xbench
```

HLE and BrowseComp use the same runner and judge with their converted files.
The task `answer` is retained as reference metadata and excluded from the
policy prompt. Exact answer matches are accepted directly; other predictions
receive five independent judge votes at temperature 0.6, requiring a majority.
No-majority results remain explicit errors. `JUDGE_MODEL` defaults to `kimi-k3`.

Source data: [BrowseComp](https://github.com/openai/simple-evals),
[xbench-DeepSearch-2510](https://huggingface.co/datasets/xbench/DeepSearch-2510),
[HLE](https://huggingface.co/datasets/cais/hle),
[Online-Mind2Web](https://huggingface.co/datasets/osunlp/Online-Mind2Web),
and [WebVoyager](https://github.com/MinorJerry/WebVoyager).
Converters retain references, image attachments, and dataset provenance.
Encrypted and gated third-party task sets are downloaded locally rather than
redistributed as plaintext in this repository.
