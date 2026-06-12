# Answer Eval Framework

This directory contains a lightweight custom evaluation framework for rewriting task criteria into answer-only key points and later judging final answers against them.

Current contents:

- `prompts/rewrite_insta_criteria_answer_only.txt`
  Prompt used to rewrite Insta native criteria into answer-only key points.
- `scripts/rewrite_insta_criteria.py`
  Batch pipeline that calls an OpenAI-compatible model and writes rewritten criteria to JSONL.
- `prompts/judge_final_answer_only.txt`
  Prompt for answer-only judging against rewritten key points.
- `scripts/evaluate_final_answers_only.py`
  Batch pipeline for Insta answer-only evaluation.
- `prompts/judge_reference_answer.txt`
  Prompt for direct answer-vs-reference judging.
- `scripts/evaluate_reference_answers.py`
  Unified pipeline for FRAMES, GAIA validation, BrowseComp, or any run dir with embedded reference answers.
- `prompts/classify_validation_type.txt`
  Prompt for classifying tasks into direct-answer, webpage-content, intermediate-steps, or custom-method validation types.
- `scripts/classify_task_validation_type.py`
  Batch pipeline for classifying non-human-verification tasks across benchmark JSONL files.
- `scripts/evaluate_classified_web_tasks.py`
  Unified five-route evaluator for annotated web tasks:
  `static_answer`, `target_urls`, `reveal_script`, `defer_framework`, `operation_key_points`.
- `ANNOTATION_INSTRUCTIONS.md`
  Human annotation guide for Online-Mind2Web, WebVoyager, and Insta-v3. Current annotation uses only methods 1, 2, 4, and 5; method 4 is displayed as "轨迹证据核验".

Default input:

- `/data/czj/HybridWebAgentBenchmarks/data_samples_v2/web_insta_v3_test50_nohuman_seed42.jsonl`

Example:

```bash
python3 /data/czj/answer_eval_framework/scripts/rewrite_insta_criteria.py \
  --api-key YOUR_KEY \
  --base-url http://HOST:PORT/v1 \
  --model kimi-k2.5
```

Output:

- JSONL with one record per task under `outputs/`
- Summary JSON with counts and basic metadata

Unified reference-answer evaluation example:

```bash
python3 /data/czj/answer_eval_framework/scripts/evaluate_reference_answers.py \
  --run-dir /data/czj/HybridWebAgentBenchmarks/run_v2/frames_50_kimi_k25_summary8001_w4_max100_20260416 \
  --run-dir /data/czj/HybridWebAgentBenchmarks/run_v2/gaia_validation50_kimi_k25_summary8001_w4_max100_20260415 \
  --run-dir /data/czj/HybridWebAgentBenchmarks/run_v2/browsecomp_50_kimi_k25_summary8001_w4_max100_20260416 \
  --output-dir /data/czj/answer_eval_framework/outputs/reference_eval_kimi-k2.5 \
  --api-key YOUR_KEY \
  --base-url http://HOST:PORT/v1 \
  --model kimi-k2.5
```

Validation-type classification example:

```bash
python3 /data/czj/answer_eval_framework/scripts/classify_task_validation_type.py \
  --api-key YOUR_KEY \
  --base-url http://HOST:PORT/v1 \
  --model kimi-k2.5
```

By default, this classification script runs on:

- `/data/czj/HybridWebAgentBenchmarks/data_v2/web_online_mind2web_256_nohuman_final.jsonl`
- `/data/czj/HybridWebAgentBenchmarks/data_v2/web_webvoyager_643_final.jsonl`
- `/data/czj/HybridWebAgentBenchmarks/data_samples_v2/web_insta_v3_test50_nohuman_seed42.jsonl`

It filters to `needs_human_verification = false` by default and writes:

- one JSONL record per task with `validation_type`, `confidence`, `reason`, `annotation_focus`, and optional custom-method fields
- a summary JSON with overall and per-dataset counts

Classified WebVoyager evaluation example:

```bash
python3 /data/czj/answer_eval_framework/scripts/evaluate_classified_web_tasks.py \
  --run-dir /data/czj/HybridWebAgentBenchmarks/run_v2/webvoyager_643_final_kimi_k25_summary8001_w4_max100_20260421 \
  --annotations /data/czj/HybridWebAgentBenchmarks/data_v2/annotation_service/web_webvoyager_643_annotations.json \
  --classification /data/czj/answer_eval_framework/outputs/online_m2w_webvoyager.validation_type_classification.kimi-k2.5.jsonl \
  --classification-dataset-file web_webvoyager_643_final.jsonl \
  --answer-key-points /path/to/answer_only_keypoints.jsonl \
  --output-dir /data/czj/answer_eval_framework/outputs/webvoyager_643_classified_eval \
  --browser-snapshot-command 'browser_snapshot {url}' \
  --api-key YOUR_KEY \
  --base-url http://HOST:PORT/v1 \
  --model kimi-k2.5
```

Notes for the five-route evaluator:

- `static_answer`: treats manually annotated `reference_answers` as required answer key points and checks whether the agent final answer covers all of them.
- `target_urls`: fetches each annotated target URL through the local Jina Reader key pool at `http://127.0.0.1:8766` by default, captures a browser full-page screenshot, then judges answer correctness against both evidence sources. Override with `--jina-reader-base-url` or `JINA_READER_BASE_URL` if needed.
- `reveal_script`: executes the annotated Python snippet, reads its stdout as evidence, then judges answer correctness against that evidence.
- `defer_framework`: a **stateful, step-by-step judge** that maintains an evaluation
  memory of every answer key point (see `STATEFUL_DEFER_DESIGN.md`). It:
  1. classifies each key point as `answer_bearing` (must appear in the final answer text)
     or `evidence_bearing` (must be verified from the trajectory),
  2. collapses redundant/duplicate navigation steps (lossless, no LLM),
  3. reads the remaining trajectory one step at a time, updating each point's status
     (`pending` / `satisfied` / `partial` / `contradicted`) with a logged evidence trail
     and the supporting step ids,
  4. settles the `answer_bearing` points against the final answer and derives the label
     (pass only if every answer_bearing and every evidence_bearing point is satisfied).
  The full evaluation memory is written to `point_states` / `step_updates` in each result
  row; `stage1`/`stage2`/`stage3` fields are still emitted as a backward-compatible
  projection. Use `--answer-key-points` to provide a JSONL file with `task_id` and
  `answer_key_points`; native `criteria` are not used as key points implicitly.
  LLM calls default to `--defer-temperature 0` for reproducible single-pass scoring;
  raise it (e.g. `0.6`) together with `--repeat-count` for majority-vote sampling.
- `operation_key_points`: a **stateful, timing-aware page-state judge** (see
  `STATEFUL_OPERATION_DESIGN.md`). It shares the per-step engine with `defer_framework`:
  it reads the trajectory one step at a time, tracking each required `page_state`'s
  achievement in an evaluation memory, with timing semantics — `intermediate` states
  only need to be passed through once (stay satisfied after), while `final` states are
  rechecked against the end-of-trajectory state and must still hold at the end. All
  page states must end `satisfied` to pass. Results carry `page_state_results` (the
  evaluation memory) and `step_updates`, plus the legacy `key_point_results` projection.

For `defer_framework`, the Phase A point classification (`answer_bearing` vs `evidence_bearing`) depends only on `(task_id, question, answer_key_points)` and can be precomputed once per `(dataset, model)` pair, then reused across every run / repeat / mode. Use `scripts/classify_defer_answer_points.py` to build the cache and pass it to the evaluator via `--point-classification`. Cache files follow `outputs/<dataset_stem>.point_classification.<model_slug>.jsonl`; on a cache hit Phase A skips the LLM call entirely and the result row carries `classification_of_points.source = "cache"`.

```bash
python3 /data/czj/answer_eval_framework/scripts/classify_defer_answer_points.py \
  --annotations annotations.json \
  --dataset-file web_insta_v3_test50_nohuman_seed42.jsonl \
  --model kimi-k2.5 \
  --base-url http://47.82.159.100:8000/v1 \
  --api-key $OPENAI_API_KEY \
  --workers 8 \
  --resume \
  --output outputs/web_insta_v3_test50_nohuman_seed42.point_classification.kimi-k2.5.jsonl
```

For `target_urls`, screenshots are saved under `webpage_evidence/<mode>/<task_id>/` in the output directory and attached to the judge request as image inputs immediately after their matching evidence block. The screenshot uses Playwright full-page capture and records the page's scroll dimensions.

The legacy `--browser-snapshot-command` option is still available for reveal scripts that call `BROWSER_SNAPSHOT_COMMAND`.
