# Reproducibility record

## Release scope

The release ports the paper's `hybrid_react_v2` harness, teacher phase
controller, capability-probing/task-design/review workflow, SFT export, and
benchmark judges from the research working tree. Local key pools, private
gateways, OSS image publishing, training-cluster paths, and tunnel launchers
are replaced by environment configuration and direct APIs. Screenshots are
inline by default; a generic HTTPS image host remains optional.

## Verified historical results

The 100 HyWeBench task/checklist definitions are shared across all 14 archived
verdict sets. `evaluation/hywebench/score.py` recomputes the task-success and
information/operation-subgoal rates used in the paper. HyWe-35B gives
47.0 / 82.2 / 68.8; the corresponding base gives 28.0 / 74.8 / 43.7.
`evaluation/hywebench/paper_metrics.json` records the expected rounded values.
Raw baseline screenshots and historical search results are not all in the
code repository, so re-aggregation and new live judgment are different checks.

## Search provider discrepancy

The paper appendix says Serper powers `search` and `scholar`, with Jina used
for `visit`. The archived evaluation launchers source an environment that
sets `HYBRID_SEARCH_BACKEND=jina`: `search` used Jina Search, `visit` Jina
Reader, and `scholar` Serper. This release follows that run configuration in
`configs/hywe.env.example` and preserves the alternative Serper backend.
Collection's capability probing defaults to Serper. Provider substitutions
can change which facts an agent can reach and therefore affect live scores.

## Runtime configuration

`scripts/run_hywe.sh` supplies the recorded sampling settings: temperature 1,
top-p 0.95, top-k 20, repetition penalty 1.05, 8,192 output tokens, 100 operation
or 300 search steps, five format retries, two error/format attempts, and five
step-limit restarts. The example env sets the 131,072 / 65,536 fold watermarks,
224,000 serving window, and search-tool loop guards. The original separate
`deepseek-v4-flash` summarizer can be restored with `SUMMARY_*`; the example
reuses the policy endpoint to reduce setup requirements.

The released `HyWe-35B` served name is recognized as a 0–1000 coordinate model.
Custom served names should pass `--coord-scale 1000` explicitly if the name
does not identify its model family.

## Release corrections

Portability checks caught and corrected stale defaults that would prevent a
fresh checkout from following the workflow: collector v8 selection after
module import, relative subprocess paths, tokenizer configuration, judge
model variables shared with the policy, and small WebJudge batches with more
workers than tasks. Infrastructure/probe-judge errors retain diagnostics but
are no longer converted to capability evidence. These corrections affect new
runs; the archived paper verdicts are preserved without rewriting scores.

## Limits

The migrated search dataset's source card lists 10,413 samples, but the
verified current source file contains 10,012 non-empty JSONL records. Its
source SHA-256 and bytes are preserved on HF; the original card is retained
as `MODELSCOPE_README.md` alongside the correction in the new card.
The [migration manifest](dataset-migration-manifest.json) pins the verified
HF revisions and records every source file's size and SHA-256, the observed
sample counts, and the hybrid screenshot-reference check.

Live websites, CAPTCHA policies, search indices, provider models, and API
quotas change. Full training provenance is limited to the saved configuration
documented in `training/`. A passing local scripted smoke run verifies browser,
transport, switching, export/judge plumbing; it does not establish real-model
performance or reproduce a fresh training run.
