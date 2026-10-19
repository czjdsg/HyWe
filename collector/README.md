# HyWeCollector

Copy and source `configs/collector.env.example` as `configs/collector.env`.
All LLM roles default to **kimi-k3**. The same endpoint may serve every role,
or each role can be overridden. Exploration, fact extraction, execution,
phase verification, trajectory audit, and operation-purity review need vision.
Task synthesis uses Jina embeddings (`JINA_API_KEY`, optional
`JINA_EMBEDDING_BASE_URL` and `JINA_EMBEDDING_MODEL`) separately from the LLM.

Run the shell entry points from the repository root:

| Stage | Entry point | Main outputs |
| --- | --- | --- |
| 1. Capability probing | `scripts/stage1_capability_probing.sh` | `runs/stage1/<domain>/result.json`, `fact_tags.json`, `facts_refiltered.json`, `facts_labeled.json` |
| 2. Blueprint/task synthesis | `scripts/stage2_task_design.sh` | `runs/hybrid_task_design_<TAG>/design_catalog.json`, `tasks.jsonl`, `tasks_dedup.jsonl`, `input_<TAG>.jsonl` |
| 3. Coordinated execution | `scripts/stage3_teacher_rollout.sh` | `runs/hybrid_rollout_<TAG>/rollouts/<task_id>/` |
| 4. Review and refinement | `scripts/stage4_review_refine.sh` | audit/verdict folders, `runs/final_pick_<TAG>/`, `runs/train_<TAG>.opure.jsonl` |

The paper groups blueprint synthesis as Stage 2 and naturalization, coordinated
execution, and review/refinement as Stage 3. The four launchers expose these
operations separately; the stage numbers in filenames are workflow steps.

## Stage 1: probe capability boundaries

```bash
LIMIT=5 SEM_A=2 SEM_B=2 bash collector/scripts/stage1_capability_probing.sh
```

The v8 explorer separates the browser operator from the fact extractor. Facts
pass a value filter and are turned into questions for a search-only agent
with budget `PROBE_MAX_ITER=8`. Its predictions are compared to the observed
values before refiltering and stability annotation. `PIPE_T_A` and `PIPE_T_B`
bound each site's wall time. Partial facts are salvaged after explorer timeout.

The default screened pool contains 2,000 sites. `LIMIT=0` uses all of it;
`POOL` and `OUT` choose another site list and output directory. Per-site logs
are stored in `_logs/` and `pipeline_state.jsonl`. Failed probes retain their
error reasons and receive no capability label; the refilter excludes errors,
including legacy files that assigned a default O label to infrastructure errors.
Rerunning retries sites whose cached probe counts contain errors.

The final S/O prototype catalog was abstracted from these facts and is shipped
in `hybrid_task_design.seed.json`. Re-running Stage 1 does not automatically
rewrite that curated catalog.

## Stage 2: generate tasks

```bash
TAG=demo COUNT=20 SEED=29 bash collector/scripts/stage2_task_design.sh
```

Semantic prototype pairing builds a cached catalog; seeded blueprint sampling
chooses patterns, sites, entity bindings, composition, and difficulty. The LLM
naturalizes a public question and private per-phase plan. Validators reject
drafts that violate site scopes, bindings, component coverage, switching
conditions, or answer schema. The launcher deduplicates exact normalized
questions and IDs against HyWeBench and optional `OLD_TASKS`.

The expanded design pool contains 3,169 entries, extending the original
1,170-site pool. The site pools include derived metadata from
[InSTA-150k-v3](https://huggingface.co/datasets/data-for-agents/insta-150k-v3).
The screened pool and its detail file are included so the default workflow
does not need to redownload InSTA.

## Stage 3: teacher execution

```bash
TAG=demo MAX_WORKERS=2 bash collector/scripts/stage3_teacher_rollout.sh
```

The executor acts through the inference framework. A separate controller
verifies phase completion, binds intermediate results, and gates `switch`
and `answer`. Private plan text is excluded from the executor's public
prompt. Only O-start tasks get a browser start URL. Teacher inputs may use
1000×1000 WEBP screenshots, while the original 1280×720 frames remain on disk
for export. The launcher saves the teacher's phase-control records with the
trajectory and supports `--resume`.

## Stage 4: review, refine, export

```bash
TAG=demo bash collector/scripts/stage4_review_refine.sh
# Individual steps can be resumed explicitly:
TAG=demo bash collector/scripts/stage4_review_refine.sh audit cot site switch coverage merge assemble residue export opurity
```

The chain audits factual progress and private-plan leakage, rewrites leaking
reasoning, verifies site fidelity and switching rationale, checks answer
coverage, merges earlier phase results, assembles survivors, performs residual
leak cleanup, exports segments, and checks operation purity. Benchmark IDs are
excluded during assembly. Failures and reviewer outputs remain on disk.
The scripts use `JUDGE_BASE_URL`, `JUDGE_API_KEY`, and `JUDGE_MODEL`; screenshots
are sent inline by default.

Use a new `TAG` or clear only the affected cached reviewer outputs when changing
judge settings; `--resume` intentionally reuses completed verdicts.
