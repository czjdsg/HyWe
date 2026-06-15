# HybridDataConversion

Convert raw search/operation agent trajectories (RedSearcher, OpenSeeker,
MolmoWeb, WebChain, ...) into the single-step training-sample format expected
by [HybridWebAgent](https://github.com/czjdsg/HybridWebAgent) (hybrid mode).

Pipeline is two-stage per source:

| Stage | What it does | LLM? |
|---|---|---|
| **Stage 1 — Skeleton** | Slice source trajectories into per-step records; map tool names + arguments to hybrid format; drop unsupportable trajectories. | no |
| **Stage 2 — LLM Rewrite** | For each skeleton step, call an LLM to fill `this_turn` / `cumulative_progress` / `remaining_subgoals` / `next_action_rationale` / `action_description` (search) or screenshot-grounded fields (op), and emit a fully-formed hybrid sample. | yes |

## Layout

```
HybridDataConversion/
├── README.md
├── stage1_skeleton.py                # SEARCH stage-1 (RedSearcher + OpenSeeker)
├── stage2_rewrite.py                 # SEARCH stage-2 driver
├── molmoweb_stage1_convert.py        # OP/molmoweb stage-1
├── molmoweb_stage2_rewrite.py        # OP/molmoweb stage-2 driver (multimodal: screenshot in)
├── webchain_stage1_convert.py        # OP/webchain stage-1
├── webchain_stage2_rewrite.py        # OP/webchain stage-2 driver (multimodal)
├── lib/
│   ├── parse_source.py               # extract <think>/<tool_call>/<answer> from raw assistant text
│   ├── tool_mapping.py               # source tool → hybrid tool deterministic mapping
│   ├── stage2_rewriter.py            # SEARCH stage-2 rewriter prompt + client
│   ├── molmoweb_stage2_rewriter.py   # OP/molmoweb rewriter prompt + client (screenshot)
│   ├── webchain_stage2_rewriter.py   # OP/webchain rewriter prompt + client (screenshot)
│   ├── molmoweb_to_hybrid.py         # OP/molmoweb stage-1 mapping helpers
│   ├── webchain_to_hybrid.py         # OP/webchain stage-1 mapping helpers
│   └── judge.py                      # LLM-as-judge utility (task-success / answer eval)
├── scripts/
│   ├── build_search_hybrid_routed_train.py   # build final SFT messages JSONL
│   └── search_rerun_partial_after_full.py    # resume / patch partial rewrites
└── viewer/
    └── src_vs_rewrite_viewer.py      # SOURCE ↔ REWRITE side-by-side diagnostic viewer
                                      #   - search:  tool_call before/after mapping +
                                      #              source_think + observation
                                      #   - op:      screenshot + source_thought +
                                      #              merge_note + merged source steps
                                      #   highlights negative-feedback phrases and
                                      #   shows the original step_quality_reason inline
```

## Stage 1 — quick start (SEARCH)

Dry-run on 5 trajectories from each source:

```bash
python stage1_skeleton.py --source both --limit 5 --tag smoke --out-dir dry_run
```

Full run:

```bash
python stage1_skeleton.py --source both --out-dir data
```

Flags:
- `--source openseeker|redsearcher|both`
- `--limit N` cap traj per source
- `--openseeker-only-correct` keep only OpenSeeker `Correct`-labeled trajectories
- `--tag NAME` suffix the output filename

## Stage 1 — output schema (SEARCH)

Each line is one source assistant turn:

```json
{
  "source": "openseeker",
  "source_traj_id": "openseeker:line:123",
  "step_idx": 7,
  "total_steps": 23,
  "is_initial": false,
  "is_final": false,
  "language": "en",
  "task": "<source question>",

  "source_think": "<text inside <think>...</think>>",
  "source_raw_assistant": "<full original assistant text>",

  "mapped_tool_call": {"name": "search", "arguments": {"query": ["..."]}},
  "mapped_answer": null,
  "source_observation_after": "<next tool_response text>",

  "meta": {
    "ground_truth_answer": "...",
    "trajectory_correctness": "Correct" | "Incorrect" | null,
    "tool_used": "search"
  }
}
```

## Stage 2 — rewrite (SEARCH)

```bash
python stage2_rewrite.py \
  --inputs data/openseeker_skeleton_correct.jsonl,data/redsearcher_skeleton_correct.jsonl \
  --base-url $OPENAI_BASE_URL --model $MODEL --api-key $OPENAI_API_KEY \
  --out-path data/hybrid_search_rewrite.jsonl
```

The rewriter is per-trajectory sequential (cumulative_progress carries
forward) and parallel across trajectories via a thread pool.

## Operation pipelines (molmoweb / webchain)

Mirror the search flow but multimodal: stage 2 always sends the BEFORE-action
screenshot as the second user-content item.

```bash
python molmoweb_stage1_convert.py --out data/hybrid_op_molmoweb_skeleton.jsonl
python molmoweb_stage2_rewrite.py  --in data/hybrid_op_molmoweb_skeleton.jsonl \
  --out data/hybrid_op_molmoweb_full.jsonl --base-url $BASE --model $MODEL

python webchain_stage1_convert.py --out data/hybrid_op_webchain_skeleton.jsonl
python webchain_stage2_rewrite.py  --in data/hybrid_op_webchain_skeleton.jsonl \
  --out data/hybrid_op_webchain_full.jsonl --base-url $BASE --model $MODEL
```

OP records additionally carry `screenshot_ref`, `source_thought` (human
annotator's free-form reasoning), `src_merge_indices` (which source steps
were merged into this hybrid step) and `merge_note`.

## Tool mapping (Stage 1, SEARCH)

| Source tool | Source params | Hybrid tool | Hybrid params |
|---|---|---|---|
| `search` (OpenSeeker) | `{query: array<str>}` | `search` | unchanged |
| `search` (RedSearcher) | `{query: str, topn?, source?}` | `search` | `{query: [str]}` (wrap, drop extras) |
| `visit` (both) | `{url, goal}` | `visit` | `{url, goal, parse_type:"auto"}` |
| `google_scholar` (RED) | `{query: str}` | `scholar` | `{query: [str]}` |
| `PythonInterpreter` (RED) | `{}` + outer `<code>` block | `code_sandbox` | `{code, language:"python"}` |
| `google_maps` (RED) | — | ❌ no equivalent | trajectory dropped |

## Stage 1 — what's dropped, why

| Reason | Default behavior |
|---|---|
| Trajectory contains `google_maps` | Drop entire trajectory |
| Any assistant turn has `>1 tool_call` | Drop entire trajectory |
| Final turn has empty `<answer>` text | Drop entire trajectory |
| Non-final assistant turn has no `tool_call` | Drop entire trajectory (malformed) |
| OpenSeeker `Incorrect` (only if `--openseeker-only-correct`) | Filtered out |

Run prints a JSON report of `drop_reasons` and totals at the end.

## Build final SFT messages

```bash
python scripts/build_search_hybrid_routed_train.py \
  --rewrite data/hybrid_search_rewrite.jsonl \
  --skeletons data/openseeker_skeleton_correct.jsonl,data/redsearcher_skeleton_correct.jsonl \
  --out data/hybrid_search_train_messages.jsonl
```

This stitches per-step rewrites into full `{messages: [...]}` SFT samples
that match the HybridWebAgent inference prompt format (system + per-step
user containing `Previous Thought` / `Recent Actions` / `Previous
Observation`, target = the rewrite JSON).

## Diagnostic viewer

```bash
python viewer/src_vs_rewrite_viewer.py 8088 \
  --skeleton data/redsearcher_skeleton_correct.jsonl,data/openseeker_skeleton_correct.jsonl \
  --rewrite  data/hybrid_search_rewrite.jsonl \
  --op       data/hybrid_op_molmoweb_full.jsonl,data/hybrid_op_webchain_full.jsonl
```

Side-by-side per step:
- **left** = source (raw original format + parsed think + tool_call BEFORE mapping + observation_after; or, for op, the BEFORE-action screenshot + source_thought + merge_note)
- **right** = rewrite (mapped tool_call + this_turn + cumulative_progress + remaining_subgoals + step_quality + step_quality_reason)

Highlights:
- yellow: negative-feedback phrases in source/rewrite text ("不理想 / 换个思路 / no results / instead ...")
- red words inside the orange `step_quality_reason` chip: backward-looking words ("previous / the result / was too / returned / 上一步 / 搜索结果 ...") — direct visual evidence when the quality label is actually grading the PREVIOUS step's action.
- ⛓ purple chip: for op steps, shows merged source step numbers + merge_note.
