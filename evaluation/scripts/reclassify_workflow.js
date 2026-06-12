export const meta = {
  name: 'reclassify-defer-dual-axis',
  description: 'Re-classify 494 defer tasks with dual-axis (answer+evidence) support, then adversarially QC each shard',
  whenToUse: 'Regenerating point_classification caches with multi-axis labels',
  phases: [
    { title: 'Classify', detail: 'one agent per ~20-task shard, emits axes per point' },
    { title: 'QC', detail: 'adversarial reviewer re-checks every point of the shard' },
    { title: 'Repair', detail: 'fix only the points QC flagged' },
  ],
}

// ---- shared classification rubric (mirrors prompts/defer_classify_points.txt) ----
const RUBRIC = `
You classify each answer key point of a web-agent task into how it must be VERIFIED.
Each point gets a non-empty "axes" list drawn from:

- "answer"  : the FINAL ANSWER TEXT itself must explicitly contain this information
              (names a specific entity/product/recipe/place; states a value/list/address/time/details).
              Checked against the answer string.
- "evidence": this requirement CANNOT be confirmed from the answer text alone; it must be
              verified from the agent's TRAJECTORY (visited pages, visible page state, search/
              visit/code outputs, screenshots). E.g. a numeric/status constraint that must hold
              on the source page (rating >= 4, in-stock, sorted order, "lowest among results"),
              "sourced from site X" (must actually reach that page), a filter/sort the page must show.

DUAL = both axes. Assign BOTH only when the requirement is genuinely dual: the fact must be
STATED in the answer AND be demonstrably TRUE on the source page.
  - "report the rating, which must be above 4.0"  -> ["answer","evidence"]  (answer states it; page proves >4.0)
  - "give a specific in-stock product"            -> ["answer","evidence"]  (answer names it; page proves in-stock)
Single "answer" only: purely about answer-text content with no page-state fact beyond the answer.
  - "detail the cooking method", "list the ingredients", "give the total time"  -> ["answer"]
Single "evidence" only: pure provenance/filter/sort/source constraint the answer need not restate.
  - "sourced from X", "evidence trail shows sorting", "filtered by red color"    -> ["evidence"]

Rules:
1. Do NOT default to dual; assign it only when BOTH are truly required.
2. Use the task wording to disambiguate. Base it on what the requirement DEMANDS, not on success.
3. Do not merge/split/reword/drop/add points. Exactly one classification per input point, in order.
4. "axes" must be ordered with "evidence" first when present, then "answer" (e.g. ["evidence","answer"]).
`

const CLASSIFY_SCHEMA = {
  type: 'object',
  required: ['tasks'],
  properties: {
    tasks: {
      type: 'array',
      items: {
        type: 'object',
        required: ['task_id', 'dataset', 'classifications'],
        properties: {
          task_id: { type: 'string' },
          dataset: { type: 'string' },
          classifications: {
            type: 'array',
            items: {
              type: 'object',
              required: ['point_id', 'axes', 'reason'],
              properties: {
                point_id: { type: 'integer' },
                axes: {
                  type: 'array',
                  minItems: 1,
                  maxItems: 2,
                  items: { enum: ['answer', 'evidence'] },
                },
                reason: { type: 'string' },
              },
            },
          },
        },
      },
    },
  },
}

const QC_SCHEMA = {
  type: 'object',
  required: ['flagged'],
  properties: {
    // points the reviewer believes are mislabeled, with the corrected axes
    flagged: {
      type: 'array',
      items: {
        type: 'object',
        required: ['task_id', 'point_id', 'corrected_axes', 'why'],
        properties: {
          task_id: { type: 'string' },
          point_id: { type: 'integer' },
          corrected_axes: {
            type: 'array', minItems: 1, maxItems: 2,
            items: { enum: ['answer', 'evidence'] },
          },
          why: { type: 'string' },
        },
      },
    },
  },
}

const N = args && args.n_shards ? args.n_shards : 25
log(`Re-classifying ${N} shards with dual-axis support`)

const shardIds = Array.from({ length: N }, (_, i) => i)

// pipeline: classify a shard -> QC the shard's classifications -> apply QC corrections
const results = await pipeline(
  shardIds,
  // STAGE 1: classify
  (shardId) => agent(
    `${RUBRIC}\n\n` +
    `Read the shard file at outputs/reclassify_shards/shard_${String(shardId).padStart(2, '0')}.json ` +
    `(a JSON array of tasks; each has dataset, task_id, question, answer_key_points (ordered), ` +
    `and current_kinds for reference only — do NOT trust current_kinds blindly, re-decide from scratch). ` +
    `For EVERY task and EVERY answer key point, decide its axes. ` +
    `Return all tasks in the shard. point_id is the 0-based index of the point within that task's answer_key_points.`,
    { label: `classify:shard_${shardId}`, phase: 'Classify', schema: CLASSIFY_SCHEMA }
  ),
  // STAGE 2: adversarial QC of the same shard
  (classified, shardId) => agent(
    `${RUBRIC}\n\n` +
    `You are an ADVERSARIAL reviewer. Read the original shard at ` +
    `outputs/reclassify_shards/shard_${String(shardId).padStart(2, '0')}.json to see each task's ` +
    `question and answer_key_points. Here is a prior classification to audit:\n\n` +
    `${JSON.stringify(classified)}\n\n` +
    `Re-derive the correct axes for every point independently. FLAG a point ONLY if its axes are wrong ` +
    `under the rubric — common errors: (a) a dual requirement labeled single (e.g. "rating above 4.0" ` +
    `that must be BOTH reported in the answer AND true on the page got only "evidence"); ` +
    `(b) a pure provenance/filter/sort constraint wrongly given "answer"; ` +
    `(c) a pure answer-content requirement wrongly given "evidence". ` +
    `For each flagged point give corrected_axes and a one-line why. If everything is correct, return an empty flagged list. ` +
    `Be precise: do not flag defensible judgment calls, only genuine errors.`,
    { label: `qc:shard_${shardId}`, phase: 'QC', schema: QC_SCHEMA }
  ).then((qc) => ({ shardId, classified, qc })),
)

// Apply QC corrections in-process (deterministic) and return per-shard merged result.
const merged = results.filter(Boolean).map(({ shardId, classified, qc }) => {
  const flags = {}
  for (const f of (qc.flagged || [])) {
    flags[`${f.task_id}::${f.point_id}`] = f.corrected_axes
  }
  let applied = 0
  for (const t of classified.tasks) {
    for (const c of t.classifications) {
      const key = `${t.task_id}::${c.point_id}`
      if (flags[key]) {
        c.axes_before = c.axes
        c.axes = flags[key]
        c.qc_corrected = true
        applied++
      }
    }
  }
  return { shardId, tasks: classified.tasks, qc_flags: (qc.flagged || []).length, qc_applied: applied }
})

const totalTasks = merged.reduce((a, m) => a + m.tasks.length, 0)
const totalFlags = merged.reduce((a, m) => a + m.qc_flags, 0)
log(`Classified ${totalTasks} tasks across ${merged.length} shards; QC flagged ${totalFlags} points`)

return { shards: merged, totalTasks, totalFlags }
