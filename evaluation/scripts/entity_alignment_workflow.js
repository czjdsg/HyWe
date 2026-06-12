export const meta = {
  name: 'entity-alignment-design',
  description: 'Audit the entity-binding gap in the stateful defer judge and design+stress-test fixes',
  whenToUse: 'Deciding how to ensure per-point evidence binds to the same entity the final answer names',
  phases: [
    { title: 'Audit', detail: 'map exactly where entity identity is/ isn\'t enforced in the evaluator' },
    { title: 'Design', detail: 'independent candidate solutions to the cross-point binding gap' },
    { title: 'StressTest', detail: 'adversarially break each design against concrete failure trajectories' },
    { title: 'Synthesize', detail: 'pick + combine into one recommended design' },
  ],
}

const FILE = 'scripts/evaluate_classified_web_tasks.py'
const PROMPTS = 'prompts/defer_update_state.txt, prompts/defer_finalize.txt, prompts/defer_classify_points.txt'

const PROBLEM = `
PROBLEM — entity alignment / cross-point binding gap in the stateful defer judge:

The judge classifies an answer's required attributes into per-point axes (answer/evidence) and
verifies each independently while walking the trajectory step by step. Each point's evidence axis
is marked "satisfied" as soon as SOME step shows a page meeting that property. Each axis keeps its
own supporting_step_ids / evidence_log.

FAILURE MODE the user is worried about: a multi-step trajectory may show product/recipe/entity X
satisfying attribute A at step 5, and a DIFFERENT entity Y satisfying attribute B at step 9. Both
points get marked satisfied. But the FINAL ANSWER names entity Y. Attribute A was only ever verified
on X, never on the answered entity Y. The judge wrongly counts the task solved because it never
bound "the evidence for each attribute must all come from the SAME entity, and that entity must be
the one the final answer commits to."

Concretely: "find an air fryer with digital display AND auto-shutoff AND under \$100". Agent views
fryer X (has digital display), then fryer Y (auto-shutoff, \$89), answers "buy Y". Point[digital
display] was satisfied from X, not Y. Y may have no digital display. Judge says solved. WRONG.

Current weak defense: finalize() lets an answer-only verdict mark an evidence-only point contradicted
"if the answer commits to a different entity than the trajectory supported", but this is a single
LLM judgment with no structured entity identity to compare, no per-step entity tag, and it does not
fire for dual-axis points whose evidence axis was satisfied on the wrong entity.
`

phase('Audit')
const audit = await agent(
  `${PROBLEM}\n\n` +
  `Read ${FILE} (focus: init_point_states, _update_axis, apply_step_update, update_state_with_step, ` +
  `walk_trajectory, finalize_defer_state, recompute_aggregate, render_memory_block, build_step_block_text) ` +
  `and the prompts ${PROMPTS}. ` +
  `Produce a precise audit: ` +
  `(1) Every place an entity identity COULD be tracked but currently isn't (step records, axis state, memory block). ` +
  `(2) Exactly what the current finalize consistency-check can and cannot catch, with line refs. ` +
  `(3) Whether per-step records carry any entity handle (URL, product id, page title, recipe name) that a tag could key on — quote the step-record fields available via build_step_block_text. ` +
  `(4) The dual-axis sub-case: when evidence axis is satisfied on entity X but the answer names Y, does anything currently catch it? ` +
  `Return findings as structured JSON.`,
  { label: 'audit:current-binding', phase: 'Audit', schema: {
    type: 'object', required: ['trackable_entity_handles', 'finalize_catches', 'finalize_misses', 'step_record_fields', 'dual_axis_gap', 'key_line_refs'],
    properties: {
      trackable_entity_handles: { type: 'array', items: { type: 'string' } },
      finalize_catches: { type: 'array', items: { type: 'string' } },
      finalize_misses: { type: 'array', items: { type: 'string' } },
      step_record_fields: { type: 'array', items: { type: 'string' } },
      dual_axis_gap: { type: 'string' },
      key_line_refs: { type: 'array', items: { type: 'string' } },
    },
  } }
)

phase('Design')
// Three independent designers, each told to take a distinct stance, so we get diverse approaches.
const STANCES = [
  { key: 'entity-tag', stance: 'Track an explicit entity identity per satisfied axis. Each time an axis is marked satisfied, record WHICH entity (URL/title/name) the evidence came from. At finalize, require all satisfied axes of a point — and across points that refer to one answer entity — to share the entity the answer names. Design the data structure, the per-step LLM output change, and the finalize check.' },
  { key: 'answer-first-rebind', stance: 'Flip the order: at finalize, extract the entity the answer commits to, then RE-VERIFY each required attribute specifically against the trajectory evidence FOR THAT ENTITY ONLY (re-scan supporting steps, discard evidence from other entities). Walk-phase satisfaction becomes provisional candidate evidence, not final. Design how to re-scan and what memory the walk must keep to enable it.' },
  { key: 'candidate-grouping', stance: 'During the walk, group evidence by candidate entity (maintain a per-candidate attribute scorecard, not a per-point global flag). A point is only "satisfied" within a candidate. At finalize, select the answered candidate and check ALL its attributes are satisfied on THAT candidate. Design the candidate-keyed memory and how points map onto it.' },
]
const designs = await parallel(STANCES.map(s => () =>
  agent(
    `${PROBLEM}\n\nAUDIT FINDINGS:\n${JSON.stringify(audit)}\n\n` +
    `Design a concrete fix taking THIS stance:\n${s.stance}\n\n` +
    `Be concrete about: the data-structure change (what fields, where), the prompt change (what each ` +
    `per-step / finalize call must now output), the code change (which functions: init_point_states, ` +
    `_update_axis, finalize_defer_state, recompute_aggregate), backward-compat impact on kind/kinds and ` +
    `stage projections, and cost (extra LLM calls?). State the design's OWN weakest point honestly.`,
    { label: `design:${s.key}`, phase: 'Design', schema: {
      type: 'object', required: ['name', 'data_structure', 'prompt_change', 'code_change', 'compat_impact', 'extra_cost', 'weakest_point'],
      properties: {
        name: { type: 'string' },
        data_structure: { type: 'string' },
        prompt_change: { type: 'string' },
        code_change: { type: 'string' },
        compat_impact: { type: 'string' },
        extra_cost: { type: 'string' },
        weakest_point: { type: 'string' },
      },
    } }
  ).then(d => ({ ...d, key: s.key }))
))

phase('StressTest')
// Adversarially attack each design with concrete trajectories designed to slip through.
const ATTACKS = [
  'Entity with NO stable URL/id (single-page app, infinite scroll list, results all on one URL) — can the entity handle even be formed?',
  'Answer names entity Y but ALSO restates attribute A correctly (A happens to be true of Y too) — does the fix still require A to be VERIFIED on Y, or does it accept the answer\'s claim?',
  'Two genuinely different candidates both partially qualify; agent compares them across many steps then picks one. Does the fix correctly attribute each attribute to the chosen one without false-contradicting?',
  'Attribute verified on the chosen entity at step 3, but a later step shows a DIFFERENT entity — does the fix avoid spuriously rebinding to the later entity?',
]
const stress = await parallel(designs.filter(Boolean).flatMap(d =>
  ATTACKS.map((atk, i) => () =>
    agent(
      `Design under test:\n${JSON.stringify(d)}\n\n` +
      `Adversarial trajectory / attack:\n${atk}\n\n` +
      `Walk through whether this design produces the CORRECT verdict on this attack. ` +
      `Be a skeptic: assume it fails unless you can show it holds. If it fails, say exactly how and ` +
      `what minimal patch would fix it. Verdict: holds | fails | partial.`,
      { label: `stress:${d.key}:${i}`, phase: 'StressTest', schema: {
        type: 'object', required: ['design_key', 'attack_index', 'verdict', 'why', 'patch'],
        properties: {
          design_key: { type: 'string' },
          attack_index: { type: 'integer' },
          verdict: { enum: ['holds', 'fails', 'partial'] },
          why: { type: 'string' },
          patch: { type: 'string' },
        },
      } }
    ).then(r => ({ ...r, design_key: d.key, attack_index: i }))
  )
))

phase('Synthesize')
const synth = await agent(
  `${PROBLEM}\n\nAUDIT:\n${JSON.stringify(audit)}\n\n` +
  `DESIGNS:\n${JSON.stringify(designs.filter(Boolean))}\n\n` +
  `STRESS-TEST RESULTS:\n${JSON.stringify(stress.filter(Boolean))}\n\n` +
  `Synthesize ONE recommended design. It may combine elements from multiple stances. It must survive ` +
  `all four attacks (or explicitly note residual risk). Specify the exact minimal changes: which fields ` +
  `to add to point/axis state, which prompts to edit and how, which functions to change, extra LLM cost, ` +
  `and the backward-compat story (kind/kinds, stage projections, existing dual-axis caches). Give a ` +
  `concrete worked example on the air-fryer X/Y trajectory showing the corrected verdict. ` +
  `Also state what does NOT need to change.`,
  { label: 'synthesize:recommendation', phase: 'Synthesize', schema: {
    type: 'object', required: ['recommended_design', 'state_changes', 'prompt_changes', 'code_changes', 'extra_cost', 'compat_story', 'worked_example', 'no_change_needed', 'residual_risk'],
    properties: {
      recommended_design: { type: 'string' },
      state_changes: { type: 'string' },
      prompt_changes: { type: 'string' },
      code_changes: { type: 'string' },
      extra_cost: { type: 'string' },
      compat_story: { type: 'string' },
      worked_example: { type: 'string' },
      no_change_needed: { type: 'string' },
      residual_risk: { type: 'string' },
    },
  } }
)

return { audit, designs: designs.filter(Boolean), stress: stress.filter(Boolean), synth }
