export const meta = {
  name: 'so-neutrality-audit',
  description: 'Audit defer prompts + evidence guidance for O-side (browser/page) wording bias vs S-side (search/visit/scholar)',
  whenToUse: 'Ensuring the evaluator does not privilege browser-page evidence over search/visit/scholar text evidence',
  phases: [
    { title: 'Audit', detail: 'one auditor per file, find O-biased wording' },
    { title: 'Synthesize', detail: 'consolidate findings + neutral rewrites' },
  ],
}

const CONTEXT = `
BACKGROUND. This is the stateful "defer" judge of a web-agent benchmark. Tasks run in a HYBRID agent
that has TWO equally-valid execution paths:
  - S-side (search-based): search / scholar / visit / code_sandbox  -> evidence is the TEXTUAL result
    (search result snippets, candidate pages, paper metadata, visited page/PDF text, computed output).
  - O-side (operation-based): browser interaction (goto/click/fill/scroll) -> evidence is the resulting
    VISIBLE PAGE STATE + a full-page screenshot.

CRITICAL: for MANY tasks BOTH paths can legitimately solve the task and produce valid evidence. A fact
verified from a SEARCH RESULT or a VISITED PAGE's TEXT is just as valid as a fact read off a browser
SCREENSHOT. The evaluator must NOT privilege the browser/operation path. Wording that implies evidence
must live "on the source PAGE", must be "shown on the page", requires "reaching that page" in a browser,
or that treats screenshots as the canonical evidence while search/visit text is secondary — is BIASED
toward O-side and will wrongly fail correct S-side trajectories.

YOUR JOB: find every phrase that leans O-side (browser/page/screenshot) where an S-side (search/visit/
scholar text) source would satisfy the SAME requirement, and propose a NEUTRAL rewrite that names both
modalities or uses modality-agnostic language (e.g. "established by the trajectory evidence (a search/
visit result OR a visible page/screenshot)"). Do NOT neutralize wording that is GENUINELY page-specific
(e.g. a requirement that literally demands an interactive page action like "apply a red-color filter and
show the filtered list" — that is inherently O-side; flag it as legitimately O-side, not a bias).
`

phase('Audit')
const FILES = [
  { key: 'classify', path: 'prompts/defer_classify_points.txt',
    note: 'Classifies points into answer/evidence axes. Watch for "on the source page", "reaching that page", "the page must satisfy" as the ONLY framing of evidence — evidence can also be a search/visit text result.' },
  { key: 'update', path: 'prompts/defer_update_state.txt',
    note: 'Per-step walk. It DOES enumerate search/scholar/visit/code/browser. Check the status definitions, evidence_log instruction, and screenshot rule 7 for page/screenshot-centric phrasing that ignores text-modality evidence.' },
  { key: 'finalize', path: 'prompts/defer_finalize.txt',
    note: 'Settles verdict. Watch "different entity/PAGE", "the trajectory contradicted", "sourced from page" framing.' },
  { key: 'guidance', path: 'scripts/evaluate_classified_web_tasks.py',
    note: 'ONLY audit the function get_step_evidence_guidance (around line 585-627) and the browser-step branch of compose_observation_text. Compare the per-modality "Primary evidence for this step is..." strings: is the browser one stronger/more authoritative than the search/visit/scholar ones?' },
]

const FINDING_SCHEMA = {
  type: 'object',
  required: ['file', 'findings', 'overall_bias'],
  properties: {
    file: { type: 'string' },
    overall_bias: { enum: ['none', 'mild', 'moderate', 'strong'] },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['quote', 'why_biased', 'neutral_rewrite', 'legitimately_o_side'],
        properties: {
          quote: { type: 'string', description: 'exact biased phrase from the file' },
          why_biased: { type: 'string' },
          neutral_rewrite: { type: 'string', description: 'proposed modality-agnostic replacement' },
          legitimately_o_side: { type: 'boolean', description: 'true if this is genuinely page-action-specific and should NOT be neutralized' },
        },
      },
    },
  },
}

const audits = await parallel(FILES.map(f => () =>
  agent(
    `${CONTEXT}\n\nAudit ONLY this file for O-side bias: ${f.path}\n` +
    `Focus hint: ${f.note}\n` +
    `Read the file (for the .py file read only the named functions). Quote each biased phrase EXACTLY, ` +
    `explain why it leans O-side, and give a neutral rewrite. Mark legitimately_o_side=true for phrases ` +
    `that are inherently about an interactive page action and should stay. Be precise; do not invent bias ` +
    `where the wording already names both modalities.`,
    { label: `audit:${f.key}`, phase: 'Audit', schema: FINDING_SCHEMA }
  ).then(r => ({ ...r, key: f.key }))
))

phase('Synthesize')
const synth = await agent(
  `${CONTEXT}\n\nPer-file audit findings:\n${JSON.stringify(audits.filter(Boolean))}\n\n` +
  `Synthesize a single prioritized fix list. For each REAL bias (exclude legitimately_o_side ones), give: ` +
  `file, exact old phrase, exact new phrase, and a one-line rationale. Also give an overall verdict on how ` +
  `O-biased the current wording is, and call out the SINGLE most important fix. Keep rewrites minimal and ` +
  `surgical — change only what creates the bias, preserve everything else verbatim.`,
  { label: 'synthesize:fixlist', phase: 'Synthesize', schema: {
    type: 'object',
    required: ['overall_verdict', 'most_important_fix', 'fixes'],
    properties: {
      overall_verdict: { type: 'string' },
      most_important_fix: { type: 'string' },
      fixes: {
        type: 'array',
        items: {
          type: 'object',
          required: ['file', 'old', 'new', 'rationale'],
          properties: {
            file: { type: 'string' },
            old: { type: 'string' },
            new: { type: 'string' },
            rationale: { type: 'string' },
          },
        },
      },
    },
  } }
)

return { audits: audits.filter(Boolean), synth }
