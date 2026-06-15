"""Stage 2 rewriter: turn skeleton steps into fully-filled hybrid search-mode samples.

Per-trajectory sequential (cumulative_progress carries forward); parallel across
trajectories via a ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from openai import OpenAI


# -----------------------------------------------------------------------------
# Prompt
# -----------------------------------------------------------------------------


CURRENT_RUN_DATE = "2026-05-01"
INFERRED_DATE_START = date(2026, 1, 1)
INFERRED_DATE_DAYS = 120  # 2026-01-01 through 2026-04-30, before CURRENT_RUN_DATE.


def make_time_insensitive_inferred_date(seed: str) -> str:
    """Pick a stable per-trajectory 2026 date before CURRENT_RUN_DATE."""
    digest = sha256(seed.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % INFERRED_DATE_DAYS
    return (INFERRED_DATE_START + timedelta(days=offset)).isoformat()


SYSTEM_PROMPT = """You are rewriting a recorded search-agent trajectory into a clean, fielded JSON record for supervised-fine-tuning. Write ALL text fields in FIRST PERSON as if you are the agent narrating your own reasoning.

You will receive ONE step of a trajectory and must output exactly ONE JSON object. The trajectory was already executed; your job is to faithfully describe what happened at this step.

The agent operates with these tools:
- search({query: array<string>})
- scholar({query: array<string>})
- visit({url, goal, parse_type?})
- code_sandbox({code, language, ...})

There are THREE step types with different schemas:

### INITIAL step (step 1, first turn — task decomposition; the first tool_call IS provided to you):
{
  "task_analysis":         "<Analyze the task: identify all concrete sub-goals — facts to find, values to collect, actions to perform. Determine which sub-goals to tackle first and why. Pure task decomposition — do NOT describe any actions taken or results found.>",
  "remaining_subgoals":    [<ALL atomic sub-goals needed to finish the task>],
  "next_action_rationale": "<one sentence: WHY this first tool_call advances a remaining_subgoal. Must be consistent with the provided tool_call.>",
  "action_description":    "<5-15 words: short description of THIS first tool call. Must describe the provided tool_call.>"
}

### NON-FINAL step (step 2+, tool_call is set):
{
  "this_turn":              "<one paragraph: what THIS turn newly contributed. May be a fact surfaced from the latest tool result, a confirmation, or a failure. Be specific — record names, dates, numbers, relationships, URLs, quotes. Do NOT summarize at a high level.>",
  "cumulative_progress":    "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified). DO NOT write what is still missing, plans, or guesses — those belong in remaining_subgoals. Be COMPREHENSIVE — capture every task-relevant concrete fact. Build it by taking the prior cumulative_progress and walking through it POINT BY POINT: KEEP items still true; MODIFY/CORRECT items now refined (state both old and new); ADD new items confirmed in this_turn; DELETE items proven wrong. This is the authoritative record of what was accomplished and what the final answer will be composed from.>",
  "remaining_subgoals":     [<atomic open items still required>],
  "next_action_rationale":  "<one sentence: WHY this tool_call will close some remaining_subgoal>",
  "action_description":     "<5-15 words: short description of THIS tool call>",
  "date_evidence":          <null OR "YYYY-MM-DD": the RETAINED date evidence after this step. If earlier steps already collected date_evidence and this turn shows no newer/better date evidence, KEEP the previous date_evidence. If this turn literally surfaces a concrete date useful for auditing time-sensitive evidence — e.g. a 'most recent' article/result dated X, a visible 'last updated' / 'published' date, a 'today is X' value — decide whether to keep the previous value or update to the better date. null only if no date evidence has been collected anywhere so far. Do NOT invent or guess; only use dates literally observed in the trajectory.>,
  "step_quality":           "<good | mediocre | poor | fatal>",
  "step_quality_reason":    "<≤ 25 words>"
}

### FINAL step (final_answer is set; this is a normal step that ends with the answer instead of a tool_call):
{
  "this_turn":             "<one paragraph: what the final observation newly shows — a final value confirmation, a final state observation, or a blocker.>",
  "cumulative_progress":   "<COMPREHENSIVE final knowledge base. Must contain EVERY fact needed to compose the answer. This is the sole source for the answer field.>",
  "remaining_subgoals":    [<sub-goals NOT met, or [] if fully solved>],
  "next_action_rationale": "<one sentence: WHY the collected evidence is now sufficient to answer, so the next action is to finish with the answer rather than call another tool.>",
  "answer":                "<the final answer. For information-seeking tasks, give the concrete answer. Composed EXCLUSIVELY from facts in cumulative_progress. Do NOT introduce any fact absent from cumulative_progress.>",
  "time_dependent":        <true | false: would the CORRECT answer change depending on the date the task is run? true if the task asks for the 'latest / most recent / current / today's / upcoming' state of something that changes over time (news, prices, schedules, standings, 'who currently holds X'). false if the answer is a fixed historical/static fact that does not change with the date.>,
  "time_dependent_reason": "<one sentence: WHY it is or isn't time-dependent.>",
  "date_evidence":         <null OR "YYYY-MM-DD": the RETAINED date evidence available at the final step. Keep the previous date_evidence if no final observation provides a better date. null only if no date evidence was collected across the whole trajectory.>,
  "inferred_date":         <"YYYY-MM-DD" | null: if time_dependent is true, infer the run date from the evidence and Current run date; if time_dependent is false, set null.>,
  "step_quality":          "<good | mediocre | poor | fatal>",
  "step_quality_reason":   "<≤ 25 words>"
}
(FINAL step has NO action_description and NO tool_call — it ends the trajectory with the answer.)

Step quality rubric:
  good     — clear progress; tool call well-targeted; no redundancy. Final: answer supported by cumulative_progress.
  mediocre — acceptable but not efficient; slight redundancy or overly broad query. Final: answer correct but evidence gaps glossed over.
  poor     — wasteful; repeats prior call; chases irrelevant tangent; reasoning incoherent. Final: answer asserted without evidence.
  fatal    — hard hallucination; infinite loop; abandons task; answer contradicts cumulative_progress.

Hard rules:
- Output EXACTLY ONE JSON object. No XML tags. No markdown fences. No prose outside the JSON.
- Use the EXACT field names for the correct step type. No extra fields. No missing fields.
- ALL text fields in FIRST PERSON ("I found...", "I need to..."). Never "the agent" or third person.
- Do NOT invent facts not in source_think + prior_observation.
- cumulative_progress is the SOLE source for the answer. Every task-relevant detail from each observation must be recorded here — names, dates, numbers, descriptions, relationships. If a fact matters for the answer, it MUST appear in cumulative_progress.
- CRITICAL: Do NOT leak future information. Write as if you have no knowledge beyond the current step.
- `date_evidence` is carried forward across steps. If there is no new date evidence in the current observation, preserve the previous retained date_evidence instead of resetting it to null. If there is new observed date evidence, keep the old value or update it based on which date best supports auditing the time-sensitive answer.
- The Current run date is provided in every user message. Use Current run date as the actual date for deciding current/latest/time-sensitive tasks. Current run date is NOT date_evidence. For final `inferred_date`: when `time_dependent` is true, infer the run date from the retained date_evidence and Current run date; when `time_dependent` is false, set null.
"""


USER_TEMPLATE_INITIAL = """## Task
{task}

## Current run date
{current_run_date}

## Step
1 of {total_steps}  (INITIAL — first turn, task decomposition)

## Tool call the agent emitted at this first step (already mapped)
{mapped_tool_call_json}

## Your job
Produce the INITIAL-step JSON described in the system prompt. Analyze the task and decompose it into sub-goals (task_analysis + remaining_subgoals — pure decomposition, no results yet). Then write next_action_rationale and action_description for the FIRST tool call shown above: next_action_rationale states WHY this call advances a sub-goal, and action_description briefly describes the call. Both MUST be consistent with the provided tool_call. Do NOT describe any observation or result — none exists yet.
"""


USER_TEMPLATE_NONFINAL = """## Task
{task}

## Current run date
{current_run_date}

## Step
{step_idx} of {total_steps}  (non-final)

## Recent Actions (up to 10 most recent tool calls preceding this step)
{recent_actions}

## Prior cumulative_progress (from previous step)
{prior_cumulative_progress}

## Prior remaining_subgoals (from previous step)
{prior_remaining_subgoals}

## Prior observation (the tool_response right before this step)
{prior_observation}

## Source reasoning at this step (verbatim)
{source_think}

## Tool call the agent emitted at this step (already mapped)
{mapped_tool_call_json}

## Retained date evidence from earlier steps (audit trail for time-sensitive evidence; not the run date)
{prior_date_evidence}

## Your job
Produce the non-final-step JSON described in the system prompt. Use Recent Actions to detect loops, redundancy, or wasted backtracking when judging step_quality. For `date_evidence`: output the retained date evidence after this step. If the current observation shows no new useful date, keep the previous retained date_evidence. If the current observation literally shows a new useful date, decide whether to keep the old value or update to the new one. Never invent a date.
"""


USER_TEMPLATE_FINAL = """## Task
{task}

## Current run date
{current_run_date}

## Step
{step_idx} of {total_steps}  (FINAL — agent emitted <answer>)

## Recent Actions (up to 10 most recent tool calls preceding this final step)
{recent_actions}

## Prior cumulative_progress (from previous step)
{prior_cumulative_progress}

## Prior remaining_subgoals (from previous step)
{prior_remaining_subgoals}

## Prior observation (the tool_response right before this final step)
{prior_observation}

## Source reasoning at this step (verbatim)
{source_think}

## Final answer emitted by the agent
{final_answer}

## Retained date evidence collected across the trajectory (audit trail for time-sensitive evidence; not the run date)
{prior_date_evidence}

## Your job
Produce the FINAL-step JSON described in the system prompt. Verify the final answer is supported by what the trajectory actually accomplished (use Recent Actions + cumulative_progress). Also judge `time_dependent`: would the correct answer change depending on what date the task is run? For `date_evidence`: carry forward the retained date evidence unless the final observation provides a better observed date. For `inferred_date`: if time_dependent is true, infer the run date from the retained date_evidence and Current run date; if time_dependent is false, set it to null.
"""


# -----------------------------------------------------------------------------
# Client / rewriter
# -----------------------------------------------------------------------------


@dataclass
class RewriterConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 60.0
    max_tokens: int = 1500
    temperature: float = 0.0
    retry_temperature: float = 0.7
    obs_truncate_chars: int = 12000
    think_truncate_chars: int = 6000
    task_truncate_chars: int = 4000


class StepRewriter:
    def __init__(self, cfg: RewriterConfig):
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.api_key or "EMPTY",
            base_url=cfg.base_url,
            timeout=cfg.timeout,
        )

    def _truncate(self, s: str, n: int) -> str:
        s = s or ""
        if len(s) <= n:
            return s
        # keep head + tail to preserve context
        head = n // 2
        tail = n - head
        return s[:head] + "\n...[truncated]...\n" + s[-tail:]

    def rewrite_step(
        self,
        *,
        task: str,
        step_idx: int,
        total_steps: int,
        is_final: bool,
        prior_cumulative_progress: str,
        prior_remaining_subgoals: list[str],
        prior_observation: str,
        source_think: str,
        mapped_tool_call: dict | None,
        final_answer: str | None,
        recent_actions: list[dict] | None = None,
        prior_date_evidence: list[str] | None = None,
        temperature: float | None = None,
        time_insensitive_inferred_date: str | None = None,
    ) -> tuple[dict | None, dict]:
        """Returns (parsed_json, debug_info). parsed_json=None on failure."""
        task_c = self._truncate(task, self.cfg.task_truncate_chars)
        obs_c = self._truncate(prior_observation or "(no prior observation)", self.cfg.obs_truncate_chars)
        think_c = self._truncate(source_think or "(no source reasoning recorded)", self.cfg.think_truncate_chars)
        prior_cp = prior_cumulative_progress or "(none — this is step 1)"
        prior_sg = json.dumps(prior_remaining_subgoals or [], ensure_ascii=False)

        recent_actions = recent_actions or []
        if recent_actions:
            lines = []
            for ra in recent_actions[-10:]:
                tc = ra.get("tool_call") or {}
                desc = ra.get("action_description") or ""
                args_str = json.dumps(tc.get("arguments", {}), ensure_ascii=False)[:120] if tc else ""
                tc_summary = f"{tc.get('name','?')}({args_str})" if tc else "answer"
                lines.append(f"  S{ra.get('step_idx','?')}: {desc}  ·  {tc_summary}")
            recent_actions_text = "\n".join(lines)
        else:
            recent_actions_text = "(none — this is the first step)"

        is_initial = (step_idx == 1)

        de_list = [d for d in (prior_date_evidence or []) if d]
        date_evidence_text = (
            "\n".join(f"  - retained evidence: {d}" for d in de_list)
            if de_list else "(none collected yet)"
        )
        arbitrary_date_placeholder = (
            time_insensitive_inferred_date
            or make_time_insensitive_inferred_date(task_c)
        )

        if is_initial:
            user_msg = USER_TEMPLATE_INITIAL.format(
                task=task_c,
                current_run_date=CURRENT_RUN_DATE,
                total_steps=total_steps,
                mapped_tool_call_json=json.dumps(mapped_tool_call or {}, ensure_ascii=False, indent=2),
            )
        elif is_final:
            user_msg = USER_TEMPLATE_FINAL.format(
                task=task_c,
                current_run_date=CURRENT_RUN_DATE,
                step_idx=step_idx,
                total_steps=total_steps,
                recent_actions=recent_actions_text[:3000],
                prior_cumulative_progress=prior_cp,
                prior_remaining_subgoals=prior_sg,
                prior_observation=obs_c,
                source_think=think_c,
                final_answer=final_answer or "(empty)",
                prior_date_evidence=date_evidence_text,
            )
        else:
            user_msg = USER_TEMPLATE_NONFINAL.format(
                task=task_c,
                current_run_date=CURRENT_RUN_DATE,
                step_idx=step_idx,
                total_steps=total_steps,
                recent_actions=recent_actions_text[:3000],
                prior_cumulative_progress=prior_cp,
                prior_remaining_subgoals=prior_sg,
                prior_observation=obs_c,
                source_think=think_c,
                mapped_tool_call_json=json.dumps(mapped_tool_call or {}, ensure_ascii=False, indent=2),
                prior_date_evidence=date_evidence_text,
            )

        request_temperature = self.cfg.temperature if temperature is None else temperature
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.cfg.max_tokens,
                temperature=request_temperature,
            )
        except Exception as exc:
            return None, {
                "latency_ms": int((time.time() - t0) * 1000),
                "error": f"{type(exc).__name__}: {exc}",
                "temperature": request_temperature,
            }

        elapsed_ms = int((time.time() - t0) * 1000)
        raw = (resp.choices[0].message.content or "").strip()
        parsed = self._parse_json(raw)
        info = {
            "latency_ms": elapsed_ms,
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
            "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            "temperature": request_temperature,
        }
        if parsed is None:
            info["error"] = "json_parse_failed"
            info["raw_head"] = raw[:300]
        return parsed, info

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Parse the model's response as a single JSON object."""
        s = text.strip()
        # Strip ```json ... ``` fence if present
        m = re.match(r"^```(?:json)?\s*(\{.*\})\s*```\s*$", s, flags=re.DOTALL)
        if m:
            s = m.group(1)
        try:
            obj = json.loads(s)
        except Exception:
            # Try to find the first {...} block heuristically
            m2 = re.search(r"\{[\s\S]*\}", s)
            if not m2:
                return None
            try:
                obj = json.loads(m2.group(0))
            except Exception:
                return None
        if not isinstance(obj, dict):
            return None
        return obj


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


INITIAL_REQUIRED = {
    "task_analysis",
    "remaining_subgoals",
    "next_action_rationale",
    "action_description",
}

NON_FINAL_REQUIRED = {
    "this_turn",
    "cumulative_progress",
    "remaining_subgoals",
    "next_action_rationale",
    "action_description",
    "date_evidence",
    "step_quality",
    "step_quality_reason",
}

FINAL_REQUIRED = {
    "this_turn",
    "cumulative_progress",
    "remaining_subgoals",
    "next_action_rationale",
    "answer",
    "time_dependent",
    "time_dependent_reason",
    "date_evidence",
    "inferred_date",
    "step_quality",
    "step_quality_reason",
}

VALID_QUALITY = {"good", "mediocre", "poor", "fatal"}


def normalize_date_evidence_carry_forward(obj: dict, prior_date_evidence: list[str] | None) -> None:
    """Carry forward retained date_evidence when the model emits null/missing."""
    prior = [d for d in (prior_date_evidence or []) if isinstance(d, str) and d.strip()]
    prior_latest = prior[-1] if prior else None
    raw = obj.get("date_evidence", None)
    if raw is None:
        obj["date_evidence"] = prior_latest
    elif isinstance(raw, str):
        raw = raw.strip()
        obj["date_evidence"] = raw or prior_latest


def _coerce_string_field(obj: dict, key: str) -> None:
    """If `key` is a list of strings, join with bullet lines into a single
    string. Mutates obj in place. No-op if already a string or missing."""
    v = obj.get(key)
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        obj[key] = "\n".join(f"- {x.strip()}" for x in v if x.strip())


def validate_rewrite(
    obj: dict,
    is_final: bool,
    is_initial: bool = False,
    time_insensitive_inferred_date: str | None = None,
) -> str | None:
    """Returns None if OK, else error string. Coerces list-of-string fields to
    string in place so downstream consumers see a uniform shape."""
    if is_initial:
        required = INITIAL_REQUIRED
    elif is_final:
        required = FINAL_REQUIRED
    else:
        required = NON_FINAL_REQUIRED
    missing = sorted(required - set(obj.keys()))
    if missing:
        return f"missing keys: {missing}"
    if is_initial:
        ta = obj.get("task_analysis")
        if not isinstance(ta, str) or not ta.strip():
            return "`task_analysis` must be non-empty string"
        sg = obj.get("remaining_subgoals")
        if not isinstance(sg, list) or any(not isinstance(x, str) for x in sg):
            return "`remaining_subgoals` must be a list of strings"
        for k in ("next_action_rationale", "action_description"):
            v = obj.get(k)
            if not isinstance(v, str) or not v.strip():
                return f"`{k}` must be non-empty string"
        return None
    de = obj.get("date_evidence", None)
    if de is not None and not (isinstance(de, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", de)):
        return "`date_evidence` must be null or a YYYY-MM-DD string"
    # Accept list-of-strings for the paragraph-style fields, auto-flatten.
    for k in ("this_turn", "cumulative_progress"):
        _coerce_string_field(obj, k)
        v = obj.get(k)
        if not isinstance(v, str) or not v.strip():
            return f"`{k}` must be non-empty string"
    sg = obj.get("remaining_subgoals")
    if not isinstance(sg, list) or any(not isinstance(x, str) for x in sg):
        return "`remaining_subgoals` must be a list of strings"
    if is_final:
        # Final step is an answer step: it must NOT carry action_description
        # (the model sometimes emits it out of habit). Drop it silently.
        obj.pop("action_description", None)
        for k in ("next_action_rationale", "answer", "time_dependent_reason"):
            _coerce_string_field(obj, k)
            v = obj.get(k)
            if not isinstance(v, str) or not v.strip():
                return f"`{k}` must be non-empty string"
        if not isinstance(obj.get("time_dependent"), bool):
            return "`time_dependent` must be a boolean"
        idate = obj.get("inferred_date", None)
        if idate is not None and not (isinstance(idate, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", idate)):
            return "`inferred_date` must be null or a YYYY-MM-DD string"
        if obj.get("time_dependent"):
            if idate is None:
                return "`inferred_date` must be a YYYY-MM-DD string when `time_dependent` is true"
        else:
            obj["inferred_date"] = (
                time_insensitive_inferred_date
                or make_time_insensitive_inferred_date("default")
            )
    else:
        for k in ("next_action_rationale", "action_description"):
            _coerce_string_field(obj, k)
            v = obj.get(k)
            if not isinstance(v, str) or not v.strip():
                return f"`{k}` must be non-empty string"
    # Quality fields apply to both final and non-final.
    q = obj.get("step_quality")
    if not isinstance(q, str) or q.strip().lower() not in VALID_QUALITY:
        return f"`step_quality` must be one of {sorted(VALID_QUALITY)}, got {q!r}"
    obj["step_quality"] = q.strip().lower()  # normalize
    _coerce_string_field(obj, "step_quality_reason")
    qr = obj.get("step_quality_reason")
    if not isinstance(qr, str) or not qr.strip():
        return "`step_quality_reason` must be a non-empty string"
    return None
