"""Multimodal Stage 2 rewriter for MolmoWeb hybrid skeleton.

Takes per-step input + the BEFORE-action screenshot, produces clean JSON fields
(this_turn / cumulative_progress / remaining_subgoals / next_action_rationale /
action_description / step_quality / step_quality_reason) for SFT.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from hashlib import sha256
from typing import Any

from openai import OpenAI


CURRENT_RUN_DATE = "2026-05-01"
INFERRED_DATE_START = date(2026, 1, 1)
INFERRED_DATE_DAYS = 120  # 2026-01-01 through 2026-04-30, before CURRENT_RUN_DATE.


def make_time_insensitive_inferred_date(seed: str) -> str:
    """Pick a stable per-trajectory 2026 date before CURRENT_RUN_DATE."""
    digest = sha256(seed.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big") % INFERRED_DATE_DAYS
    return (INFERRED_DATE_START + timedelta(days=offset)).isoformat()


SYSTEM_PROMPT = """You are rewriting a recorded BROWSER-AGENT trajectory into a clean, fielded JSON record so it can be used as supervised-fine-tuning data for a different browser agent that operates in pixel-coordinate mode (hybrid). Write ALL text fields in FIRST PERSON ("I searched for...", "I found that...", "I need to...") as if you are the agent narrating your own reasoning.

The hybrid agent has these tools (you will see one of them per step):
- click({x, y})           : click at integer pixel coordinates
- hover({x, y})           : hover to reveal menus/tooltips
- fill({x, y, text, submit?})  : focus an input at (x, y), type text, optional Enter
- scroll({direction: "up"|"down", x?, y?}) : page or region scroll
- drag({from_x, from_y, to_x, to_y})       : press, hold, drag to release point
- press_key({keys: [...]})                 : keyboard combination (last item = trigger, rest = modifiers)
- goto({url}) / new_tab({url}) / go_back / switch_tab / close_tab / wait

The trajectory was already executed by a human. Your job is NOT to plan the next move — it is to FAITHFULLY DESCRIBE and JUSTIFY what happened at this step in structured form, anchored to the screenshot.

Inputs per step:
- task                 : the user's original instruction
- initial_url          : URL the agent landed on at trajectory start
- step_idx/total_steps : position in trajectory
- prior_cumulative_progress / prior_remaining_subgoals : state carried in from the previous step
- source_thought       : the human annotator's free-form reasoning at this step (raw material — may be verbose, rough, occasionally wrong; use as input not gospel)
- tool_call            : the canonical hybrid tool call the agent ACTUALLY emitted (treat as ground truth; do NOT contradict it)
- screenshot           : the page state the agent saw IMMEDIATELY BEFORE the action (image input)
- final_answer         : present only on the final step (no tool_call)

There are THREE step types with different schemas:

### INITIAL step (step 1, first turn — task decomposition; the first tool_call IS provided to you):
{
  "task_analysis":         "<Analyze the task in first person: (1) identify all concrete sub-goals — facts to find, values to collect, actions to perform; (2) determine which to tackle first and why. Pure task decomposition — do NOT describe any actions taken or results found.>",
  "remaining_subgoals":    [<ALL atomic sub-goals needed to finish the task — JSON array of strings>],
  "next_action_rationale": "<one sentence: WHY this first tool_call (with these coordinates / text / direction) advances a remaining_subgoal. Must be consistent with the provided tool_call and the screenshot.>",
  "action_description":    "<5-15 words: short human-readable description of THIS first UI action (e.g. 'Click the search box at top-center'). Must describe the provided tool_call.>",
  "date_evidence":         <null OR "YYYY-MM-DD": the RETAINED date evidence after this first step. Use a date ONLY if the screenshot/source observation literally shows a concrete date useful for auditing time-sensitive evidence — e.g. a 'most recent' item dated X, a visible 'last updated' / 'published' date, a 'today is X' value. null if none. Do NOT invent or guess — only dates literally visible.>
}

### NON-FINAL step (step 2+, tool_call is set):
{
  "this_turn":             "<one paragraph: ground in what the screenshot shows (page state, visible elements, what UI region the action targets). End with the expected outcome of this action. Be specific — record names, dates, numbers, relationships, URLs, quotes. Do NOT summarize at a high level.>",
  "cumulative_progress":   "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified). DO NOT write what is still missing, plans, or guesses — those belong in remaining_subgoals. Be COMPREHENSIVE — capture every task-relevant concrete fact. Build it by taking the prior cumulative_progress and walking through it POINT BY POINT: KEEP items still true; MODIFY/CORRECT items now refined (state both old and new); ADD new items confirmed in this_turn; DELETE items proven wrong. This is the authoritative record of what was accomplished and what the final answer will be composed from.>",
  "remaining_subgoals":    [<atomic open items needed to finish the task; JSON array of strings>],
  "next_action_rationale": "<one sentence: WHY THIS specific tool_call (with these coordinates / text / direction) closes one of the remaining_subgoals.>",
  "action_description":    "<5-15 words: short human-readable description of the UI action, e.g. 'Click the Sign In button at top-right', 'Fill search box with Julia language stars', 'Drag the price slider down to $500'. Goes into compact action history.>",
  "date_evidence":         <null OR "YYYY-MM-DD": the RETAINED date evidence after this step. If earlier steps already collected date_evidence and this screenshot/source observation shows no newer/better date evidence, KEEP the previous date_evidence. If this step literally surfaces a concrete date useful for auditing time-sensitive evidence — e.g. a 'most recent' item dated X, a visible 'last updated' / 'published' date, a 'today is X' value — decide whether to keep the previous value or update to the better date. null only if no date evidence has been collected anywhere so far. Do NOT invent or guess; only use dates literally observed in the trajectory.>,
  "step_quality":          "<good | mediocre | poor | fatal>",
  "step_quality_reason":   "<≤ 25 words: short sentence citing specific visual cues or action targeting.>"
}

### FINAL step (final_answer is set; this is a normal step that ends with the answer instead of a tool_call):
{
  "this_turn":             "<observed result + final state>",
  "cumulative_progress":   "<COMPREHENSIVE final knowledge base. Must contain EVERY fact needed to compose the answer. This is the sole source for the answer field.>",
  "remaining_subgoals":    [<open blockers, or []>],
  "next_action_rationale": "<one sentence: WHY the observed state / collected evidence is now sufficient to answer, so the next action is to finish with the answer rather than perform another UI action.>",
  "answer":                "<the final answer. Composed EXCLUSIVELY from facts in cumulative_progress. Do NOT introduce any fact absent from cumulative_progress.>",
  "time_dependent":        <true | false: would the CORRECT answer change depending on the date the task is run? true if the task asks for the 'latest / most recent / current / today's / upcoming' state of something that changes over time (news, prices, schedules, standings, 'who currently holds X'). false if the answer is a fixed historical/static fact.>,
  "time_dependent_reason": "<one sentence: WHY it is or isn't time-dependent.>",
  "date_evidence":         <null OR "YYYY-MM-DD": the RETAINED date evidence available at the final step. Keep the previous date_evidence if no final screenshot/source observation provides a better observed date. null only if no date evidence was collected across the whole trajectory.>,
  "inferred_date":         <"YYYY-MM-DD" | null: if time_dependent is true, infer the run date from the evidence and Current run date; if time_dependent is false, set null.>,
  "step_quality":          "<good | mediocre | poor | fatal>",
  "step_quality_reason":   "<≤ 25 words>"
}
(FINAL step has NO action_description and NO tool_call — it ends the trajectory with the answer.)

Step quality rubric:
  good     — clear progress; action visually targets the right element; no redundancy. Final: answer supported by cumulative_progress.
  mediocre — acceptable but not efficient; slight redundancy or overly broad action. Final: answer correct but evidence gaps glossed over.
  poor     — wasteful; repeats prior action; targets wrong element; chases irrelevant tangent. Final: answer asserted without evidence.
  fatal    — hard visual hallucination; infinite loop; abandons task; answer contradicts cumulative_progress.

Hard rules:
- Output EXACTLY ONE JSON object. No markdown fences. No XML tags. No prose outside the JSON.
- Use the EXACT field names for the correct step type (INITIAL / NON-FINAL / FINAL). No extra fields. No missing fields.
- ALL text fields must be in FIRST PERSON ("I analyzed...", "I found...", "I need to..."). Never use "the agent", "the user", or third person.
- All string fields non-empty.
- cumulative_progress and this_turn MUST be strings, NOT arrays. Use \\n + dash bullets inside the string for structure if needed.
- remaining_subgoals is a JSON array of strings (may be []).
- step_quality MUST be one of the four literal strings: "good", "mediocre", "poor", "fatal".
- action_description is required on non-final, omitted on final.
- Ground all visual claims in what the screenshot actually shows. Do NOT invent UI elements.
- Do NOT contradict the tool_call. Treat its coordinates / text / direction as correct by construction; your job is to articulate WHY they target the right element.
- Do NOT invent facts not in source_thought + screenshot.
- cumulative_progress is the SOLE source for the answer. Every task-relevant detail from each observation must be recorded here.
- In FINAL steps, the original terminal answer text is NOT browser evidence. Never cite it, never say the agent provided an answer, and never add facts from it into this_turn or cumulative_progress unless those facts are already supported by prior cumulative_progress, source_thought, or the screenshot. If the terminal answer contains unsupported facts, keep the unsupported sub-goals open and answer with the supported blocker/partial result instead of importing the terminal answer.
- CRITICAL: Do NOT leak future information. Every field must ONLY reference what has been observed UP TO AND INCLUDING this step. Never mention, hint at, or foreshadow what happens in later steps, what will be found, or what the eventual answer will be. Write as if you have no knowledge beyond the current step.
- `date_evidence` is carried forward across steps. If there is no new date evidence in the current screenshot/source observation, preserve the previous retained date_evidence instead of resetting it to null. If there is new observed date evidence, keep the old value or update it based on which date best supports auditing the time-sensitive answer.
- The Current run date is provided in every user message. Use Current run date as the actual date for deciding current/latest/time-sensitive tasks. Current run date is NOT date_evidence. For final `inferred_date`: when `time_dependent` is true, infer the run date from the retained date_evidence and Current run date; when `time_dependent` is false, set null."""


USER_TEMPLATE_INITIAL = """## Task
{task}

## Current run date
{current_run_date}

## Trajectory context
- initial_url: {initial_url}
- step 1 of {total_steps}  (INITIAL — first turn, task decomposition)

## Current Browser State
- URL: {current_url}

## Source reasoning at this step (verbatim from the human annotator)
{source_thought}

## Tool call the agent emitted at this step
{tool_call_json}

## Screenshot of the initial page: (attached as image)

## Your job
Produce the INITIAL-step JSON described in the system prompt. Analyze the task and decompose it into sub-goals (task_analysis + remaining_subgoals — pure decomposition, no results yet). Then write next_action_rationale and action_description for the FIRST tool call shown above (grounded in the screenshot): next_action_rationale states WHY this call advances a sub-goal, and action_description briefly describes the UI action. Both MUST be consistent with the provided tool_call. For `date_evidence`: set it to a YYYY-MM-DD ONLY if THIS first screenshot/source observation literally shows a concrete date useful for auditing time-sensitive evidence; otherwise null. Do NOT describe any observation or result in task_analysis.
"""


USER_TEMPLATE_NONFINAL = """## Task
{task}

## Current run date
{current_run_date}

## Trajectory context
- initial_url: {initial_url}
- step {step_idx} of {total_steps}  (non-final)

## Current Browser State
- URL: {current_url}

## Recent Actions (up to 10 most recent tool calls preceding this step)
{recent_actions}

## Prior cumulative_progress (from previous step)
{prior_cumulative_progress}

## Prior remaining_subgoals (from previous step)
{prior_remaining_subgoals}

## Source reasoning at this step (verbatim from the human annotator)
{source_thought}

## Tool call the agent emitted at this step
{tool_call_json}

## Retained date evidence from earlier steps (audit trail for time-sensitive evidence; not the run date)
{prior_date_evidence}

## Screenshot the agent saw BEFORE this action: (attached as image)

## Your job
Produce the NON-FINAL-step JSON described in the system prompt. Use the Recent Actions list to detect loops, redundancy, or wasted backtracking when judging step_quality. For `date_evidence`: output the retained date evidence after this step. If the current screenshot/source observation shows no new useful date, keep the previous retained date_evidence. If it literally shows a new useful date, decide whether to keep the old value or update to the new one. Never invent a date.
"""


USER_TEMPLATE_FINAL = """## Task
{task}

## Current run date
{current_run_date}

## Trajectory context
- initial_url: {initial_url}
- step {step_idx} of {total_steps}  (FINAL)

## Current Browser State
- URL: {current_url}

## Recent Actions (up to 10 most recent tool calls preceding this final step)
{recent_actions}

## Prior cumulative_progress (from previous step)
{prior_cumulative_progress}

## Prior remaining_subgoals (from previous step)
{prior_remaining_subgoals}

## Source reasoning at this step (verbatim from the human annotator)
{source_thought}

## Original terminal answer text (label only; NOT browser evidence)
{final_answer}

## Retained date evidence collected across the trajectory (audit trail for time-sensitive evidence; not the run date)
{prior_date_evidence}

## Screenshot the agent saw BEFORE giving the answer: (attached as image)

## Your job
Produce the FINAL-step JSON described in the system prompt. Verify whether the original terminal answer text is supported by what the trajectory actually accomplished (use Recent Actions + prior cumulative_progress + source_thought + screenshot). The original terminal answer text is not an observation and not evidence: do not cite it, do not mention that it was provided, and do not copy unsupported facts from it into this_turn, cumulative_progress, next_action_rationale, or answer. If it contains facts that are not supported by the observed trajectory, keep the corresponding sub-goals unresolved and make `answer` a supported blocker/partial answer instead. Also judge `time_dependent`: would the correct answer change depending on what date the task is run? For `date_evidence`: carry forward the retained date evidence unless the final screenshot/source observation provides a better observed date. For `inferred_date`: if time_dependent is true, infer the run date from the retained date_evidence and Current run date; if time_dependent is false, set it to null.
"""


@dataclass
class MMRewriterConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 90.0
    max_tokens: int = 1500
    temperature: float = 0.0
    retry_temperature: float = 0.7
    image_max_w: int = 1280     # always resized to this in the wrapper
    image_max_h: int = 720
    jpeg_quality: int = 80


class MMStepRewriter:
    """Multimodal step rewriter. Always sends a screenshot as the second user-content item."""

    def __init__(self, cfg: MMRewriterConfig):
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.api_key or "EMPTY",
            base_url=cfg.base_url,
            timeout=cfg.timeout,
        )

    def _encode_image(self, image_bytes: bytes) -> str:
        from PIL import Image
        try:
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as exc:
            raise ValueError(f"image decode failed: {exc}") from exc
        img = img.convert("RGB")
        tw, th = self.cfg.image_max_w, self.cfg.image_max_h
        if (img.width, img.height) != (tw, th):
            img = img.resize((tw, th), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.cfg.jpeg_quality, optimize=True)
        return base64.b64encode(buf.getvalue()).decode()

    def rewrite_step(
        self,
        *,
        task: str,
        initial_url: str,
        step_idx: int,
        total_steps: int,
        is_final: bool,
        prior_cumulative_progress: str,
        prior_remaining_subgoals: list[str],
        source_thought: str,
        tool_call: dict | None,
        final_answer: str | None,
        screenshot_bytes: bytes,
        current_url: str = "",
        recent_actions: list[dict] | None = None,
        prior_date_evidence: list[str] | None = None,
        temperature: float | None = None,
    ) -> tuple[dict | None, dict]:
        """Returns (parsed_rewrite | None, info_dict).
        info_dict always has: latency_ms, prompt_tokens, completion_tokens, error (if any), raw."""
        # Encode image
        try:
            img_b64 = self._encode_image(screenshot_bytes)
        except Exception as exc:
            return None, {"latency_ms": 0, "error": f"image_encode: {exc}", "raw": ""}

        # Build user text
        subgoals_text = json.dumps(prior_remaining_subgoals or [], ensure_ascii=False)
        # Format recent actions list: each entry shows "Sn: <action_description>  tool_call_summary"
        recent_actions = recent_actions or []
        if recent_actions:
            lines = []
            for ra in recent_actions[-10:]:
                tc = ra.get("tool_call") or {}
                desc = ra.get("action_description") or ""
                tc_summary = (f"{tc.get('name','?')}({json.dumps(tc.get('arguments',{}), ensure_ascii=False)[:90]})"
                              if tc else "answer")
                lines.append(f"  S{ra.get('step_idx','?')}: {desc}  ·  {tc_summary}")
            recent_actions_text = "\n".join(lines)
        else:
            recent_actions_text = "(none — this is the first step)"

        is_initial = (step_idx == 1 and not is_final)

        de_list = [d for d in (prior_date_evidence or []) if d]
        date_evidence_text = (
            "\n".join(f"  - retained evidence: {d}" for d in de_list)
            if de_list else "(none collected yet)"
        )

        if is_initial:
            user_text = USER_TEMPLATE_INITIAL.format(
                task=(task or "")[:4000],
                current_run_date=CURRENT_RUN_DATE,
                initial_url=(initial_url or "")[:300],
                total_steps=total_steps,
                current_url=(current_url or initial_url or "")[:300],
                source_thought=(source_thought or "(no source thought recorded)")[:4000],
                tool_call_json=json.dumps(tool_call or {}, ensure_ascii=False, indent=2)[:2000],
            )
        elif is_final:
            user_text = USER_TEMPLATE_FINAL.format(
                task=(task or "")[:4000],
                current_run_date=CURRENT_RUN_DATE,
                initial_url=(initial_url or "")[:300],
                step_idx=step_idx,
                total_steps=total_steps,
                current_url=(current_url or initial_url or "")[:300],
                recent_actions=recent_actions_text[:3000],
                prior_cumulative_progress=(prior_cumulative_progress or "(empty — this is early in the trajectory)")[:6000],
                prior_remaining_subgoals=subgoals_text[:3000],
                source_thought=(source_thought or "(no source thought recorded)")[:4000],
                final_answer=(final_answer or "")[:3000],
                prior_date_evidence=date_evidence_text,
            )
        else:
            user_text = USER_TEMPLATE_NONFINAL.format(
                task=(task or "")[:4000],
                current_run_date=CURRENT_RUN_DATE,
                initial_url=(initial_url or "")[:300],
                step_idx=step_idx,
                total_steps=total_steps,
                current_url=(current_url or initial_url or "")[:300],
                recent_actions=recent_actions_text[:3000],
                prior_cumulative_progress=(prior_cumulative_progress or "(empty — this is the first step)")[:6000],
                prior_remaining_subgoals=subgoals_text[:3000],
                source_thought=(source_thought or "(no source thought recorded)")[:4000],
                tool_call_json=json.dumps(tool_call or {}, ensure_ascii=False, indent=2)[:2000],
                prior_date_evidence=date_evidence_text,
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]},
        ]

        request_temperature = self.cfg.temperature if temperature is None else temperature
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=request_temperature,
                max_tokens=self.cfg.max_tokens,
            )
        except Exception as exc:
            return None, {"latency_ms": int((time.time() - t0) * 1000),
                          "error": f"{type(exc).__name__}: {exc}",
                          "raw": "",
                          "temperature": request_temperature}
        latency_ms = int((time.time() - t0) * 1000)
        info = {
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
            "completion_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
            "raw": "",
            "temperature": request_temperature,
        }
        try:
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            info["error"] = f"response_parse: {exc}"
            return None, info
        info["raw"] = raw

        parsed = self._parse_json(raw)
        if parsed is None:
            info["error"] = "json_parse_failed"
            return None, info
        return parsed, info

    def _parse_json(self, raw: str) -> dict | None:
        s = (raw or "").strip()
        # Strip code fences if present
        if s.startswith("```"):
            s = s.split("```", 2)[1] if s.count("```") >= 2 else s.lstrip("`")
            if s.startswith("json"):
                s = s[4:]
            s = s.strip()
            if s.endswith("```"):
                s = s[:-3].strip()
        # Find first '{' to last '}'
        i = s.find("{")
        j = s.rfind("}")
        if i < 0 or j < 0 or j < i:
            return None
        try:
            return json.loads(s[i:j+1])
        except Exception:
            return None


INITIAL_REQUIRED = {
    "task_analysis",
    "remaining_subgoals",
    "next_action_rationale",
    "action_description",
    "date_evidence",
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

FORBIDDEN_FINAL_ANSWER_LEAK_PATTERNS = [
    r"\bfinal answer\b",
    r"\bterminal answer\b",
    r"\boriginal answer\b",
    r"\braw answer\b",
    r"\bemitted answer\b",
    r"\banswer provided\b",
    r"\bprovided (?:by|from) the agent\b",
    r"\bagent (?:has )?provided\b",
    r"\bfrom (?:the )?answer\b",
    r"\[ANSWER\]",
    r"\bsubsequent view\b",
]


def _coerce_string(obj: Any) -> Any:
    """Auto-join list-of-strings into a single bullet string (some models return lists)."""
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return "\n".join(f"- {x}" for x in obj)
    return obj


def _contains_forbidden_final_answer_leak(text: str) -> bool:
    return any(
        re.search(pattern, text or "", flags=re.IGNORECASE)
        for pattern in FORBIDDEN_FINAL_ANSWER_LEAK_PATTERNS
    )


def validate_rewrite(
    obj: dict,
    is_final: bool,
    is_initial: bool = False,
    time_insensitive_inferred_date: str | None = None,
) -> str | None:
    if not isinstance(obj, dict):
        return "output is not a dict"
    if is_initial:
        required = INITIAL_REQUIRED
    elif is_final:
        required = FINAL_REQUIRED
    else:
        required = NON_FINAL_REQUIRED
    missing = sorted(required - set(obj.keys()))
    if missing:
        return f"missing required keys: {missing}"
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
        de = obj.get("date_evidence", None)
        if de is not None and not (isinstance(de, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", de)):
            return "date_evidence must be null or YYYY-MM-DD"
        return None
    for k in ("this_turn", "cumulative_progress"):
        if k in obj:
            obj[k] = _coerce_string(obj[k])
    if not isinstance(obj.get("remaining_subgoals"), list):
        return "remaining_subgoals must be a JSON array"
    de = obj.get("date_evidence", None)
    if de is not None and not (isinstance(de, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", de)):
        return "date_evidence must be null or YYYY-MM-DD"
    if obj.get("step_quality") not in VALID_QUALITY:
        return f"step_quality must be one of {sorted(VALID_QUALITY)}"
    for k in ("this_turn", "cumulative_progress", "step_quality_reason"):
        v = obj.get(k)
        if not isinstance(v, str) or not v.strip():
            return f"{k} must be non-empty string"
    if is_final:
        # Final step is an answer step: it must NOT carry action_description
        # (the model sometimes emits it out of habit). Drop it silently.
        obj.pop("action_description", None)
        for k in ("next_action_rationale", "answer", "time_dependent_reason"):
            v = obj.get(k)
            if not isinstance(v, str) or not v.strip():
                return f"{k} must be non-empty string"
        for k in ("this_turn", "cumulative_progress", "next_action_rationale", "step_quality_reason"):
            if _contains_forbidden_final_answer_leak(obj.get(k, "")):
                return f"{k} leaks original terminal answer provenance"
        if not isinstance(obj.get("time_dependent"), bool):
            return "time_dependent must be a boolean"
        idate = obj.get("inferred_date", None)
        if idate is not None and not (isinstance(idate, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", idate)):
            return "inferred_date must be null or YYYY-MM-DD"
        if obj.get("time_dependent"):
            if idate is None:
                return "inferred_date must be a YYYY-MM-DD string when time_dependent is true"
        else:
            obj["inferred_date"] = (
                time_insensitive_inferred_date
                or make_time_insensitive_inferred_date("default")
            )
    else:
        for k in ("next_action_rationale", "action_description"):
            v = obj.get(k)
            if not isinstance(v, str) or not v.strip():
                return f"{k} must be non-empty string"
    return None
