"""Multimodal Stage 2 rewriter for MolmoWeb hybrid skeleton.

Takes per-step input + the BEFORE-action screenshot, produces clean JSON fields
(this_turn / cumulative_progress / remaining_subgoals / next_action_rationale /
action_description / step_quality / step_quality_reason) for SFT.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


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

### INITIAL step (step 1, first turn — task decomposition, no observation yet):
{
  "task_analysis":      "<Analyze the task in first person: (1) identify all concrete sub-goals — facts to find, values to collect, actions to perform; (2) determine which to tackle first and why. Pure task decomposition — do NOT describe any actions taken or results found.>",
  "remaining_subgoals": [<ALL atomic sub-goals needed to finish the task — JSON array of strings>],
  "action_description": <5-15 words: short description of the first tool call>
}

### NON-FINAL step (step 2+, tool_call is set):
{
  "this_turn":             "<one paragraph: ground in what the screenshot shows (page state, visible elements, what UI region the action targets). End with the expected outcome of this action. Be specific — record names, dates, numbers, relationships, URLs, quotes. Do NOT summarize at a high level.>",
  "cumulative_progress":   "<RUNNING summary of EVERYTHING accomplished so far that is relevant to the task. ONLY write progress that has actually been made (facts confirmed, values collected, URLs identified). DO NOT write what is still missing, plans, or guesses — those belong in remaining_subgoals. Be COMPREHENSIVE — capture every task-relevant concrete fact. Build it by taking the prior cumulative_progress and walking through it POINT BY POINT: KEEP items still true; MODIFY/CORRECT items now refined (state both old and new); ADD new items confirmed in this_turn; DELETE items proven wrong. This is the authoritative record of what was accomplished and what the final answer will be composed from.>",
  "remaining_subgoals":    [<atomic open items needed to finish the task; JSON array of strings>],
  "next_action_rationale": "<one sentence: WHY THIS specific tool_call (with these coordinates / text / direction) closes one of the remaining_subgoals.>",
  "action_description":    "<5-15 words: short human-readable description of the UI action, e.g. 'Click the Sign In button at top-right', 'Fill search box with Julia language stars', 'Drag the price slider down to $500'. Goes into compact action history.>",
  "step_quality":          "<good | mediocre | poor | fatal>",
  "step_quality_reason":   "<≤ 25 words: short sentence citing specific visual cues or action targeting.>"
}

### FINAL step (final_answer is set, tool_call is null):
{
  "this_turn":            "<observed result + final state summary>",
  "cumulative_progress":  "<COMPREHENSIVE final knowledge base. Must contain EVERY fact needed to compose the answer. This is the sole source for the answer field.>",
  "remaining_subgoals":   [<open blockers, or []>],
  "answer_plan":          "<one or two sentences: how the answer is composed from cumulative_progress>",
  "answer":               "<the final answer. Composed EXCLUSIVELY from facts in cumulative_progress. Do NOT introduce any fact absent from cumulative_progress.>",
  "step_quality":         "<good | mediocre | poor | fatal>",
  "step_quality_reason":  "<≤ 25 words>"
}

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
- action_description is required on non-final steps, omitted on final.
- answer_plan and answer are required ONLY on the final step.
- Ground all visual claims in what the screenshot actually shows. Do NOT invent UI elements.
- Do NOT contradict the tool_call. Treat its coordinates / text / direction as correct by construction; your job is to articulate WHY they target the right element.
- Do NOT invent facts not in source_thought + screenshot.
- cumulative_progress is the SOLE source for the answer. Every task-relevant detail from each observation must be recorded here.
- CRITICAL: Do NOT leak future information. Every field must ONLY reference what has been observed UP TO AND INCLUDING this step. Never mention, hint at, or foreshadow what happens in later steps, what will be found, or what the eventual answer will be. Write as if you have no knowledge beyond the current step."""


USER_TEMPLATE_INITIAL = """## Task
{task}

## Trajectory context
- initial_url: {initial_url}
- step 1 of {total_steps}  (INITIAL — first turn, task decomposition)

## Tool call the agent emitted at this step
{tool_call_json}

## Screenshot of the initial page: (attached as image)

## Your job
Produce the INITIAL-step JSON described in the system prompt. Analyze the task and decompose it into sub-goals. Do NOT describe any actions taken or results — this is pure task analysis BEFORE any observation.
"""


USER_TEMPLATE_NONFINAL = """## Task
{task}

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

## Screenshot the agent saw BEFORE this action: (attached as image)

## Your job
Produce the NON-FINAL-step JSON described in the system prompt. Use the Recent Actions list to detect loops, redundancy, or wasted backtracking when judging step_quality.
"""


USER_TEMPLATE_FINAL = """## Task
{task}

## Trajectory context
- initial_url: {initial_url}
- step {step_idx} of {total_steps}  (answer synthesis — if task is completed)

## All Actions taken during this trajectory
{recent_actions}

## Cumulative progress from the last action step
{prior_cumulative_progress}

## Remaining subgoals from the last action step
{prior_remaining_subgoals}

## Your job
The trajectory above has NO explicit final answer from the human annotator. You must decide:

1. **If the task appears COMPLETED** (all subgoals resolved, remaining_subgoals is empty or trivial): produce the FINAL-step JSON with a synthesized answer summarizing what was accomplished.

2. **If the task appears INCOMPLETE** (significant remaining_subgoals unresolved, the agent was still mid-task): output EXACTLY this JSON:
{{"task_completed": false, "reason": "<one sentence explaining why the task is not done>"}}

When in doubt (e.g. the last action navigated to a target page but didn't extract specific info), lean toward generating an answer that describes what was achieved so far.
"""


@dataclass
class MMRewriterConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 90.0
    max_tokens: int = 1500
    temperature: float = 0.0
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

        is_initial = (step_idx == 1)

        if is_initial:
            user_text = USER_TEMPLATE_INITIAL.format(
                task=(task or "")[:4000],
                initial_url=(initial_url or "")[:300],
                total_steps=total_steps,
                tool_call_json=json.dumps(tool_call or {}, ensure_ascii=False, indent=2)[:2000],
            )
        elif is_final:
            user_text = USER_TEMPLATE_FINAL.format(
                task=(task or "")[:4000],
                initial_url=(initial_url or "")[:300],
                step_idx=step_idx,
                total_steps=total_steps,
                current_url=(current_url or initial_url or "")[:300],
                recent_actions=recent_actions_text[:3000],
                prior_cumulative_progress=(prior_cumulative_progress or "(empty — this is early in the trajectory)")[:6000],
                prior_remaining_subgoals=subgoals_text[:3000],
                source_thought=(source_thought or "(no source thought recorded)")[:4000],
                final_answer=(final_answer or "")[:3000],
            )
        else:
            user_text = USER_TEMPLATE_NONFINAL.format(
                task=(task or "")[:4000],
                initial_url=(initial_url or "")[:300],
                step_idx=step_idx,
                total_steps=total_steps,
                current_url=(current_url or initial_url or "")[:300],
                recent_actions=recent_actions_text[:3000],
                prior_cumulative_progress=(prior_cumulative_progress or "(empty — this is the first step)")[:6000],
                prior_remaining_subgoals=subgoals_text[:3000],
                source_thought=(source_thought or "(no source thought recorded)")[:4000],
                tool_call_json=json.dumps(tool_call or {}, ensure_ascii=False, indent=2)[:2000],
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ]},
        ]

        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
        except Exception as exc:
            return None, {"latency_ms": int((time.time() - t0) * 1000),
                          "error": f"{type(exc).__name__}: {exc}",
                          "raw": ""}
        latency_ms = int((time.time() - t0) * 1000)
        info = {
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
            "completion_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
            "raw": "",
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

    def synthesize_answer(
        self,
        *,
        task: str,
        initial_url: str,
        step_idx: int,
        total_steps: int,
        prior_cumulative_progress: str,
        prior_remaining_subgoals: list[str],
        recent_actions: list[dict] | None = None,
    ) -> tuple[dict | None, dict]:
        """Text-only call to synthesize a final answer step (no screenshot)."""
        recent_actions = recent_actions or []
        if recent_actions:
            lines = []
            for ra in recent_actions[-15:]:
                tc = ra.get("tool_call") or {}
                desc = ra.get("action_description") or ""
                tc_summary = (f"{tc.get('name','?')}({json.dumps(tc.get('arguments',{}), ensure_ascii=False)[:90]})"
                              if tc else "answer")
                lines.append(f"  S{ra.get('step_idx','?')}: {desc}  ·  {tc_summary}")
            recent_actions_text = "\n".join(lines)
        else:
            recent_actions_text = "(none)"

        subgoals_text = json.dumps(prior_remaining_subgoals or [], ensure_ascii=False)
        user_text = USER_TEMPLATE_FINAL.format(
            task=(task or "")[:4000],
            initial_url=(initial_url or "")[:300],
            step_idx=step_idx,
            total_steps=total_steps,
            recent_actions=recent_actions_text[:5000],
            prior_cumulative_progress=(prior_cumulative_progress or "(empty)")[:6000],
            prior_remaining_subgoals=subgoals_text[:3000],
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
        except Exception as exc:
            return None, {"latency_ms": int((time.time() - t0) * 1000),
                          "error": f"{type(exc).__name__}: {exc}", "raw": ""}
        latency_ms = int((time.time() - t0) * 1000)
        info = {
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(resp.usage, "prompt_tokens", None) if resp.usage else None,
            "completion_tokens": getattr(resp.usage, "completion_tokens", None) if resp.usage else None,
            "raw": "",
        }
        try:
            raw = resp.choices[0].message.content or ""
        except Exception as exc:
            info["error"] = f"response_parse: {exc}"
            return None, info
        info["raw"] = raw[:500]
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
    "action_description",
}

NON_FINAL_REQUIRED = {
    "this_turn",
    "cumulative_progress",
    "remaining_subgoals",
    "next_action_rationale",
    "action_description",
    "step_quality",
    "step_quality_reason",
}
FINAL_REQUIRED = {
    "this_turn",
    "cumulative_progress",
    "remaining_subgoals",
    "answer_plan",
    "answer",
    "step_quality",
    "step_quality_reason",
}
VALID_QUALITY = {"good", "mediocre", "poor", "fatal"}


def _coerce_string(obj: Any) -> Any:
    """Auto-join list-of-strings into a single bullet string (some models return lists)."""
    if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
        return "\n".join(f"- {x}" for x in obj)
    return obj


def validate_rewrite(obj: dict, is_final: bool, is_initial: bool = False) -> str | None:
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
        return None
    for k in ("this_turn", "cumulative_progress"):
        if k in obj:
            obj[k] = _coerce_string(obj[k])
    if not isinstance(obj.get("remaining_subgoals"), list):
        return "remaining_subgoals must be a JSON array"
    if obj.get("step_quality") not in VALID_QUALITY:
        return f"step_quality must be one of {sorted(VALID_QUALITY)}"
    for k in ("this_turn", "cumulative_progress", "step_quality_reason"):
        v = obj.get(k)
        if not isinstance(v, str) or not v.strip():
            return f"{k} must be non-empty string"
    if is_final:
        for k in ("answer_plan", "answer"):
            v = obj.get(k)
            if not isinstance(v, str) or not v.strip():
                return f"{k} must be non-empty string"
    else:
        for k in ("next_action_rationale", "action_description"):
            v = obj.get(k)
            if not isinstance(v, str) or not v.strip():
                return f"{k} must be non-empty string"
    return None
