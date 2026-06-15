"""Two-stage answer-equivalence judge.

Stage A (no LLM): cheap substring/normalization classification.
Stage B (LLM):    only invoked on the ambiguous bucket if requested.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI


SubstringLabel = Literal["EMPTY", "EXACT", "GT_IN_PRED", "PRED_IN_GT", "MISMATCH"]
LLMVerdict = Literal["yes", "no", "uncertain"]
FinalLabel = Literal["CORRECT", "INCORRECT", "UNCERTAIN", "EMPTY"]


def _norm(s: str) -> str:
    """Lower-case + strip every non-alnum + Unicode NFKC normalize."""
    if not s:
        return ""
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"[\s\W_]+", "", s.lower(), flags=re.UNICODE)


def classify_substring(gt: str, pred: str) -> SubstringLabel:
    g = _norm(gt)
    p = _norm(pred)
    if not g or not p:
        return "EMPTY"
    if g == p:
        return "EXACT"
    if g in p:
        return "GT_IN_PRED"
    if p in g:
        return "PRED_IN_GT"
    return "MISMATCH"


# -----------------------------------------------------------------------------
# LLM judge
# -----------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = """You evaluate whether a model's answer correctly answers the user's task, given the ground-truth answer as reference.

You are NOT grading style or completeness — only whether the model's answer correctly identifies the same entity, value, concept, fact, or result as the ground truth in the context of the user's task.

Use the task to disambiguate when the GT and model answer look different on the surface but refer to the same target the task is asking about. The task tells you what level of specificity matters, which axis is being asked about, and which entity is the intended subject.

Two answers are equivalent if they identify the same thing the task is asking for, even when they differ in:
- spelling variants of the same name (e.g. "Georges de Choziba" vs "George of Choziba")
- translations (e.g. "ISO9001:2000 质量体系认证" vs "ISO 9001:2000 Quality Management System Certification")
- abbreviations / expansions (e.g. "U.S." vs "United States")
- minor formatting (dates, units, capitalization)
- one answer is more specific than the other while both clearly satisfy the task (e.g. task asks "which tracking stations", GT says "South African space tracking stations", model lists the two specific stations Hartebeesthoek + Pretoria — both correctly answer the task)
- verbose elaboration that contains the ground-truth fact as the primary commitment
- different units of the same magnitude (e.g. "1 km" vs "1000 m")

They are NOT equivalent when:
- the model picks a closely related but distinct entity (predecessor band, wrong dynasty member, different film)
- the model only describes the topic without committing to the specific answer the task asks for
- the model hedges with multiple guesses and the correct one is not the primary commitment
- the model answers a different question than what the task asks (e.g. task asks for a year, model gives a person)

Output strictly ONE token from {yes, no, uncertain}, no other text.
- yes:       confident the model's answer correctly satisfies the task at the same target as the ground truth
- no:        confident the model's answer is wrong for the task
- uncertain: even with the task context, cannot decide (gt or pred is genuinely ambiguous)
"""


JUDGE_USER_TEMPLATE = """Task: {task}

Ground truth answer: {gt}

Model's final answer: {pred}

Does the model's final answer correctly answer the task, in the same way as the ground truth? Reply with yes / no / uncertain."""


@dataclass
class LLMJudgeConfig:
    base_url: str
    model: str
    api_key: str = "EMPTY"
    timeout: float = 30.0
    max_tokens: int = 8
    temperature: float = 0.0
    pred_truncate_chars: int = 4000
    gt_truncate_chars: int = 1500
    task_truncate_chars: int = 2000


class LLMJudge:
    def __init__(self, cfg: LLMJudgeConfig):
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.api_key or "EMPTY",
            base_url=cfg.base_url,
            timeout=cfg.timeout,
        )

    def judge(self, gt: str, pred: str, task: str = "") -> tuple[LLMVerdict, dict]:
        """Returns (verdict, debug_info). On failure returns ('uncertain', {...error}).

        Truncates long pred / GT / task to keep token cost down.
        """
        gt_c = (gt or "")[: self.cfg.gt_truncate_chars]
        pred_c = (pred or "")[: self.cfg.pred_truncate_chars]
        task_c = (task or "(no task provided)")[: self.cfg.task_truncate_chars]
        t0 = time.time()
        try:
            resp = self.client.chat.completions.create(
                model=self.cfg.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": JUDGE_USER_TEMPLATE.format(task=task_c, gt=gt_c, pred=pred_c),
                    },
                ],
                max_tokens=self.cfg.max_tokens,
                temperature=self.cfg.temperature,
            )
            elapsed_ms = int((time.time() - t0) * 1000)
            raw = (resp.choices[0].message.content or "").strip().lower()
            verdict = self._parse_verdict(raw)
            return verdict, {
                "latency_ms": elapsed_ms,
                "raw": raw,
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
            }
        except Exception as exc:
            elapsed_ms = int((time.time() - t0) * 1000)
            return "uncertain", {
                "latency_ms": elapsed_ms,
                "error": f"{type(exc).__name__}: {exc}",
            }

    @staticmethod
    def _parse_verdict(text: str) -> LLMVerdict:
        s = text.strip().strip("'\"`.,!").lower()
        # Some models prepend with "Answer: " or wrap with quotes.
        first_word = re.split(r"[\s,/]", s, maxsplit=1)[0]
        if first_word in ("yes", "y", "true", "equivalent"):
            return "yes"
        if first_word in ("no", "n", "false", "wrong"):
            return "no"
        return "uncertain"


# -----------------------------------------------------------------------------
# Combined two-stage judgment
# -----------------------------------------------------------------------------


def combine_to_final_label(
    sub: SubstringLabel,
    llm_verdict: LLMVerdict | None,
) -> FinalLabel:
    if sub == "EMPTY":
        return "EMPTY"
    if sub in ("EXACT", "GT_IN_PRED", "PRED_IN_GT"):
        return "CORRECT"
    # sub == "MISMATCH" → fall back to LLM verdict if available
    if llm_verdict == "yes":
        return "CORRECT"
    if llm_verdict == "no":
        return "INCORRECT"
    if llm_verdict == "uncertain":
        return "UNCERTAIN"
    # no LLM verdict at all → leave as UNCERTAIN
    return "UNCERTAIN"
