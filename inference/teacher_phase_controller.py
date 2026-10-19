"""Private phase control for teacher rollouts.

The controller owns the full synthetic solution contract.  The decision model
only receives a compiled view of the current objective, verified public
bindings, and sanitized controller feedback.  Raw verifier decisions and future
phases never enter the decision-model prompt.
"""

from __future__ import annotations

import copy
import os
import json
import re
from dataclasses import dataclass
from typing import Any


def _mode(value: Any, default: str = "S") -> str:
    value = str(value or "").strip().upper()
    return value if value in {"S", "O"} else default


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def phase_termination_reason(
    trigger: str,
    verdict: dict[str, Any],
    *,
    captcha_seen: bool = False,
) -> str:
    """Return a hard-stop reason, leaving recoverable blocks to feedback."""

    block_kind = _clean_text(verdict.get("block_kind")).lower()
    verifier_captcha = bool(verdict.get("blocked")) and block_kind == "captcha"
    if captcha_seen or verifier_captcha:
        return "captcha"

    # 验证器判定的不可恢复局面。之前这两种都只能烧满阶段预算才终止:实测 216 条
    # F3/F4 的 phase_budget_exceeded **全部**卡在第一个组件、零 verified binding,
    # 中位 82 步却什么也没产出。验证器其实早就看出来了(有条轨迹到 step107 才说
    # "这些属性从未同时出现"),只是没有让它据此停下来的通道。这里补上。
    # 两种分开记:task_infeasible 是任务合成的欠账,agent_stuck 是模型能力问题,
    # 混成一个标签的话,后面谁都不知道该去修哪一头。
    unrec = verdict.get("unrecoverable")
    if isinstance(unrec, dict):
        kind = _clean_text(unrec.get("kind")).lower()
        # 要求给出理由 —— 没有证据的"不可行"就是提前放弃,比烧满预算更糟
        if kind in {"task_infeasible", "agent_stuck"} and _clean_text(unrec.get("reason")):
            return kind

    if trigger == "phase_budget" and not verdict.get("phase_complete"):
        return "phase_budget_exceeded"
    return ""


@dataclass(frozen=True)
class ControlDecision:
    allow: bool
    advance: bool = False
    final: bool = False
    feedback: str = ""
    reason: str = ""
    terminate: bool = False


class TeacherPhaseController:
    """State machine over the interleaved phase/switch teacher contract."""

    def __init__(self, contract: dict[str, Any], start_mode: str):
        if not isinstance(contract, dict):
            raise ValueError("private_teacher_steps must be an object")
        raw_steps = contract.get("steps") or []
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ValueError("private_teacher_steps.steps must be a non-empty list")
        self.contract = copy.deepcopy(contract)
        self.steps = [copy.deepcopy(step) for step in raw_steps if isinstance(step, dict)]
        if not self.steps:
            raise ValueError("private_teacher_steps.steps contains no usable entries")
        self.cursor = 0
        runtime_mode = _mode(start_mode, "S")
        contract_mode = _mode(contract.get("start_mode"), runtime_mode)
        if runtime_mode != contract_mode:
            raise ValueError(
                f"runtime starts in {runtime_mode}, teacher contract starts in {contract_mode}"
            )
        self.active_mode = contract_mode
        self.phase_start_step = 1
        self.phase_action_count = 0
        # 同一阶段内 answer 被驳回的次数。教师收到驳回后常常原样再交,形成活锁:
        # 8500 批实测 202 条轨迹(6%)被驳回 >=20 次,最极端一条 200 步全在这个
        # 循环里,这 6% 烧掉了整批 64% 的 token。驳回反馈显然没有改变它的行为,
        # 继续给只是重复付钱,所以到上限就停。
        self.answer_rejections_in_phase = 0
        self.actions_since_check = 0
        self.no_progress_streak = 0
        self._last_action_fingerprint = ""
        self.boundary_ready = False
        self.final_ready = False
        self.completed = False
        self.bindings: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.pending_feedback: list[dict[str, Any]] = []
        self._normalize_cursor()

    def _normalize_cursor(self) -> None:
        while self.cursor < len(self.steps):
            kind = str(self.steps[self.cursor].get("type") or "").lower()
            if kind in {"phase", "switch"}:
                return
            self.cursor += 1

    @property
    def current(self) -> dict[str, Any] | None:
        if self.completed or self.cursor >= len(self.steps):
            return None
        return self.steps[self.cursor]

    @property
    def current_kind(self) -> str:
        return str((self.current or {}).get("type") or "").lower()

    @property
    def current_phase(self) -> dict[str, Any] | None:
        return self.current if self.current_kind == "phase" else None

    @property
    def current_phase_is_deep_search(self) -> bool:
        phase = self.current_phase
        if phase is None or _mode(phase.get("mode"), self.active_mode) != "S":
            return False
        return any(
            _clean_text(component.get("depth")).lower().replace("-", "_")
            in {"deep", "deep_search"}
            for component in phase.get("components") or []
            if isinstance(component, dict)
        )

    def phase_action_budget(
        self,
        default_max_actions: int,
        deep_search_max_actions: int | None = None,
    ) -> int:
        if self.current_phase_is_deep_search and deep_search_max_actions is not None:
            return deep_search_max_actions
        return default_max_actions

    @property
    def current_phase_index(self) -> int:
        phases = 0
        for index, step in enumerate(self.steps):
            if index > self.cursor:
                break
            if str(step.get("type") or "").lower() == "phase":
                phases += 1
        return max(1, phases)

    @property
    def is_final_phase(self) -> bool:
        if self.current_kind != "phase":
            return False
        return not any(
            str(step.get("type") or "").lower() == "phase"
            for step in self.steps[self.cursor + 1 :]
        )

    @property
    def expected_switch_to(self) -> str | None:
        if self.current_kind == "switch":
            return _mode((self.current or {}).get("to_mode"), "") or None
        if self.current_kind == "phase" and self.cursor + 1 < len(self.steps):
            next_step = self.steps[self.cursor + 1]
            if str(next_step.get("type") or "").lower() == "switch":
                return _mode(next_step.get("to_mode"), "") or None
        return None

    @property
    def requires_switch(self) -> bool:
        return self.current_kind == "switch" or self.boundary_ready

    @property
    def permits_answer(self) -> bool:
        return self.final_ready or self.completed

    def _site_lines(self, phase: dict[str, Any]) -> list[str]:
        seen: set[tuple[str, str]] = set()
        lines: list[str] = []
        for component in phase.get("components") or []:
            for site in component.get("sites") or []:
                if not isinstance(site, dict):
                    continue
                domain = _clean_text(site.get("domain"))
                url = _clean_text(site.get("entry_url"))
                key = (domain, url)
                if key in seen or not any(key):
                    continue
                seen.add(key)
                label = domain or "declared site"
                lines.append(f"- {label}" + (f": {url}" if url else ""))
        return lines

    def _binding_lines(self) -> list[str]:
        lines: list[str] = []
        for binding in self.bindings.values():
            label = _clean_text(binding.get("label") or "verified result")
            value = binding.get("value")
            if isinstance(value, (dict, list)):
                value_text = json.dumps(value, ensure_ascii=False)
            else:
                value_text = _clean_text(value)
            evidence = [str(step) for step in binding.get("evidence_steps") or []]
            suffix = f" (confirmed at trajectory step(s) {', '.join(evidence)})" if evidence else ""
            lines.append(f"- {label}: {value_text}{suffix}")
        return lines

    def render_teacher_context(self) -> str:
        """Compile the decision-model-safe current card.

        Internal IDs, completion predicates, dependency field names, future
        phases, and expected route metadata are deliberately absent.
        """
        lines = ["# Current rollout objective"]
        current = self.current or {}
        if self.current_kind == "switch":
            target = "search-based" if self.expected_switch_to == "S" else "operation-based"
            lines.extend(
                [
                    f"Hand control to the {target} path now by calling `switch`.",
                    "Base the switch reason only on the user request and the public trajectory so far.",
                    "Do not attempt another business action before this handoff.",
                ]
            )
        elif self.current_kind == "phase":
            components = [c for c in current.get("components") or [] if isinstance(c, dict)]
            goals = [_clean_text(component.get("goal")) for component in components]
            goals = [goal for goal in goals if goal]
            if len(goals) == 1:
                lines.append(goals[0])
            elif goals:
                lines.append("Complete all of the following related objectives in order:")
                lines.extend(f"- {goal}" for goal in goals)
            else:
                lines.append("Continue the current side of the user task using observable evidence.")
            sites = self._site_lines(current)
            if sites:
                lines.append("Relevant declared site(s):")
                lines.extend(sites)
            if self.boundary_ready:
                lines.extend(
                    [
                        "The current objective has been verified from the public trajectory.",
                        "Call `switch` now with a short, natural reason grounded in the work just completed.",
                    ]
                )
            elif self.final_ready:
                lines.extend(
                    [
                        "The requested work has been verified from the public trajectory.",
                        "Return the final answer now, using only the evidence already gathered.",
                    ]
                )
            else:
                lines.append(
                    "Do not give the final answer yet. Continue until this current objective is supported by tool evidence; then request a path switch if another path is needed."
                )

        binding_lines = self._binding_lines()
        if binding_lines:
            lines.append("Verified results available from earlier work:")
            lines.extend(binding_lines)
        if self.pending_feedback:
            lines.append("Controller feedback from the latest boundary check:")
            lines.extend(f"- {event['feedback']}" for event in self.pending_feedback)
        return "\n".join(lines)

    def verifier_reference(self) -> dict[str, Any]:
        phase = self.current_phase
        if phase is None:
            return {"kind": self.current_kind, "step": copy.deepcopy(self.current)}
        transition = None
        if self.cursor + 1 < len(self.steps):
            candidate = self.steps[self.cursor + 1]
            if str(candidate.get("type") or "").lower() == "switch":
                transition = copy.deepcopy(candidate)
        return {
            "kind": "phase",
            "phase": copy.deepcopy(phase),
            "transition_after_phase": transition,
            "is_final_phase": self.is_final_phase,
            "answer_schema": copy.deepcopy(self.contract.get("answer_schema") or []),
            "verified_bindings": copy.deepcopy(self.bindings),
        }

    def note_action(
        self,
        step: int,
        tool_call: dict[str, Any] | None = None,
        observation: Any = "",
    ) -> None:
        if self.current_kind != "phase":
            return
        self.phase_action_count += 1
        self.actions_since_check += 1

        observation_text = str(observation or "").strip()
        fingerprint = json.dumps(tool_call or {}, ensure_ascii=False, sort_keys=True)
        fingerprint += "\n" + observation_text
        lowered = observation_text.lower()
        explicit_failure = (
            "- status: failure" in lowered
            or lowered.startswith("error executing")
            or lowered.startswith("tool error")
        )
        exact_repeat = bool(
            self._last_action_fingerprint
            and fingerprint == self._last_action_fingerprint
        )
        self.no_progress_streak = (
            self.no_progress_streak + 1 if explicit_failure or exact_repeat else 0
        )
        self._last_action_fingerprint = fingerprint

    def should_periodic_check(
        self,
        interval: int,
        max_actions: int,
        no_progress_actions: int = 3,
        *,
        deep_search_max_actions: int | None = None,
    ) -> tuple[bool, str]:
        if self.current_kind != "phase" or self.boundary_ready or self.final_ready:
            return False, ""
        action_budget = self.phase_action_budget(
            max_actions,
            deep_search_max_actions,
        )
        if action_budget > 0 and self.phase_action_count >= action_budget:
            return True, "phase_budget"
        if no_progress_actions > 0 and self.no_progress_streak >= no_progress_actions:
            return True, "no_progress"
        if interval > 0 and self.actions_since_check >= interval:
            return True, "periodic"
        return False, ""

    def record_verifier_event(
        self,
        *,
        trigger: str,
        trajectory_step: int,
        verdict: dict[str, Any],
        raw_response: str,
        usage: dict[str, Any],
        feedback: str = "",
    ) -> int:
        event_id = len(self.events) + 1
        event = {
            "event_id": event_id,
            "trigger": trigger,
            "trajectory_step": trajectory_step,
            "phase_cursor": self.cursor,
            "phase_kind": self.current_kind,
            "verdict": copy.deepcopy(verdict),
            "raw_response": raw_response,
            "usage": copy.deepcopy(usage),
            "feedback": feedback,
            "feedback_delivered_to_step": None,
        }
        self.events.append(event)
        self.actions_since_check = 0
        self.no_progress_streak = 0
        if feedback:
            self.pending_feedback.append({"event_id": event_id, "feedback": feedback})
        return event_id

    def pending_feedback_ids(self) -> list[int]:
        return [int(item["event_id"]) for item in self.pending_feedback]

    def mark_feedback_delivered(self, teacher_step: int) -> list[int]:
        ids = self.pending_feedback_ids()
        for event_id in ids:
            self.events[event_id - 1]["feedback_delivered_to_step"] = teacher_step
        self.pending_feedback = []
        return ids

    def _store_bindings(self, verdict: dict[str, Any]) -> None:
        evidence = [
            int(step) for step in verdict.get("evidence_steps") or []
            if isinstance(step, int) or str(step).isdigit()
        ]
        values = verdict.get("produced_bindings") or {}
        if not isinstance(values, dict):
            return
        phase = self.current_phase or {}
        components = [c for c in phase.get("components") or [] if isinstance(c, dict)]
        declared = {
            str(component.get("produces") or ""): _clean_text(component.get("goal"))
            for component in components
            if component.get("produces")
        }
        for key, value in values.items():
            key = str(key)
            if declared and key not in declared:
                continue
            self.bindings[key] = {
                "label": declared.get(key) or "verified result",
                "value": copy.deepcopy(value),
                "evidence_steps": evidence,
            }

    def apply_phase_verdict(self, verdict: dict[str, Any]) -> None:
        if self.current_kind != "phase":
            return
        complete = bool(verdict.get("phase_complete"))
        if not complete:
            return
        self._store_bindings(verdict)
        if self.is_final_phase:
            self.final_ready = True
        else:
            self.boundary_ready = True

    def decide_switch(self, verifier_complete: bool | None = None) -> ControlDecision:
        if self.current_kind == "switch":
            return ControlDecision(allow=True, advance=True, reason="planned current handoff")
        if self.current_kind != "phase":
            return ControlDecision(False, feedback="No path handoff is available now.")
        complete = self.boundary_ready if verifier_complete is None else verifier_complete
        if not complete:
            return ControlDecision(
                False,
                feedback=(
                    "The current objective is not complete yet. Continue on the current path and gather the missing public evidence or visible operation result before switching."
                ),
                reason="current phase not verified",
            )
        if self.is_final_phase:
            return ControlDecision(
                False,
                feedback="The current objective is complete and no path handoff is needed. Return the evidence-grounded final answer.",
                reason="final phase should answer",
            )
        return ControlDecision(allow=True, advance=True, reason="current phase verified")

    ANSWER_REJECT_CAP = max(1, int(os.environ.get("HYBRID_ANSWER_REJECT_CAP", "8")))

    def _reject_answer(self, feedback: str, reason: str) -> ControlDecision:
        self.answer_rejections_in_phase += 1
        if self.answer_rejections_in_phase > self.ANSWER_REJECT_CAP:
            return ControlDecision(
                False, terminate=True, feedback=feedback,
                reason=f"answer rejected {self.answer_rejections_in_phase}x in one phase",
            )
        return ControlDecision(False, feedback=feedback, reason=reason)

    def decide_answer(self, verifier_complete: bool | None = None) -> ControlDecision:
        if self.current_kind != "phase":
            return self._reject_answer(
                "Do not answer yet. Complete the current path handoff first.",
                "pending handoff",
            )
        complete = self.final_ready if verifier_complete is None else verifier_complete
        if complete and self.is_final_phase:
            return ControlDecision(True, final=True, reason="final phase verified")
        if self.boundary_ready:
            return self._reject_answer(
                "The current objective is complete, but the overall task is not. "
                "Call `switch` now with a natural reason grounded in the work just completed.",
                "current phase complete but another path remains",
            )
        return self._reject_answer(
            "Do not answer yet. The current objective still lacks required public evidence "
            "or a visibly confirmed operation result. Continue on the current path.",
            "premature answer",
        )

    def advance_after_switch(self, actual_to_mode: str, next_step: int) -> None:
        self.answer_rejections_in_phase = 0   # 计数是"每阶段"的
        expected = self.expected_switch_to
        if expected and _mode(actual_to_mode, "") != expected:
            raise ValueError(f"switch target mismatch: expected {expected}, got {actual_to_mode}")
        if self.current_kind == "switch":
            self.cursor += 1
        elif self.current_kind == "phase" and self.boundary_ready:
            self.cursor += 1
            if self.cursor < len(self.steps) and str(self.steps[self.cursor].get("type") or "").lower() == "switch":
                self.cursor += 1
        else:
            raise RuntimeError("cannot advance an unverified phase")
        self._normalize_cursor()
        self.active_mode = _mode(actual_to_mode, self.active_mode)
        self.phase_start_step = next_step
        self.phase_action_count = 0
        self.actions_since_check = 0
        self.no_progress_streak = 0
        self._last_action_fingerprint = ""
        self.boundary_ready = False
        self.final_ready = False
        self.pending_feedback = []
        if self.cursor >= len(self.steps):
            self.completed = True

    def mark_completed(self) -> None:
        self.completed = True

    def public_state(self) -> dict[str, Any]:
        """Metadata safe to put in result.json (no private contract text)."""
        return {
            "enabled": True,
            "current_phase_number": self.current_phase_index,
            "current_kind": self.current_kind,
            "phase_start_step": self.phase_start_step,
            "phase_action_count": self.phase_action_count,
            "no_progress_streak": self.no_progress_streak,
            "boundary_ready": self.boundary_ready,
            "final_ready": self.final_ready,
            "completed": self.completed,
            "verified_binding_count": len(self.bindings),
            "verifier_event_count": len(self.events),
            "feedback_event_count": sum(bool(event.get("feedback")) for event in self.events),
        }


_INTERNAL_LABEL_RE = re.compile(
    r"\b(?:phase|component|policy)(?:[\s_-]*(?:id)?[\s_-]*)?[A-Za-z]*\d+\b|"
    r"\b(?:done_when|consumes|produces|private_teacher_steps|expected_mode_route|"
    r"expected_switch_count)\b",
    re.IGNORECASE,
)


def sanitize_teacher_feedback(value: Any, fallback: str) -> str:
    """Keep verifier feedback short and strip obvious private-schema labels."""
    text = _clean_text(value)
    text = _INTERNAL_LABEL_RE.sub("the current objective", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return fallback
    return text[:1000]
