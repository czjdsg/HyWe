#!/usr/bin/env python3
"""Convert synthetic hybrid tasks into private-teacher v2 ReAct inputs.

The public task stays unchanged.  The standard solution is stored in a private
field consumed by ``run_hybrid_web_agent.py`` and scrubbed from saved results.
Only O-start tasks receive ``start_url``; S-start tasks intentionally omit it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_TASKS = Path("runs/hybrid_task_design_v2_glm_5k/tasks.jsonl")
DEFAULT_OUT = Path("runs/hybrid_rollout_v2_smoke/input.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--task-ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-iter", type=int, default=200)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def select_tasks(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.task_ids:
        by_id = {str(row.get("task_id")): row for row in rows}
        missing = [task_id for task_id in args.task_ids if task_id not in by_id]
        if missing:
            raise ValueError(f"Unknown task ids: {', '.join(missing)}")
        selected = [by_id[task_id] for task_id in args.task_ids]
    else:
        selected = list(rows)
        random.Random(args.seed).shuffle(selected)
    if args.limit > 0:
        selected = selected[: args.limit]
    return selected


def normalize_policies(solution: dict[str, Any]) -> list[dict[str, Any]]:
    policies = solution.get("switch_policies")
    if isinstance(policies, list) and policies:
        return [policy for policy in policies if isinstance(policy, dict)]
    policy = solution.get("switch_policy")
    return [policy] if isinstance(policy, dict) and policy else []


def find_policy(
    policies: list[dict[str, Any]],
    from_phase: str,
    to_phase: str,
    from_mode: str,
    to_mode: str,
) -> dict[str, Any]:
    for policy in policies:
        if (
            str(policy.get("from_phase")) == from_phase
            and str(policy.get("to_phase")) == to_phase
        ):
            return policy
    for policy in policies:
        if str(policy.get("from")) == from_mode and str(policy.get("to")) == to_mode:
            return policy
    raise ValueError(
        f"Missing switch policy {from_phase}/{from_mode} -> {to_phase}/{to_mode}"
    )


def site_view(site: dict[str, Any]) -> dict[str, Any]:
    return {
        "site_id": site.get("site_id"),
        "domain": site.get("domain"),
        "entry_url": site.get("entry_url"),
        "role": site.get("role"),
    }


def switch_step(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "switch",
        "policy_id": policy.get("policy_id"),
        "from_phase": policy.get("from_phase"),
        "to_phase": policy.get("to_phase"),
        "from_mode": policy.get("from"),
        "to_mode": policy.get("to"),
        "kind": policy.get("kind"),
        "trigger": policy.get("trigger"),
        "minimum_attempt": policy.get("minimum_attempt"),
        "forbidden_early_switch": policy.get("forbidden_early_switch"),
    }


def phase_step(
    phase: dict[str, Any],
    sites_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    components: list[dict[str, Any]] = []
    for component in phase.get("components") or []:
        scopes = component.get("site_scope") or []
        if isinstance(scopes, str):
            scopes = [scopes]
        if not scopes and component.get("site_id"):
            scopes = [component["site_id"]]
        component_sites = [site_view(sites_by_id[s]) for s in scopes if s in sites_by_id]
        components.append(
            {
                "component_id": component.get("component_id"),
                "mode": component.get("mode"),
                "prototype_id": component.get("prototype_id"),
                "prototype_name": component.get("prototype_name"),
                "depth": component.get("depth"),
                "sites": component_sites,
                "goal": component.get("goal"),
                "consumes": component.get("consumes") or [],
                "produces": component.get("produces"),
                "done_when": component.get("done_when"),
            }
        )
    return {
        "type": "phase",
        "phase_id": phase.get("phase_id"),
        "mode": phase.get("mode"),
        "is_fallback_attempt": bool(phase.get("attempt")),
        "components": components,
    }


def build_teacher_steps(task: dict[str, Any]) -> dict[str, Any]:
    solution = task.get("solution") or {}
    phases = [phase for phase in solution.get("phases") or [] if isinstance(phase, dict)]
    policies = normalize_policies(solution)
    sites = [site for site in task.get("sites") or [] if isinstance(site, dict)]
    sites_by_id = {str(site.get("site_id")): site for site in sites}
    start_mode = str(solution.get("start_mode") or "S").upper()
    steps: list[dict[str, Any]] = []

    previous_phase = "entry"
    previous_mode = start_mode
    for phase in phases:
        phase_id = str(phase.get("phase_id") or "")
        mode = str(phase.get("mode") or "").upper()
        if mode != previous_mode:
            policy = find_policy(
                policies,
                from_phase=previous_phase,
                to_phase=phase_id,
                from_mode=previous_mode,
                to_mode=mode,
            )
            steps.append(switch_step(policy))
        steps.append(phase_step(phase, sites_by_id))
        previous_phase = phase_id
        previous_mode = mode

    contract = task.get("difficulty_contract") or {}
    expected_switches = int(contract.get("switch_count") or 0)
    built_switches = sum(step.get("type") == "switch" for step in steps)
    if built_switches != expected_switches:
        raise ValueError(
            f"{task.get('task_id')}: built {built_switches} switches, "
            f"contract requires {expected_switches}"
        )

    return {
        "source_task_id": task.get("task_id"),
        "switch_form": task.get("switch_form"),
        "difficulty": task.get("difficulty"),
        "start_mode": start_mode,
        "expected_mode_route": contract.get("mode_route") or [],
        "expected_switch_count": expected_switches,
        "cross_site_required": bool(contract.get("cross_site_required")),
        "sites": [site_view(site) for site in sites],
        "steps": steps,
        "dependency_edges": solution.get("dependency_edges") or [],
        "answer_schema": task.get("answer_schema") or [],
        "coherence_rationale": task.get("coherence_rationale"),
    }


def convert(task: dict[str, Any], max_iter: int) -> dict[str, Any]:
    solution = task.get("solution") or {}
    start_mode = str(solution.get("start_mode") or "S").upper()
    sites = [site for site in task.get("sites") or [] if isinstance(site, dict)]
    row: dict[str, Any] = {
        "task_id": task.get("task_id"),
        "question": task.get("question"),
        "max_iter": max_iter,
        "source": {
            "switch_form": task.get("switch_form"),
            "difficulty": task.get("difficulty"),
            "blueprint_id": task.get("blueprint_id"),
        },
        "private_teacher_steps": build_teacher_steps(task),
    }
    if start_mode == "O":
        start_url = str((sites[0] if sites else {}).get("entry_url") or "").strip()
        if not start_url:
            raise ValueError(f"{task.get('task_id')}: O-start task has no entry URL")
        row["start_url"] = start_url
    elif start_mode != "S":
        raise ValueError(f"{task.get('task_id')}: unsupported start mode {start_mode!r}")
    return row


def main() -> None:
    args = parse_args()
    if args.max_iter <= 0:
        raise ValueError("--max-iter must be positive")
    selected = select_tasks(read_jsonl(args.tasks), args)
    converted = [convert(task, args.max_iter) for task in selected]
    for row in converted:
        start_mode = row["private_teacher_steps"]["start_mode"]
        if (start_mode == "O") != bool(row.get("start_url")):
            raise AssertionError(f"start_url invariant failed for {row['task_id']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in converted:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in converted:
        teacher = row["private_teacher_steps"]
        key = f"{teacher['switch_form']}/{teacher['difficulty']}/{teacher['start_mode']}-start"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"output": str(args.out), "tasks": len(converted), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
