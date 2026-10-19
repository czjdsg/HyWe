#!/usr/bin/env python3
"""Generate ungrounded hybrid task designs for the four switch forms.

This pipeline deliberately designs tasks before feasibility and evidence checks.
It composes typed S/O prototypes with site profiles, emits an explicit solution
contract, and can ask GLM to turn deterministic blueprints into natural tasks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Iterable
from urllib.parse import urlparse

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collector.embedding import JinaPoolEmbedder, extract_json


DEFAULT_SEED = Path(__file__).with_name("hybrid_task_design.seed.json")
DEFAULT_POOL = ROOT / "collector/data/hybrid_pool_insta2k_expanded.json"
DEFAULT_DETAIL = ROOT / "collector/data/insta_screened.detail.jsonl"
DEFAULT_OUT = ROOT / "runs/task_design"
# 默认配比。定向补数据时用 HYBRID_FORM_RATIOS 覆盖(JSON),例如只补 F2/F3/F4:
#   HYBRID_FORM_RATIOS='{"F2":0.3,"F3":0.4,"F4":0.3}'
# 之所以做成环境变量而不是命令行参数:这一步常被别的脚本包起来调,env 传递不用
# 逐层加 flag。键必须是 F1..F4 的子集,值会被归一化。
FORM_RATIOS = {"F1": 0.32, "F2": 0.28, "F3": 0.25, "F4": 0.15}
_form_override = os.environ.get("HYBRID_FORM_RATIOS", "").strip()
if _form_override:
    _parsed = json.loads(_form_override)
    if not _parsed or not set(_parsed) <= {"F1", "F2", "F3", "F4"}:
        raise ValueError(f"HYBRID_FORM_RATIOS 键只能是 F1..F4,收到 {sorted(_parsed)}")
    _total = float(sum(_parsed.values()))
    if _total <= 0:
        raise ValueError("HYBRID_FORM_RATIOS 的值之和必须为正")
    FORM_RATIOS = {k: v / _total for k, v in _parsed.items()}
FORM_DIFFICULTY_RATIOS = {
    "F1": {"standard": 1.0},
    "F2": {"standard": 1.0},
    "F3": {"easy": 0.4, "medium": 0.4, "hard": 0.2},
    "F4": {"easy": 0.4, "medium": 0.4, "hard": 0.2},
}
SITE_COUNT_PATTERNS = {
    "standard": {"F1": 1, "F2": 1},
    "easy": {"F3": 1, "F4": 1},
    "medium": {"F3": 2, "F4": 2},
    "hard": {"F3": 3, "F4": 3},
}
ROUTE_PATTERNS = {
    "standard": {
        "F1": ["O", "S"],
        "F2": ["S", "O"],
    },
    "easy": {
        "F3": ["S", "O"],
        "F4": ["O", "S"],
    },
    "medium": {
        "F3": ["S", "O", "S", "O"],
        "F4": ["O", "S", "O"],
    },
    "hard": {
        "F3": ["S", "O", "S", "O", "S", "O"],
        "F4": ["O", "S", "O", "S", "O"],
    },
}
DOMAIN_OPERATION_OVERRIDES = {
    "airbnb.com": "O_STAY_AVAILABILITY",
    "booking.com": "O_STAY_AVAILABILITY",
    "marriott.com": "O_STAY_AVAILABILITY",
}
META_TASK_RE = re.compile(
    r"\b(?:switch\s+(?:to|from|between|modes?|tools?)|operation mode|operation path|"
    r"search mode|search path|S[- ]side|O[- ]side)\b",
    re.I,
)


def normalize_site_domain(value: object) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    parsed = urlparse(text if "://" in text else f"https://{text}")
    host = (parsed.hostname or "").removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path or "").rstrip("/")
    return host + path


def site_aliases(site: dict) -> set[str]:
    domain = normalize_site_domain(site.get("domain"))
    entry = normalize_site_domain(site.get("entry_url"))
    aliases = {domain, entry} - {""}
    domain_host = domain.split("/", 1)[0]
    entry_host = entry.split("/", 1)[0]
    if entry_host and entry_host != domain_host:
        aliases.add(entry_host)
    return aliases


def stable_id(prefix: str, *parts: object) -> str:
    raw = "\0".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def allocate(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value / sum(ratios.values()) for key, value in ratios.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(result.values())
    order = sorted(raw, key=lambda key: raw[key] - result[key], reverse=True)
    for key in order[:remaining]:
        result[key] += 1
    return result


def prototype_text(row: dict) -> str:
    return "\n".join(
        [
            f"id: {row.get('id')}",
            f"name: {row.get('name')}",
            f"depth: {row.get('depth')}",
            f"goal: {row.get('goal')}",
            "categories: " + ", ".join(row.get("categories") or []),
            "domain patterns: " + ", ".join(row.get("domain_patterns") or []),
            "entity space: " + ", ".join(row.get("entity_space") or []),
            "scenario patterns: " + "; ".join(row.get("scenario_patterns") or []),
            f"question form: {row.get('question_skeleton')}",
            f"output type: {row.get('output_type')}",
            "output fields: " + ", ".join(row.get("output_fields") or []),
        ]
    )


def build_catalog(seed: dict, embedder: JinaPoolEmbedder, path: Path) -> dict:
    operations = [dict(row) for row in seed.get("operation_prototypes") or []]
    searches = [dict(row) for row in seed.get("search_prototypes") or []]
    rows = [*operations, *searches]
    vectors = embedder.embed([prototype_text(row) for row in rows])
    for row, vector in zip(rows, vectors):
        row["embedding_text"] = prototype_text(row)
        row["embedding"] = vector
    catalog = {
        "schema_version": seed.get("schema_version", "1.0"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "switch_forms": seed.get("switch_forms") or {},
        "difficulty_profiles": seed.get("difficulty_profiles") or {},
        "embedding": {
            "backend": "jina_embeddings",
            "base_url": embedder.base_url,
            "model": embedder.model,
            "task": embedder.task,
            "dimension": embedder.dimension,
        },
        "operation_prototypes": operations,
        "search_prototypes": searches,
    }
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    return catalog


def load_sites(pool_path: Path, detail_path: Path) -> list[dict]:
    raw = json.loads(pool_path.read_text(encoding="utf-8"))
    pool = raw if isinstance(raw, list) else raw.get("all_usable") or []
    details = {row.get("domain"): row for row in read_jsonl(detail_path) if row.get("domain")}
    sites = []
    for row in pool:
        if not isinstance(row, dict) or not row.get("domain") or not row.get("category"):
            continue
        detail = details.get(row["domain"]) or {}
        sites.append(
            {
                "domain": row["domain"],
                "entry_url": row.get("entry_url") or "",
                "category": row["category"],
                "image_required": bool(row.get("image_required")),
                "suggested_switch_kind": row.get("suggested_switch_kind") or "",
                "capability_hint": (
                    detail.get("instruction")
                    or row.get("switch_trigger_observation")
                    or row.get("evidence")
                    or ""
                )[:1200],
                # 抽样权重必须一起带过来:这里是按白名单重建 dict 的,漏一个字段就
                # 静默变成默认值。加权抽样刚上线时就栽在这 —— 权重写进池子文件了,
                # 抽出来的分布却纹丝不动,查到最后是这一行没带 weight。
                "weight": row.get("weight", 1),
            }
        )
    return sites


def compatible_operations(site: dict, operations: list[dict]) -> list[dict]:
    category_rows = [
        row for row in operations if site.get("category") in (row.get("categories") or [])
    ]
    domain = str(site.get("domain") or "").lower()
    override_id = DOMAIN_OPERATION_OVERRIDES.get(domain)
    if override_id:
        override = [row for row in category_rows if row.get("id") == override_id]
        if override:
            return override
    explicit = [
        row
        for row in category_rows
        if any(pattern.lower() in domain for pattern in row.get("domain_patterns") or [])
    ]
    if explicit:
        return explicit
    return [row for row in category_rows if not row.get("domain_patterns")]


def weighted_sample_sites(rows: list[dict], count: int,
                          rng: random.Random) -> list[dict]:
    """按 `weight` 加权、且**按域名去重**地抽 count 个站点。

    不能用复制条目来加权:选站走的是 rng.sample(无放回),池子里放两份同域名的记录,
    一个双站任务就可能把同一个站抽两次,产出"从 X 站查到 X 站"这种无意义的跨站任务。
    所以权重写在字段里,这里逐个抽、抽中即从候选里剔掉同域名的其余条目。
    weight 缺省为 1;常用站(benchmark 覆盖到的那些)给更高的值。
    """
    pool = list(rows)
    picked: list[dict] = []
    seen: set[str] = set()
    while pool and len(picked) < count:
        weights = [max(0.0, float(r.get("weight", 1) or 1)) for r in pool]
        total = sum(weights)
        if total <= 0:
            choice = rng.choice(pool)
        else:
            x = rng.random() * total
            acc = 0.0
            choice = pool[-1]
            for r, w in zip(pool, weights):
                acc += w
                if x <= acc:
                    choice = r
                    break
        dom = str(choice.get("domain") or "").lower()
        picked.append(choice)
        seen.add(dom)
        pool = [r for r in pool if str(r.get("domain") or "").lower() not in seen]
    if len(picked) < count:
        # 候选不足时退回原行为,保持与旧版一致的失败模式
        return rng.sample(rows, count)
    return picked


def choose_distinct(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    if not rows:
        raise ValueError("cannot choose from an empty prototype set")
    if len(rows) >= count:
        return rng.sample(rows, count)
    return [rows[index % len(rows)] for index in range(count)]


def cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def choose_related(
    rows: list[dict],
    count: int,
    rng: random.Random,
    anchor: dict,
) -> list[dict]:
    """Keep semantic neighbors in the candidate pool while retaining diversity."""
    if not rows:
        raise ValueError("cannot choose from an empty prototype set")
    anchor_vector = anchor.get("embedding")
    if not anchor_vector or not any(row.get("embedding") for row in rows):
        return choose_distinct(rows, count, rng)
    ranked = sorted(
        rows,
        key=lambda row: cosine_similarity(anchor_vector, row.get("embedding")),
        reverse=True,
    )
    pool = ranked[: min(len(ranked), max(count * 3, 6))]
    return choose_distinct(pool, count, rng)


def prefer_depth(rows: list[dict], difficulty: str) -> list[dict]:
    if difficulty == "easy":
        shallow = [row for row in rows if row.get("depth", "shallow") == "shallow"]
        return shallow or rows
    if difficulty == "hard":
        deep = [row for row in rows if row.get("depth") == "deep"]
        return deep or rows
    return rows


def component(
    component_id: str,
    mode: str,
    prototype: dict,
    *,
    produces: str,
    consumes: list[str] | None = None,
    site_id: str | None = None,
    site_scope: list[str] | None = None,
) -> dict:
    return {
        "component_id": component_id,
        "mode": mode,
        "prototype_id": prototype["id"],
        "prototype_name": prototype["name"],
        "depth": prototype.get("depth", "interactive" if mode == "O" else "shallow"),
        "output_type": prototype.get("output_type") or "",
        "goal": prototype["goal"],
        "produces": produces,
        "consumes": consumes or [],
        "site_id": site_id,
        "site_scope": site_scope or ([site_id] if site_id else []),
    }


def draft_question(sites: list[dict], phases: list[dict]) -> str:
    """Make a readable deterministic placeholder without exposing S/O routing."""
    domains = ", ".join(site["domain"] for site in sites)
    goals = [phase["components"][0]["goal"] for phase in phases]
    if len(goals) == 1:
        return f"For {domains}, {goals[0]}."
    sequence = "; then use that exact result to ".join(goals)
    return f"Across {domains}, complete one linked request: first {sequence}."


def effective_mode_route(start_mode: str, phases: list[dict]) -> list[str]:
    route = [start_mode]
    for phase in phases:
        mode = str(phase.get("mode") or "")
        if mode and mode != route[-1]:
            route.append(mode)
    return route


def build_switch_policies(
    form: str,
    phases: list[dict],
    start_mode: str,
    difficulty_profile: dict,
) -> list[dict]:
    boundaries: list[tuple[str, str, str, str, dict | None]] = []
    if phases and start_mode != phases[0]["mode"]:
        boundaries.append(("entry", phases[0]["phase_id"], start_mode, phases[0]["mode"], None))
    for left, right in zip(phases, phases[1:]):
        if left["mode"] != right["mode"]:
            boundaries.append(
                (left["phase_id"], right["phase_id"], left["mode"], right["mode"], left)
            )

    policies = []
    for index, (from_phase, to_phase, from_mode, to_mode, left_phase) in enumerate(boundaries, 1):
        left_component = (left_phase or {}).get("components", [{}])[0]
        if from_phase == "entry" and form == "F1":
            kind = "immediate_correction"
            trigger = "before any browser action, once task analysis shows that the next required result is public search work"
            minimum_attempt = "none; do not spend an O action merely to test whether S is appropriate"
            forbidden = "do not perform a browser action before making the correction"
        elif left_component.get("component_id") == "s_attempt":
            kind = "fallback_after_insufficiency"
            trigger = "only after the bounded exact-search attempt establishes structural unavailability of the requested live state"
            minimum_attempt = difficulty_profile["attempt_rigor"]
            forbidden = "do not switch after one weak query or one imperfect result"
        elif from_mode == "S" and to_mode == "O":
            kind = "causal_input_ready"
            trigger = "after the preceding S result has been resolved to the exact value required by the next site's controls"
            minimum_attempt = "complete the producer and preserve the resolved entity, identifier, constraint, threshold, or candidate set"
            forbidden = "do not enter O while the next operation still contains an unresolved input"
        else:
            kind = "result_binds_search"
            trigger = "after the preceding O phase has surfaced the exact result that defines the next search question"
            minimum_attempt = "finish the operation and preserve the surfaced names, identifiers, values, and comparison basis"
            forbidden = "do not issue a generic follow-up search before the operation has bound its subject"
        policies.append(
            {
                "policy_id": f"sw{index}",
                "from_phase": from_phase,
                "to_phase": to_phase,
                "from": from_mode,
                "to": to_mode,
                "kind": kind,
                "trigger": trigger,
                "minimum_attempt": minimum_attempt,
                "forbidden_early_switch": forbidden,
            }
        )
    return policies


def choose_search_component_prototype(
    searches: list[dict],
    operation: dict,
    rng: random.Random,
    *,
    role: str,
    deep: bool = False,
) -> dict:
    if role == "standalone":
        candidates = [row for row in searches if "standalone" in (row.get("roles") or [])]
    elif role == "producer":
        candidates = [
            row
            for row in searches
            if "producer" in (row.get("roles") or [])
            and row.get("output_type") in (operation.get("accepts_s_outputs") or [])
        ]
    elif role == "consumer":
        candidates = [
            row
            for row in searches
            if "consumer" in (row.get("roles") or [])
            and operation.get("output_type") in (row.get("accepts_o_outputs") or [])
        ]
    elif role == "bridge":
        candidates = [
            row
            for row in searches
            if {"consumer", "producer"}.issubset(set(row.get("roles") or []))
            and operation.get("output_type") in (row.get("accepts_o_outputs") or [])
            and row.get("output_type") in (operation.get("accepts_s_outputs") or [])
        ]
    else:
        raise ValueError(f"unknown search role {role}")
    if deep:
        candidates = [row for row in candidates if row.get("depth") == "deep"] or candidates
    else:
        candidates = [row for row in candidates if row.get("depth", "shallow") == "shallow"] or candidates
    return choose_related(candidates, 1, rng, operation)[0]


def build_blueprint(
    index: int,
    form: str,
    difficulty: str,
    sites: list[dict],
    operation: dict,
    searches: list[dict],
    difficulty_profile: dict,
    alternatives: list[dict],
    rng: random.Random,
) -> dict:
    site_count = int(difficulty_profile["site_count_by_form"][form])
    if len(sites) != site_count:
        raise ValueError(f"{difficulty} requires {site_count} sites, got {len(sites)}")
    target_sites = [
        {**site, "site_id": f"site_{offset}", "role": "target"}
        for offset, site in enumerate(sites, 1)
    ]
    phases: list[dict] = []
    s_counter = 0
    o_counter = 0
    prior_o_outputs: list[str] = []

    def add_phase(mode: str, row: dict, *, attempt: bool = False) -> dict:
        phase = {
            "phase_id": f"p{len(phases) + 1}",
            "mode": mode,
            "attempt": attempt,
            "components": [row],
        }
        phases.append(phase)
        return row

    def add_search(
        role: str,
        *,
        consumes: list[str] | None = None,
        site_scope: list[str] | None = None,
        deep: bool = False,
    ) -> dict:
        nonlocal s_counter
        s_counter += 1
        prototype = choose_search_component_prototype(
            searches, operation, rng, role=role, deep=deep
        )
        return add_phase(
            "S",
            component(
                f"s{s_counter}",
                "S",
                prototype,
                produces=f"s{s_counter}.result",
                consumes=consumes,
                site_id=site_scope[0] if site_scope and len(site_scope) == 1 else None,
                site_scope=site_scope,
            ),
        )

    def add_operation(site_index: int, consumes: list[str] | None = None) -> dict:
        nonlocal o_counter
        o_counter += 1
        site_id = target_sites[site_index]["site_id"]
        component_row = component(
            f"o{o_counter}",
            "O",
            operation,
            produces=f"o{o_counter}.result",
            consumes=consumes,
            site_id=site_id,
        )
        add_phase("O", component_row)
        prior_o_outputs.append(component_row["produces"])
        return component_row

    if form == "F1":
        first_role = "standalone" if site_count == 1 else "producer"
        previous = add_search(first_role, site_scope=[target_sites[0]["site_id"]])
        for site_index in range(1, site_count):
            consumes = [previous["produces"]]
            if difficulty == "hard" and site_index == site_count - 1:
                consumes.extend(prior_o_outputs)
            previous = add_operation(site_index, list(dict.fromkeys(consumes)))
            if site_index < site_count - 1:
                previous = add_search(
                    "bridge",
                    consumes=[previous["produces"]],
                    site_scope=[
                        target_sites[site_index]["site_id"],
                        target_sites[site_index + 1]["site_id"],
                    ],
                )
    elif form in {"F2", "F3"}:
        first_site_id = target_sites[0]["site_id"]
        if form == "F2":
            attempt_component = {
                "component_id": "s_attempt",
                "mode": "S",
                "prototype_id": "S_EXACT_REQUEST_ATTEMPT",
                "prototype_name": "Exact-request search attempt",
                "depth": "shallow",
                "output_type": "insufficiency_verdict",
                "goal": "search for the exact fully constrained live target, not a generic version of it",
                "produces": "s_attempt.insufficiency_verdict",
                "consumes": [],
                "site_id": first_site_id,
                "site_scope": [first_site_id],
            }
            previous = add_phase("S", attempt_component, attempt=True)
        else:
            previous = add_search(
                "producer",
                site_scope=[first_site_id],
                deep=difficulty == "hard",
            )
        previous = add_operation(0, [previous["produces"]])
        for site_index in range(1, site_count):
            previous = add_search(
                "bridge",
                consumes=[previous["produces"]],
                site_scope=[
                    target_sites[site_index - 1]["site_id"],
                    target_sites[site_index]["site_id"],
                ],
            )
            consumes = [previous["produces"]]
            if difficulty == "hard" and site_index == site_count - 1:
                consumes.extend(prior_o_outputs)
            previous = add_operation(site_index, list(dict.fromkeys(consumes)))
    else:
        previous = add_operation(0)
        if site_count == 1:
            previous = add_search(
                "consumer",
                consumes=[previous["produces"]],
                site_scope=[target_sites[0]["site_id"]],
            )
        else:
            for site_index in range(1, site_count):
                previous = add_search(
                    "bridge",
                    consumes=[previous["produces"]],
                    site_scope=[
                        target_sites[site_index - 1]["site_id"],
                        target_sites[site_index]["site_id"],
                    ],
                    deep=difficulty == "hard" and site_index == 1,
                )
                consumes = [previous["produces"]]
                if difficulty == "hard" and site_index == site_count - 1:
                    consumes.extend(prior_o_outputs)
                previous = add_operation(site_index, list(dict.fromkeys(consumes)))

    start_mode = {"F1": "O", "F2": "S", "F3": "S", "F4": "O"}[form]
    switch_policies = build_switch_policies(
        form, phases, start_mode, difficulty_profile
    )
    components = [phase["components"][0] for phase in phases]
    producers = {row["produces"]: row for row in components}
    edges = []
    for target in components:
        for source_ref in target.get("consumes") or []:
            source = producers[source_ref]
            if source["component_id"] == "s_attempt":
                relation = "fallback_gate"
            elif source["mode"] == "O" and target["mode"] == "S":
                relation = "query_binding"
            elif source["mode"] == "S" and target["mode"] == "O":
                relation = "required_input"
            else:
                relation = "accumulated_comparison_context"
            edges.append(
                {"from": source_ref, "to": target["component_id"], "relation": relation}
            )

    used_modes_by_site = {site["site_id"]: set() for site in target_sites}
    for row in components:
        for site_id in row.get("site_scope") or []:
            used_modes_by_site[site_id].add(row["mode"])
    for site in target_sites:
        modes = used_modes_by_site[site["site_id"]]
        site["role"] = (
            "research_and_operation" if modes == {"S", "O"}
            else "operation_target" if "O" in modes
            else "research_target"
        )

    blueprint_id = stable_id(
        "design",
        index,
        form,
        difficulty,
        *(site["domain"] for site in target_sites),
        operation["id"],
        *(row["prototype_id"] for row in components),
    )
    route = effective_mode_route(start_mode, phases)
    cardinality_range = difficulty_profile.get("entity_cardinality_range")
    entity_cardinality = (
        rng.randint(int(cardinality_range[0]), int(cardinality_range[1]))
        if cardinality_range
        else int(difficulty_profile["entity_cardinality"])
    )
    return {
        "blueprint_id": blueprint_id,
        "switch_form": form,
        "difficulty": difficulty,
        "site": target_sites[0],
        "sites": target_sites,
        "alternative_sites": alternatives,
        "entity_cardinality": entity_cardinality,
        "difficulty_contract": {
            "site_count": len(target_sites),
            "phase_count": len(phases),
            "switch_count": len(switch_policies),
            "mode_route": route,
            "cross_site_required": form in (difficulty_profile.get("cross_site_forms") or []),
            "max_components_per_phase": int(difficulty_profile["max_components_per_phase"]),
            "dependency_shape": difficulty_profile["dependency_shape"],
            "deep_search_required": bool(difficulty_profile["deep_search_required"]),
        },
        "operation_prototype": {
            key: operation.get(key)
            for key in (
                "id",
                "name",
                "goal",
                "entity_space",
                "output_type",
                "output_fields",
                "scenario_patterns",
            )
        },
        "draft_question": draft_question(target_sites, phases),
        "solution": {
            "start_mode": start_mode,
            "phases": phases,
            "switch_policy": switch_policies[0],
            "switch_policies": switch_policies,
            "dependency_edges": edges,
        },
        "expected_answer_fields": (
            list(operation.get("output_fields") or [])
            if any(row["mode"] == "O" for row in components)
            else ["publicly sourced facts requested by the S components"]
        ),
    }


def validate_blueprint(row: dict) -> str:
    form = row.get("switch_form")
    difficulty = row.get("difficulty")
    solution = row.get("solution") or {}
    phases = solution.get("phases") or []
    modes = [phase.get("mode") for phase in phases]
    components = [component for phase in phases for component in phase.get("components") or []]
    ids = {component.get("component_id") for component in components}
    produced = {component.get("produces") for component in components}
    consumed = {value for component in components for value in component.get("consumes") or []}
    if form not in {"F1", "F2", "F3", "F4"}:
        return "unknown switch form"
    if difficulty not in ROUTE_PATTERNS or form not in ROUTE_PATTERNS[difficulty]:
        return f"difficulty {difficulty} is not defined for {form}"
    expected_start = {"F1": "O", "F2": "S", "F3": "S", "F4": "O"}[form]
    if solution.get("start_mode") != expected_start:
        return f"invalid start mode for {form}"
    expected_route = ROUTE_PATTERNS[difficulty][form]
    actual_route = effective_mode_route(expected_start, phases)
    if actual_route != expected_route:
        return f"invalid route {actual_route} for {form}/{difficulty}; expected {expected_route}"
    if any(left == right for left, right in zip(modes, modes[1:])):
        return "adjacent working phases must alternate modes"
    if any(len(phase.get("components") or []) != 1 for phase in phases):
        return "every phase must contain exactly one component"
    if not ids or None in ids or len(ids) != len(components):
        return "component ids are missing or duplicated"
    if consumed - produced:
        return f"unbound dependency values: {sorted(consumed - produced)}"
    sites = row.get("sites") or []
    expected_site_count = SITE_COUNT_PATTERNS[difficulty][form]
    if len(sites) != expected_site_count:
        return f"site count {len(sites)} does not match {difficulty}"
    site_ids = {site.get("site_id") for site in sites}
    domains = {normalize_site_domain(site.get("domain")) for site in sites}
    if None in site_ids or len(site_ids) != len(sites):
        return "site ids are missing or duplicated"
    if "" in domains or len(domains) != len(sites):
        return "target site domains are missing or duplicated"
    referenced_sites = {
        site_id for component in components for site_id in component.get("site_scope") or []
    }
    if referenced_sites != site_ids:
        return f"component site scopes do not cover target sites: {sorted(site_ids - referenced_sites)}"
    if any(
        component.get("mode") == "O"
        and (
            component.get("site_id") not in site_ids
            or component.get("site_scope") != [component.get("site_id")]
        )
        for component in components
    ):
        return "every O component must bind exactly one declared site"
    edges = solution.get("dependency_edges") or []
    edge_pairs = {(edge.get("from"), edge.get("to")) for edge in edges}
    expected_edge_pairs = {
        (source, component.get("component_id"))
        for component in components
        for source in component.get("consumes") or []
    }
    if edge_pairs != expected_edge_pairs:
        return "dependency edges do not exactly cover component inputs"
    if any(edge.get("from") not in produced or edge.get("to") not in ids for edge in edges):
        return "dependency edge references an unknown component or output"
    if len(expected_route) > 2 and not edges:
        return "multi-switch task has no causal dependency"
    if form == "F2":
        first = phases[0]
        if not first.get("attempt") or len(first.get("components") or []) != 1:
            return "F2 must have one explicit S attempt phase"
        if not any(edge.get("relation") == "fallback_gate" for edge in edges):
            return "F2 search attempt does not gate the first O phase"
    policies = solution.get("switch_policies") or []
    if len(policies) != len(expected_route) - 1:
        return "switch policies do not cover every mode boundary"
    if [policy.get("policy_id") for policy in policies] != [
        f"sw{index}" for index in range(1, len(policies) + 1)
    ]:
        return "switch policy ids are missing or unordered"
    policy_modes = [(policy.get("from"), policy.get("to")) for policy in policies]
    if policy_modes != list(zip(expected_route, expected_route[1:])):
        return "switch policy modes do not match the route"
    contract = row.get("difficulty_contract") or {}
    expected_contract = {
        "site_count": len(sites),
        "phase_count": len(phases),
        "switch_count": len(policies),
        "mode_route": expected_route,
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        return "difficulty contract does not match the actual task structure"
    if form in {"F3", "F4"} and difficulty in {"medium", "hard"} and not contract.get("cross_site_required"):
        return "medium and hard tasks must require cross-site work"
    search_work = [
        component
        for component in components
        if component.get("mode") == "S" and component.get("component_id") != "s_attempt"
    ]
    if difficulty == "hard" and not any(
        component.get("depth") == "deep" for component in search_work
    ):
        return "hard task has no deep S component"
    return ""


def generate_blueprints(
    count: int,
    catalog: dict,
    sites: list[dict],
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    operations = catalog.get("operation_prototypes") or []
    searches = catalog.get("search_prototypes") or []
    difficulty_profiles = catalog.get("difficulty_profiles") or {}
    eligible_sites = [site for site in sites if compatible_operations(site, operations)]
    if not eligible_sites:
        raise ValueError("no site has a compatible operation prototype")
    form_counts = allocate(count, FORM_RATIOS)
    schedule: list[tuple[str, str]] = []
    for form, form_count in form_counts.items():
        for difficulty, difficulty_count in allocate(
            form_count, FORM_DIFFICULTY_RATIOS[form]
        ).items():
            schedule.extend([(form, difficulty)] * difficulty_count)
    rng.shuffle(schedule)

    by_operation: dict[str, list[dict]] = {}
    for operation in operations:
        by_operation[operation["id"]] = [
            site
            for site in eligible_sites
            if operation["id"]
            in {row["id"] for row in compatible_operations(site, operations)}
        ]
    blueprints = []
    seen: set[tuple] = set()
    cursor = 0
    attempts = 0
    while len(blueprints) < len(schedule):
        attempts += 1
        if attempts > count * 30:
            raise RuntimeError(f"could only create {len(blueprints)} unique blueprints")
        form, difficulty = schedule[cursor]
        profile = difficulty_profiles[difficulty]
        site_count = int(profile["site_count_by_form"][form])
        eligible_operations = [
            operation
            for operation in operations
            if len(by_operation[operation["id"]]) >= site_count
        ]
        if not eligible_operations:
            raise ValueError(f"no operation prototype supports {site_count} distinct sites")
        # 按"该原型下站点的总权重"选原型,而不是均匀选。
        # 站点权重只在组内生效是不够的:原型均匀抽的话,常用站若集中在少数几个原型下,
        # 整体占比几乎抬不起来 —— 实测权重占 31% 的常用站,最后只拿到 13% 的出场。
        _w = [sum(max(0.0, float(r.get("weight", 1) or 1))
                  for r in by_operation[op["id"]]) for op in eligible_operations]
        _tot = sum(_w)
        if _tot <= 0:
            operation = rng.choice(eligible_operations)
        else:
            _x = rng.random() * _tot
            _acc = 0.0
            operation = eligible_operations[-1]
            for op, w in zip(eligible_operations, _w):
                _acc += w
                if _x <= _acc:
                    operation = op
                    break
        target_sites = weighted_sample_sites(
            by_operation[operation["id"]], site_count, rng)
        signature = (
            form,
            difficulty,
            operation["id"],
            *(site["domain"] for site in target_sites),
            attempts % 17,
        )
        if signature in seen:
            continue
        seen.add(signature)
        selected_domains = {site["domain"] for site in target_sites}
        remaining_sites = [
            candidate
            for candidate in by_operation[operation["id"]]
            if candidate["domain"] not in selected_domains
        ]
        alternatives = [
            {
                "domain": candidate["domain"],
                "entry_url": candidate["entry_url"],
                "category": candidate["category"],
            }
            for candidate in rng.sample(remaining_sites, min(2, len(remaining_sites)))
        ]
        row = build_blueprint(
            len(blueprints),
            form,
            difficulty,
            target_sites,
            operation,
            searches,
            profile,
            alternatives,
            rng,
        )
        error = validate_blueprint(row)
        if error:
            raise AssertionError(error)
        blueprints.append(row)
        cursor += 1
    return blueprints


NATURALIZE_SYSTEM = """You design realistic hybrid web-agent tasks. You receive typed blueprints; the
sites, phase sequence, opening switch form, and causal graph are fixed, while the entity scenario
and wording are adjustable.
Return strict JSON only. Do not claim that any answer has been verified. The task question must
sound like a real user request, not a benchmark instruction: never mention S/O, modes, paths,
switching, tools, evidence, or feasibility testing.

F1-F4 define the opening switch motif, not the whole route. F1 immediately corrects from an
unnecessary O start to S before browser work. F2 starts with a good-faith exact search and moves
to O only after the defined insufficiency verdict. F3 requires an S output to bind an O input.
F4 requires an O output to bind the subsequent S question. Medium and hard tasks continue by
alternating modes across two or three fixed sites. Every consumer must genuinely require its
declared producer value; a later site must not be an unrelated parallel lookup.

For F3, do not reveal or assume the S-produced answer in the question. State the clues or rule
needed to discover it, then ask to use whatever value is found in the live operation. A named
entity's obvious URL, a threshold already stated numerically, or a location already given verbatim
does not create a real dependency. For F4, do not pre-name the exact item that O is supposed to
surface. For later switches, preserve the exact prior result and explain why it changes the next
search question or site controls.

Difficulty comes from site count, switch count, chain depth, and dependency joins, not from
parallel same-mode padding. Each phase has one component. Use every declared target site exactly
as bound and do not add another site operation. All phases must serve one user goal. You may
replace entities, dates, locations, and constraints with natural compatible ones. Do not use exact
past dates for a live task unless the scenario is explicitly historical. Express every forward-looking
date RELATIVELY ("about three weeks from now", "next month") — never as an absolute calendar date."""


def naturalize_prompt(rows: list[dict]) -> str:
    compact = []
    for row in rows:
        site_domains = {
            site["site_id"]: site["domain"] for site in row.get("sites") or []
        }
        phases = []
        for phase in row["solution"]["phases"]:
            phase_copy = {key: value for key, value in phase.items() if key != "components"}
            phase_copy["components"] = []
            for component_row in phase["components"]:
                component_copy = dict(component_row)
                component_copy["site_scope_domains"] = [
                    site_domains.get(site_id, site_id)
                    for site_id in component_row.get("site_scope") or []
                ]
                phase_copy["components"].append(component_copy)
            phases.append(phase_copy)
        compact.append(
            {
                "blueprint_id": row["blueprint_id"],
                "switch_form": row["switch_form"],
                "difficulty": row["difficulty"],
                "sites": row["sites"],
                "difficulty_contract": row["difficulty_contract"],
                "entity_cardinality": row["entity_cardinality"],
                "operation_prototype": (
                    None if row["switch_form"] == "F1" else row["operation_prototype"]
                ),
                "phases": phases,
                "switch_policies": row["solution"]["switch_policies"],
                "dependency_edges": row["solution"]["dependency_edges"],
            }
        )
    schema = {
        "tasks": [
            {
                "blueprint_id": "must match an input id",
                "question": "natural task question",
                "site_domains": {"site_1": "exact fixed domain for that site id"},
                "entities": {"typed_slot": "concrete value or constraint"},
                "component_goals": {"component_id": "concrete subgoal in this scenario"},
                "component_done_when": {"component_id": "observable completion condition"},
                "switch_triggers": {"switch_policy_id": "scenario-specific fixed trigger"},
                "dependency_bindings": [
                    {
                        "from": "exact producer variable from dependency_edges",
                        "to": "exact consumer component id from dependency_edges",
                        "as": "natural name and role of this value in the consumer",
                    }
                ],
                "answer_schema": ["fields the eventual answer must contain"],
                "coherence_rationale": "one sentence explaining the single user goal and causal link",
            }
        ]
    }
    return (
        f"CURRENT DATE: {date.today().isoformat()}\n"
        # 绝对未来日期是 bench 报废的头号原因:实测 F3/F4 里 21.7% 被判 date_blocking
        # ——"a flight departing December 12, 2026" 过了那天任务就没意义。
        # 改成相对日期后同一道题任何时候跑都成立,而且约束的"形状"(星期几、提前量、
        # 时长)保留下来,查询难度不变。
        "DATES MUST BE RELATIVE, never absolute. Write 'a Saturday about three "
        "weeks from now', 'a weeknight next month', 'a week-long trip starting "
        "roughly a month out' — never 'October 12, 2026' or 'Dec 3-10, 2026'. "
        "Keep the shape of the constraint (day-of-week, lead time, duration) so "
        "the lookup stays the same kind of lookup. A date that refers to a fixed "
        "PAST event ('the 2024 final', 'articles from 2023') may stay absolute.\n\n"
        "INPUT BLUEPRINTS:\n"
        + json.dumps(compact, ensure_ascii=False)
        + "\n\nOUTPUT SCHEMA:\n"
        + json.dumps(schema, ensure_ascii=False)
        + "\nComponent goals and done_when may name only their declared site_scope_domains "
        "plus sites belonging to directly consumed producer outputs. Do not mention unrelated "
        "sites in a component-local description. Return exactly one task per blueprint, "
        "preserving every blueprint_id."
    )


def validate_naturalization(blueprint: dict, generated: dict) -> str:
    question = re.sub(r"\s+", " ", str(generated.get("question") or "")).strip()
    if len(question) < 35:
        return "question is missing or too short"
    if META_TASK_RE.search(question):
        return "question exposes agent routing or switch mechanics"
    expected_sites = {
        site["site_id"]: normalize_site_domain(site["domain"])
        for site in blueprint.get("sites") or []
    }
    generated_sites = generated.get("site_domains")
    if not isinstance(generated_sites, dict):
        return "site_domains are missing"
    normalized_sites = {
        str(site_id): normalize_site_domain(domain)
        for site_id, domain in generated_sites.items()
    }
    if normalized_sites != expected_sites:
        return f"site_domains must exactly preserve the fixed bindings {expected_sites}"
    components = [
        component
        for phase in blueprint["solution"]["phases"]
        for component in phase["components"]
    ]
    component_ids = {component["component_id"] for component in components}
    goals = generated.get("component_goals")
    done = generated.get("component_done_when")
    if not isinstance(goals, dict) or set(goals) != component_ids:
        return "component_goals do not cover the blueprint components exactly"
    if not isinstance(done, dict) or set(done) != component_ids:
        return "component_done_when does not cover the blueprint components exactly"
    if not all(str(value).strip() for value in goals.values()):
        return "one or more component goals are empty"
    site_domains = {
        site["site_id"]: normalize_site_domain(site["domain"])
        for site in blueprint.get("sites") or []
    }
    site_tokens = {
        site_id: re.sub(r"[^a-z0-9]", "", domain.split("/", 1)[0].split(".", 1)[0])
        for site_id, domain in site_domains.items()
    }
    component_by_id = {component["component_id"]: component for component in components}
    dependency_edges = blueprint["solution"].get("dependency_edges") or []
    for component in components:
        allowed = set(component.get("site_scope") or [])
        # A phase may name the site that produced an input it directly consumes.
        # This is necessary for legitimate cross-site handoffs (and for the hard
        # accumulated comparison join), while still preventing unrelated site
        # leakage into a component's goal or completion condition.
        for edge in dependency_edges:
            if edge.get("to") != component["component_id"]:
                continue
            producer_id = str(edge.get("from") or "").split(".", 1)[0]
            producer = component_by_id.get(producer_id)
            if producer:
                allowed.update(producer.get("site_scope") or [])
        text = re.sub(
            r"[^a-z0-9]", "", f"{goals[component['component_id']]} {done[component['component_id']]}".lower()
        )
        for site_id, token in site_tokens.items():
            if len(token) >= 5 and site_id not in allowed and token in text:
                return (
                    f"component {component['component_id']} mentions site {site_id} "
                    "outside its declared site_scope"
                )
    if blueprint["difficulty"] in {"medium", "hard"} and blueprint["switch_form"] in {"F3", "F4"}:
        question_text = re.sub(r"[^a-z0-9]", "", question.lower())
        missing = [
            site_id
            for site_id, token in site_tokens.items()
            if len(token) >= 5 and token not in question_text
        ]
        if missing:
            return f"question does not explicitly name target sites: {missing}"
    policies = blueprint["solution"].get("switch_policies") or []
    expected_policy_ids = {policy["policy_id"] for policy in policies}
    triggers = generated.get("switch_triggers")
    if not isinstance(triggers, dict) or set(triggers) != expected_policy_ids:
        return "switch_triggers do not cover every switch policy exactly"
    if not all(str(value).strip() for value in triggers.values()):
        return "one or more switch triggers are empty"
    if not generated.get("coherence_rationale"):
        return "coherence rationale is missing"
    if not isinstance(generated.get("answer_schema"), list) or not generated["answer_schema"]:
        return "answer schema is missing"
    expected_edges = {
        (edge["from"], edge["to"])
        for edge in blueprint["solution"]["dependency_edges"]
    }
    bindings = generated.get("dependency_bindings")
    if not isinstance(bindings, list):
        return "dependency_bindings are missing"
    actual_edges = {
        (binding.get("from"), binding.get("to"))
        for binding in bindings
        if isinstance(binding, dict)
    }
    if actual_edges != expected_edges:
        return "dependency_bindings do not exactly cover the causal edges"
    if any(not str(binding.get("as") or "").strip() for binding in bindings):
        return "one or more dependency bindings lack a natural value role"
    return ""


def merge_task(blueprint: dict, generated: dict, model: str) -> dict:
    row = json.loads(json.dumps(blueprint))
    goals = generated["component_goals"]
    done = generated["component_done_when"]
    for phase in row["solution"]["phases"]:
        for component_row in phase["components"]:
            component_id = component_row["component_id"]
            component_row["goal"] = re.sub(r"\s+", " ", str(goals[component_id])).strip()
            component_row["done_when"] = re.sub(r"\s+", " ", str(done[component_id])).strip()
    triggers = generated["switch_triggers"]
    for policy in row["solution"]["switch_policies"]:
        policy["trigger"] = re.sub(
            r"\s+", " ", str(triggers[policy["policy_id"]])
        ).strip()
    row["solution"]["switch_policy"] = row["solution"]["switch_policies"][0]
    roles = {
        (binding["from"], binding["to"]): re.sub(
            r"\s+", " ", str(binding["as"])
        ).strip()
        for binding in generated["dependency_bindings"]
    }
    for edge in row["solution"]["dependency_edges"]:
        edge["binding_role"] = roles[(edge["from"], edge["to"])]
    row.update(
        {
            "task_id": stable_id("hybrid", row["blueprint_id"], generated["question"]),
            "question": re.sub(r"\s+", " ", generated["question"]).strip(),
            "entities": generated.get("entities") or {},
            "answer_schema": generated["answer_schema"],
            "coherence_rationale": re.sub(
                r"\s+", " ", str(generated["coherence_rationale"])
            ).strip(),
            "synth_model": model,
        }
    )
    row.pop("draft_question", None)
    return row


# kimi-k3 只接受**离散的几个** temperature 值(实测 {0, 0.6, 1.0},其余一律 400
# "Parameter 'temperature'=X is not supported"),而 qwen 任意值都行。naturalizer
# 原本写死 0.55,换 kimi 当合成模型时整批都会被拒且看不出原因 —— 报错被吞进
# run_batch 的 reason 里,表现为"通过率 0"。这里按模型把温度收敛到最近的合法值。
_DISCRETE_TEMPS = {"kimi": (0.0, 0.6, 1.0)}


def snap_temperature(model: str, want: float) -> float:
    """把目标温度收敛到该模型实际支持的取值。不认识的模型原样返回。"""
    for prefix, allowed in _DISCRETE_TEMPS.items():
        if (model or "").lower().startswith(prefix):
            return min(allowed, key=lambda t: abs(t - want))
    return want


class Naturalizer:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float,
        attempts: int,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
        )
        self.model = model
        self.attempts = max(1, attempts)

    def run_batch(self, blueprints: list[dict]) -> tuple[list[dict], list[dict]]:
        pending = {row["blueprint_id"]: row for row in blueprints}
        tasks: dict[str, dict] = {}
        last_reasons: dict[str, str] = {}
        for attempt in range(1, self.attempts + 1):
            prompt = naturalize_prompt(list(pending.values()))
            if last_reasons:
                prompt += (
                    "\n\nRETRY FEEDBACK FOR THESE BLUEPRINTS:\n"
                    + json.dumps(last_reasons, ensure_ascii=False)
                    + "\nCorrect every listed issue while preserving the blueprint structure."
                )
            started = time.monotonic()
            print(
                f"[naturalize] start size={len(pending)} attempt={attempt}/{self.attempts}",
                file=sys.stderr,
                flush=True,
            )
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": NATURALIZE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=snap_temperature(self.model, 0.55),
                    max_tokens=max(4000, 1400 * len(pending)),
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )
                output = extract_json(response.choices[0].message.content or "") or {}
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                last_reasons = {blueprint_id: reason for blueprint_id in pending}
                print(
                    f"[naturalize] error elapsed={time.monotonic() - started:.1f}s {reason}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            generated_rows = output.get("tasks") if isinstance(output, dict) else None
            if not isinstance(generated_rows, list):
                last_reasons = {
                    blueprint_id: "response has no tasks list" for blueprint_id in pending
                }
                continue
            by_id = {
                row.get("blueprint_id"): row
                for row in generated_rows
                if isinstance(row, dict) and row.get("blueprint_id") in pending
            }
            retry: dict[str, dict] = {}
            retry_reasons: dict[str, str] = {}
            accepted_now = 0
            for blueprint_id, blueprint in pending.items():
                generated = by_id.get(blueprint_id)
                if not generated:
                    retry[blueprint_id] = blueprint
                    retry_reasons[blueprint_id] = "missing from model batch"
                    continue
                error = validate_naturalization(blueprint, generated)
                if error:
                    retry[blueprint_id] = blueprint
                    retry_reasons[blueprint_id] = error
                    continue
                task = merge_task(blueprint, generated, self.model)
                task["naturalize_attempt"] = attempt
                tasks[blueprint_id] = task
                accepted_now += 1
            print(
                f"[naturalize] complete elapsed={time.monotonic() - started:.1f}s "
                f"accepted={accepted_now} retry={len(retry)}",
                file=sys.stderr,
                flush=True,
            )
            pending = retry
            last_reasons = retry_reasons
            if not pending:
                break
        return list(tasks.values()), [
            {
                "blueprint_id": blueprint_id,
                "reason": last_reasons.get(blueprint_id, "model failed without a reason"),
            }
            for blueprint_id in pending
        ]


def naturalize_all(
    blueprints: list[dict],
    naturalizer: Naturalizer,
    *,
    batch_size: int,
    workers: int,
    checkpoint_path: Path | None = None,
    initial_tasks: list[dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    batches = [blueprints[index : index + batch_size] for index in range(0, len(blueprints), batch_size)]
    tasks: list[dict] = list(initial_tasks or [])
    rejects: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(naturalizer.run_batch, batch): batch for batch in batches
        }
        completed = 0
        for future in as_completed(futures):
            try:
                batch_tasks, batch_rejects = future.result()
            except Exception as exc:
                batch_tasks, batch_rejects = [], [
                    {
                        "blueprint_id": row["blueprint_id"],
                        "reason": f"unhandled batch error: {type(exc).__name__}: {exc}",
                    }
                    for row in futures[future]
                ]
            tasks.extend(batch_tasks)
            rejects.extend(batch_rejects)
            completed += 1
            if checkpoint_path and (completed % 10 == 0 or completed == len(futures)):
                write_jsonl(
                    checkpoint_path,
                    sorted(tasks, key=lambda row: row["blueprint_id"]),
                )
    tasks.sort(key=lambda row: row["blueprint_id"])
    rejects.sort(key=lambda row: str(row.get("blueprint_id")))
    return tasks, rejects


def balanced_sample(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Sample across switch-form/difficulty cells before using spare capacity."""
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("switch_form")), str(row.get("difficulty")))
        cells.setdefault(key, []).append(row)
    for cell_rows in cells.values():
        rng.shuffle(cell_rows)
    keys = sorted(cells)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < limit and keys:
        key = keys[cursor % len(keys)]
        if cells[key]:
            selected.append(cells[key].pop())
        if not cells[key]:
            keys.remove(key)
            if not keys:
                break
            cursor %= len(keys)
        else:
            cursor += 1
    return selected


def deterministic_task(blueprint: dict) -> dict:
    row = json.loads(json.dumps(blueprint))
    row.update(
        {
            "task_id": stable_id("hybrid", row["blueprint_id"], row["draft_question"]),
            "question": row.pop("draft_question"),
            "entities": {},
            "answer_schema": row.get("expected_answer_fields") or [],
            "coherence_rationale": "Deterministic blueprint; pending language-model naturalization.",
            "synth_model": "deterministic-blueprint",
        }
    )
    return row


def summarize(rows: list[dict]) -> dict:
    forms: dict[str, int] = {}
    difficulties: dict[str, int] = {}
    pairs: dict[str, int] = {}
    routes: dict[str, int] = {}
    site_counts: dict[str, int] = {}
    switch_counts: dict[str, int] = {}
    for row in rows:
        form = row.get("switch_form") or "unknown"
        difficulty = row.get("difficulty") or "unknown"
        forms[form] = forms.get(form, 0) + 1
        difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        key = f"{form}:{difficulty}"
        pairs[key] = pairs.get(key, 0) + 1
        contract = row.get("difficulty_contract") or {}
        route = "->".join(contract.get("mode_route") or []) or "unknown"
        routes[route] = routes.get(route, 0) + 1
        site_count = str(contract.get("site_count") or 0)
        site_counts[site_count] = site_counts.get(site_count, 0) + 1
        switch_count = str(contract.get("switch_count") or 0)
        switch_counts[switch_count] = switch_counts.get(switch_count, 0) + 1
    return {
        "total": len(rows),
        "forms": forms,
        "difficulties": difficulties,
        "matrix": pairs,
        "routes": routes,
        "site_counts": site_counts,
        "switch_counts": switch_counts,
    }


def audit_tasks(rows: list[dict], rejects: list[dict]) -> dict:
    summary = summarize(rows)
    normalized_questions = {
        re.sub(r"\s+", " ", str(row.get("question") or "")).strip().lower()
        for row in rows
    }
    domains = {
        str(site.get("domain") or "")
        for row in rows
        for site in (row.get("sites") or [row.get("site") or {}])
    }
    operation_ids = {
        str((row.get("operation_prototype") or {}).get("id") or "") for row in rows
    }
    search_ids = {
        str(component.get("prototype_id") or "")
        for row in rows
        for phase in (row.get("solution") or {}).get("phases") or []
        for component in phase.get("components") or []
        if component.get("mode") == "S" and component.get("component_id") != "s_attempt"
    }
    structural_errors = [validate_blueprint(row) for row in rows]
    dual_rows = [
        row for row in rows if (row.get("solution") or {}).get("dependency_edges")
    ]
    hard_search_rows = [
        row
        for row in rows
        if row.get("difficulty") == "hard"
    ]
    summary.update(
        {
            "rejects": len(rejects),
            "acceptance_rate": len(rows) / max(1, len(rows) + len(rejects)),
            "unique_questions": len(normalized_questions - {""}),
            "duplicate_questions": len(rows) - len(normalized_questions - {""}),
            "unique_domains": len(domains - {""}),
            "operation_prototypes_used": len(operation_ids - {""}),
            "search_prototypes_used": len(search_ids - {""}),
            "structural_errors": sum(bool(error) for error in structural_errors),
            "dual_tasks_with_bound_edges": sum(
                bool((row.get("solution") or {}).get("dependency_edges"))
                and all(
                    edge.get("binding_role")
                    for edge in (row.get("solution") or {}).get("dependency_edges") or []
                )
                for row in dual_rows
            ),
            "dual_tasks_total": len(dual_rows),
            "hard_search_tasks_with_deep_s": sum(
                any(
                    component.get("mode") == "S" and component.get("depth") == "deep"
                    for phase in (row.get("solution") or {}).get("phases") or []
                    for component in phase.get("components") or []
                )
                for row in hard_search_rows
            ),
            "hard_search_tasks_total": len(hard_search_rows),
            "quality_flagged_tasks": sum(bool(task_quality_flags(row)) for row in rows),
            "potential_past_live_date_tasks": sum(
                "potential_past_live_date" in task_quality_flags(row) for row in rows
            ),
            "site_operation_mismatch_tasks": sum(
                "site_operation_mismatch" in task_quality_flags(row) for row in rows
            ),
            "cross_site_tasks": sum(
                len(row.get("sites") or []) > 1 for row in rows
            ),
            "multi_switch_tasks": sum(
                int((row.get("difficulty_contract") or {}).get("switch_count") or 0) > 1
                for row in rows
            ),
            "single_component_phase_tasks": sum(
                all(
                    len(phase.get("components") or []) == 1
                    for phase in (row.get("solution") or {}).get("phases") or []
                )
                for row in rows
            ),
        }
    )
    return summary


MONTH_NUMBERS = {
    name.lower(): number
    for number, name in enumerate(
        (
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ),
        1,
    )
}
MONTH_YEAR_RE = re.compile(
    r"\b(" + "|".join(MONTH_NUMBERS) + r")\s+(?:\d{1,2}(?:\s*[-–]\s*\d{1,2})?,?\s+)?(20\d{2})\b",
    re.I,
)
FUTURE_INTENT_RE = re.compile(
    r"\b(?:plan(?:ning)?|book(?:ing)?|trip|travel(?:ing)?|visit(?:ing)?|depart(?:ing)?|"
    r"return(?:ing)?|stay|upcoming|available|availability|install(?:ed|ation)?|"
    r"conference|fundraiser)\b",
    re.I,
)


def task_quality_flags(row: dict, today: date | None = None) -> list[str]:
    today = today or date.today()
    question = str(row.get("question") or "")
    flags: list[str] = []
    if FUTURE_INTENT_RE.search(question):
        for match in MONTH_YEAR_RE.finditer(question):
            month = MONTH_NUMBERS[match.group(1).lower()]
            year = int(match.group(2))
            if (year, month) < (today.year, today.month):
                flags.append("potential_past_live_date")
                break
    actual_operation = (row.get("operation_prototype") or {}).get("id")
    site_by_id = {
        site.get("site_id"): site for site in row.get("sites") or []
    }
    operation_sites = {
        component.get("site_id")
        for phase in (row.get("solution") or {}).get("phases") or []
        for component in phase.get("components") or []
        if component.get("mode") == "O"
    }
    if not site_by_id:
        site_by_id = {None: row.get("site") or {}}
        operation_sites = {None}
    for site_id in operation_sites:
        domain = normalize_site_domain((site_by_id.get(site_id) or {}).get("domain"))
        expected_operation = DOMAIN_OPERATION_OVERRIDES.get(domain)
        if expected_operation and actual_operation != expected_operation:
            flags.append("site_operation_mismatch")
            break
    return flags


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("catalog", "generate", "naturalize", "all"), default="all")
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--site-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--site-detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-naturalize", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--jina-pool", "--embedding-base-url", dest="jina_pool",
                        default=os.getenv("JINA_EMBEDDING_BASE_URL", "https://api.jina.ai/v1"))
    parser.add_argument("--embedding-model", default=os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3"))
    parser.add_argument("--text-base-url", default=os.getenv("SYNTH_BASE", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.getenv("SYNTH_API_KEY", "EMPTY"))
    parser.add_argument("--text-model", default=os.getenv("SYNTH_MODEL", "kimi-k3"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.out_dir / "design_catalog.json"
    seed = json.loads(args.seed_file.read_text(encoding="utf-8"))
    if args.mode in {"catalog", "all"} or not catalog_path.is_file():
        embedder = JinaPoolEmbedder(
            args.jina_pool,
            args.embedding_model,
            args.out_dir / "embedding_cache.jsonl",
        )
        catalog = build_catalog(seed, embedder, catalog_path)
        print(
            f"catalog: O={len(catalog['operation_prototypes'])} "
            f"S={len(catalog['search_prototypes'])} dim={catalog['embedding']['dimension']}",
            flush=True,
        )
    else:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog.get("schema_version") != seed.get("schema_version"):
            raise RuntimeError(
                f"catalog schema {catalog.get('schema_version')!r} does not match seed "
                f"schema {seed.get('schema_version')!r}; rebuild the catalog or use a new out-dir"
            )
    if args.mode == "catalog":
        return 0

    blueprint_path = args.out_dir / "task_blueprints.jsonl"
    if args.mode in {"generate", "all"} or not blueprint_path.is_file():
        sites = load_sites(args.site_pool, args.site_detail)
        blueprints = generate_blueprints(args.count, catalog, sites, args.seed)
        write_jsonl(blueprint_path, blueprints)
        summary = summarize(blueprints)
        (args.out_dir / "blueprint_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"generate: {len(blueprints)} blueprints {json.dumps(summary)}", flush=True)
    else:
        blueprints = read_jsonl(blueprint_path)
    if args.mode == "generate":
        return 0

    if args.max_naturalize:
        blueprints = balanced_sample(blueprints, args.max_naturalize, args.seed)
    task_path = args.out_dir / "tasks.jsonl"
    existing_tasks = read_jsonl(task_path) if args.resume else []
    if existing_tasks:
        selected_ids = {row["blueprint_id"] for row in blueprints}
        existing_tasks = [
            row for row in existing_tasks if row.get("blueprint_id") in selected_ids
        ]
        completed_ids = {row["blueprint_id"] for row in existing_tasks}
        blueprints = [
            row for row in blueprints if row["blueprint_id"] not in completed_ids
        ]
        print(
            f"resume: kept={len(existing_tasks)} pending={len(blueprints)}",
            flush=True,
        )
    if args.no_llm:
        tasks = [*existing_tasks, *(deterministic_task(row) for row in blueprints)]
        rejects: list[dict] = []
    else:
        naturalizer = Naturalizer(
            args.text_base_url,
            args.api_key,
            args.text_model,
            timeout=args.timeout,
            attempts=args.attempts,
        )
        tasks, rejects = naturalize_all(
            blueprints,
            naturalizer,
            batch_size=args.batch_size,
            workers=args.workers,
            checkpoint_path=task_path,
            initial_tasks=existing_tasks,
        )
    write_jsonl(task_path, tasks)
    write_jsonl(
        args.out_dir / "task_samples_balanced.jsonl",
        balanced_sample(tasks, min(24, len(tasks)), args.seed),
    )
    write_jsonl(args.out_dir / "naturalize_rejects.jsonl", rejects)
    write_jsonl(
        args.out_dir / "quality_flags.jsonl",
        (
            {
                "task_id": row.get("task_id"),
                "blueprint_id": row.get("blueprint_id"),
                "flags": flags,
                "question": row.get("question"),
            }
            for row in tasks
            if (flags := task_quality_flags(row))
        ),
    )
    task_summary = audit_tasks(tasks, rejects)
    (args.out_dir / "task_summary.json").write_text(
        json.dumps(task_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"naturalize: tasks={len(tasks)} rejects={len(rejects)} -> {task_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
