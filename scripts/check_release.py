#!/usr/bin/env python3
"""Validate the portable release without network calls or API credentials."""
import ast
from collections import Counter
import importlib.util
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent


def main():
    ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "runs", "logs"}
    files = [p for p in ROOT.rglob("*") if p.is_file()
             and not any(part in ignored for part in p.relative_to(ROOT).parts)]
    for path in files:
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
        elif path.suffix == ".sh":
            subprocess.run(["bash", "-n", str(path)], check=True)
        if path.suffix in {".py", ".sh", ".md", ".example", ".txt"}:
            body = path.read_text()
            if re.search(r"(?:hf_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{25,}|"
                         r"ms-[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12})", body):
                raise RuntimeError(f"Credential-shaped value found in {path.relative_to(ROOT)}")
            local_user = "wan" + "cong"
            if re.search(r"/(?:data4|home)/" + local_user + r"|/cpfs\d*/", body):
                raise RuntimeError(f"Machine-specific path in {path.relative_to(ROOT)}")
    for name in ["README.md", "INSTALL.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
                 "configs/hywe.env.example", "configs/collector.env.example",
                 "collector/data/insta_screened.pool.json", "collector/hybrid_task_design.seed.json",
                 "training/training_config.json", "scripts/smoke_test.py"]:
        assert (ROOT / name).is_file(), f"Missing release file: {name}"
    tasks_path = ROOT / "benchmarks/hywebench/hywebench_100.jsonl"
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    ids = {t["task_id"] for t in tasks}
    assert len(tasks) == len(ids) == 100
    counts = Counter(s["kind"] for t in tasks for s in t["checklist"])
    assert counts == {"information": 309, "operation": 247}, counts
    for task in tasks:
        assert bool(task["web"]) == (task["start_mode"] == "O"), task["task_id"]
    wv = [json.loads(line) for line in (ROOT / "benchmarks/webvoyager/webvoyager_643.jsonl").read_text().splitlines()]
    assert len(wv) == len({t["task_id"] for t in wv}) == 643
    path = ROOT / "evaluation/hywebench/score.py"
    spec = importlib.util.spec_from_file_location("hywe_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.loads((path.parent / "paper_metrics.json").read_text())["metrics"]
    for name, values in expected.items():
        verdicts = path.parent / "paper_verdicts" / (name + ".jsonl")
        rows = [json.loads(line) for line in verdicts.read_text().splitlines() if line.strip()]
        assert {r["task_id"] for r in rows if r.get("ok")} == ids, name
        result = module.score(str(verdicts), tasks)
        actual = [round(result[key], 1) for key in ("success", "info", "op")]
        assert actual == values, (name, actual, values)
    print(f"PASS: Python/shell syntax; 100 hybrid tasks ({counts['information']} info, "
          f"{counts['operation']} operation items); 643 WebVoyager tasks; all 14 paper metrics; release files.")


if __name__ == "__main__":
    main()
