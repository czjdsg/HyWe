#!/usr/bin/env python3
"""Download the third-party benchmarks from their official sources and convert them
to the task JSONL read by inference/run_hybrid_web_agent.py.

These benchmarks are distributed encrypted or behind a click-through agreement to
keep them out of training corpora, so this repository ships only the converter,
not the plain-text tasks. Converted files are written under benchmarks/data/
(git-ignored); please do not re-publish them.

    python benchmarks/prepare_benchmarks.py browsecomp        # BrowseComp-EN, 1,266 tasks
    python benchmarks/prepare_benchmarks.py xbench            # xbench-DeepSearch-2510, 100 tasks
    python benchmarks/prepare_benchmarks.py hle               # HLE test, 2,500 tasks (+ images)
    python benchmarks/prepare_benchmarks.py online_mind2web   # Online-Mind2Web, 300 tasks

HLE and Online-Mind2Web are gated on Hugging Face: accept their terms on the dataset
page, then export HF_TOKEN. WebVoyager (public) and HyWeBench ship with the repo.

Record schema (one JSON object per line):
    task_id, question, answer (reference; "" for operation tasks), web (start URL,
    "" = start in search-based mode), dataset, split, record_id, domain, subdomain,
    extra, raw, and optionally input_images (local image paths).
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

BROWSECOMP_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
HF = os.environ.get("HF_ENDPOINT_OFFICIAL", "https://huggingface.co")


def _fetch(url: str, token: str = "") -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "hywe-prepare/1.0"})
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=300) as resp:
        return resp.read()


def _hf_file(repo: str, path: str, gated: bool = False) -> bytes:
    token = os.environ.get("HF_TOKEN", "")
    if gated and not token:
        raise SystemExit(f"{repo} is gated: accept its terms on {HF}/datasets/{repo} and export HF_TOKEN")
    return _fetch(f"{HF}/datasets/{repo}/resolve/main/{path}", token)


def _write(rows: list[dict], name: str) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / name
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} tasks -> {out}")
    return out


def _record(dataset, split, rid, question, answer="", web="", domain="", subdomain="", extra=None, raw=None):
    return {"question": question, "task_id": f"{dataset}-{split}-{rid}", "web": web,
            "dataset": dataset, "split": split, "record_id": str(rid), "answer": answer,
            "domain": domain, "subdomain": subdomain, "extra": extra or {}, "raw": raw or {}}


# --------------------------------------------------------------------- BrowseComp
def _bc_decrypt(ciphertext_b64: str, password: str) -> str:
    """XOR with a SHA-256-derived key stream (openai/simple-evals browsecomp_eval.py)."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = hashlib.sha256(password.encode()).digest()
    key = key * (len(encrypted) // len(key)) + key[: len(encrypted) % len(key)]
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def browsecomp(_args) -> None:
    text = _fetch(BROWSECOMP_URL).decode("utf-8")
    rows = []
    for i, row in enumerate(csv.DictReader(io.StringIO(text)), 1):
        canary = row["canary"]
        topic = row.get("problem_topic", "")
        rows.append(_record("BrowseComp", "test", i, _bc_decrypt(row["problem"], canary),
                            _bc_decrypt(row["answer"], canary), domain=topic,
                            extra={"topic": topic}, raw={"problem_topic": topic, "canary": canary}))
    _write(rows, "browsecomp_en_1266.jsonl")


# ------------------------------------------------------------ xbench-DeepSearch-2510
def _xbench_decrypt(b64_text: str, canary: str) -> str:
    """base64-decode, then XOR with the row's canary string (xbench release format)."""
    raw = base64.b64decode(b64_text)
    key = canary.encode("utf-8")
    return bytes(c ^ key[i % len(key)] for i, c in enumerate(raw)).decode("utf-8")


def xbench(_args) -> None:
    text = _hf_file("xbench/DeepSearch-2510", "DeepSearch-2510.csv").decode("utf-8")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        canary, rid = row["canary"], str(row["id"]).strip()
        steps = _xbench_decrypt(row["reference_steps"], canary) if row.get("reference_steps") else ""
        rows.append(_record("DeepSearch-2510", "test", rid, _xbench_decrypt(row["prompt"], canary),
                            _xbench_decrypt(row["answer"], canary),
                            extra={"reference_steps": steps}, raw={"id": rid, "canary": canary}))
    _write(rows, "xbench_deepsearch_2510_100.jsonl")


# --------------------------------------------------------------------------- HLE
def hle(_args) -> None:
    import pandas as pd  # pyarrow is required for parquet

    blob = _hf_file("cais/hle", "data/test-00000-of-00001.parquet", gated=True)
    df = pd.read_parquet(io.BytesIO(blob))
    img_dir = DATA / "hle_images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in df.to_dict("records"):
        rid = str(r["id"])
        images = []
        image = r.get("image") or ""
        if isinstance(image, str) and image.startswith("data:image/"):
            head, b64 = image.split(",", 1)
            ext = head.split("/")[1].split(";")[0].replace("jpeg", "jpg")
            path = img_dir / f"{rid}.{ext}"
            path.write_bytes(base64.b64decode(b64))
            images.append(str(path.resolve()))
        rec = _record("HLE", "test", rid, r["question"], str(r["answer"]),
                      domain=r.get("raw_subject") or "", subdomain=r.get("category") or "",
                      extra={"answer_type": r.get("answer_type"), "has_image": bool(images)},
                      raw={"id": rid})
        if images:
            rec["input_images"] = images
        rows.append(rec)
    _write(rows, "hle_test_2500.jsonl")


# --------------------------------------------------------------- Online-Mind2Web
def online_mind2web(_args) -> None:
    tasks = json.loads(_hf_file("osunlp/Online-Mind2Web", "Online_Mind2Web.json", gated=True))
    rows = []
    for t in tasks:
        tid = t["task_id"]
        rec = _record("Online-Mind2Web", "full", tid, t["confirmed_task"], "", web=t["website"],
                      subdomain=t.get("level", ""),
                      extra={"reference_length": t.get("reference_length"), "level": t.get("level")},
                      raw={"task_id": tid})
        rec["task_id"] = tid          # the official id is what WebJudge keys on
        rows.append(rec)
    _write(rows, "online_mind2web_300.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark", choices=["browsecomp", "xbench", "hle", "online_mind2web", "all"])
    a = ap.parse_args()
    fns = {"browsecomp": browsecomp, "xbench": xbench, "hle": hle, "online_mind2web": online_mind2web}
    for name in (fns if a.benchmark == "all" else [a.benchmark]):
        fns[name](a)


if __name__ == "__main__":
    main()
