#!/usr/bin/env python3
"""Offline integration check: scripted model, real browser, search and judges.

No learned model is used and the local page is a fixture. This checks integration
and coordinate transport, not real-world task-solving ability. All network
requests stay on loopback. Outputs are saved under runs/smoke/ by default.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parent.parent
STATE = {"policy_calls": [], "judge_calls": [], "search_calls": 0, "clicked": False}
PAGE = b'''<!doctype html><html><head><title>HyWe smoke fixture</title></head>
<body><button style="position:absolute;left:540px;top:310px;width:200px;height:100px"
onclick="document.title='SMOKE_VALUE_42';this.textContent='SMOKE_VALUE_42';fetch('/clicked')">
Reveal test value</button></body></html>'''


def text_content(value):
    if isinstance(value, str):
        return value
    return "\n".join(p.get("text", "") for p in value if isinstance(p, dict))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, data, content_type="application/json"):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/clicked":
            STATE["clicked"] = True
            return self.send({"ok": True})
        if self.path == "/page":
            return self.send(PAGE, "text/html")
        return self.send({"data": [{"id": "HyWe-35B-smoke"}]})

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/search":
            STATE["search_calls"] += 1
            return self.send({"organic": [{"title": "Local smoke evidence", "link": "https://example.invalid/fixture",
                                          "snippet": "The fixture test value is SMOKE_VALUE_42."}]})
        messages = request.get("messages", [])
        images = [p for m in messages if isinstance(m.get("content"), list)
                  for p in m["content"] if p.get("type") == "image_url"]
        system = text_content(messages[0]["content"]) if messages else ""
        model = request.get("model", "")
        if model == "HyWe-35B-smoke":
            index = len(STATE["policy_calls"])
            outputs = [
                '<think>Reveal the value on the local page.</think><tool_call>{"name":"click","arguments":{"x":500,"y":500}}</tool_call>',
                '<think>Now retrieve the supporting fixture fact.</think><tool_call>{"name":"switch","arguments":{}}</tool_call>',
                '<think>Search for the test value.</think><tool_call>{"name":"search","arguments":{"query":"HyWe smoke fixture value"}}</tool_call>',
                '<think>The fixture evidence is available.</think><answer>SMOKE_VALUE_42</answer>',
            ]
            if index >= len(outputs):
                raise RuntimeError("Unexpected extra policy call in scripted smoke test")
            output = outputs[index]
            STATE["policy_calls"].append({"images": len(images), "system": system[:100]})
        elif model == "smoke-checklist":
            output = json.dumps({"results": [
                {"id": 1, "done": True, "first_step": 1, "evidence": "Step 1 shows the revealed fixture value.", "correct": None, "blocked": False},
                {"id": 2, "done": True, "first_step": 4, "evidence": "Step 4 answer is SMOKE_VALUE_42.", "correct": True, "blocked": False}],
                "env_blocked": False, "blocked_where": "", "notes": "Scripted fixture judgment"})
            STATE["judge_calls"].append({"model": model, "images": len(images),
                "inline": all(p["image_url"]["url"].startswith("data:image/") for p in images)})
        elif model == "smoke-reference":
            output = json.dumps({"extracted_final_answer": "SMOKE_VALUE_42", "reasoning": "Scripted fixture vote", "predicted_label": 1})
            STATE["judge_calls"].append({"model": model})
        elif model == "o4-mini":
            if "extract the critical elements" in system:
                output = "**Key Points**:\n1. Reveal and report the fixture value."
            elif "assigning a score" in system or "Score**" in system:
                output = "**Reasoning**: The fixture evidence is relevant.\n**Score**: 5"
            else:
                output = "Status: success\nBlocked: no"
            STATE["judge_calls"].append({"model": model})
        elif model == "gpt-4o":
            output = "SUCCESS\nBlocked: no"
            STATE["judge_calls"].append({"model": model})
        else:
            raise RuntimeError("Unexpected smoke model: " + model)
        self.send({"id": "smoke", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": output}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/smoke")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory is not empty; use a new --out directory for an independent smoke test.")
    out.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_port}"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("HYBRID_", "PROBE_", "SERPER_", "JINA_"))
           and k.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    env.update({"NO_PROXY": "*", "no_proxy": "*", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false", "HYBRID_TOKENIZER_PATH": str(out / "no-tokenizer"),
                "TOKENIZER_PATH": str(out / "no-tokenizer"), "HYBRID_SEARCH_BACKEND": "serper",
                "SERPER_BASE_URL": base, "SERPER_API_KEY": "smoke", "TASK_PROXY_URL": "",
                "HYBRID_IMAGE_TRANSPORT": "base64", "HYBRID_V2_RENDER": "multiturn",
                "BASE_URL": base + "/v1", "MODEL": "HyWe-35B-smoke", "API_KEY": "smoke",
                "OPENAI_BASE_URL": base + "/v1", "OPENAI_API_KEY": "smoke", "PY": sys.executable,
                "MAX_ROUNDS": "1", "MAX_WORKERS": "1", "WORKERS": "4", "HJ_OP_SEARCH_RULE": "0"})
    task = {"task_id": "smoke-1", "question": "Reveal the local page value, search for supporting evidence, and report the value.",
            "web": base + "/page", "answer": "SMOKE_VALUE_42",
            "checklist": [{"id": 1, "goal": "The fixture page value is revealed", "kind": "operation"},
                          {"id": 2, "goal": "The reported value is SMOKE_VALUE_42", "kind": "information"}]}
    tasks = out / "tasks.jsonl"
    tasks.write_text(json.dumps(task) + "\n")
    env["TASKS"] = str(tasks)
    def run(label, command):
        with (out / (label + ".log")).open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if result.returncode:
            raise RuntimeError(f"{label} failed: {(out / (label + '.log')).read_text()[-5000:]}")
        print("PASS: " + label, flush=True)
    try:
        run("inference", ["bash", "scripts/run_hywe.sh", str(tasks), str(out / "run"), "operation", "--no-css-som", "--max-iter", "8"])
        result = json.loads((out / "run/smoke-1/result.json").read_text())
        assert result["termination"] == "answer" and result["prediction"] == "SMOKE_VALUE_42", result.get("termination")
        tools = [(r.get("tool_call") or {}).get("name") for r in result["records"]]
        assert tools[:3] == ["click", "switch", "search"], tools
        assert STATE["clicked"] and STATE["search_calls"] == 1, STATE
        assert STATE["policy_calls"][0]["images"] > 0
        env.update({"JUDGE_BASE_URL": base + "/v1", "JUDGE_API_KEY": "smoke", "JUDGE_MODEL": "smoke-checklist"})
        run("checklist", ["bash", "evaluation/hywebench/judge.sh", str(out / "run"), str(out / "checklist")])
        verdict = json.loads((out / "checklist/verdicts.jsonl").read_text().splitlines()[-1])
        assert verdict["ok"] and verdict["complete"], verdict
        assert any(c.get("images", 0) > 0 and c.get("inline") for c in STATE["judge_calls"])
        run("mind2web", ["bash", "evaluation/online_mind2web/judge.sh", str(out / "run"), str(out / "m2w")])
        run("webvoyager", ["bash", "evaluation/webvoyager/judge.sh", str(out / "run"), str(out / "wv/results.jsonl")])
        env["JUDGE_MODEL"] = "smoke-reference"
        run("reference", [sys.executable, "evaluation/search/evaluate_reference_answers.py", "--run-dir", str(out / "run"),
                          "--output-dir", str(out / "reference"), "--base-url", base + "/v1", "--model", "smoke-reference",
                          "--api-key", "smoke", "--votes", "5", "--no-exact-match", "--max-workers", "1"])
        assert sum(c.get("model") == "smoke-reference" for c in STATE["judge_calls"]) == 5
        models = {c["model"] for c in STATE["judge_calls"]}
        assert {"o4-mini", "gpt-4o"} <= models, models
        (out / "summary.json").write_text(json.dumps(STATE, indent=2))
        print(f"PASS: real browser click, O-to-S switch, search, inline screenshots and all four judge entry points. Artifacts: {out}")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
