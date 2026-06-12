#!/usr/bin/env python3
"""Local SandboxFusion-compatible code-execution server (no Docker).

Drop-in replacement for the vemlp `code-sandbox:server` Docker container used
by `inference/hybrid_code_sandbox.py`. Speaks the same HTTP contract:

  POST /run_code   {language, code, run_timeout, compile_timeout, stdin?,
                    files?, fetch_files?}
  ->               {status, message, compile_result, run_result, files}

Supported languages (matched to whatever the host has installed; missing
toolchains return a 200 with status=Error and a clear message):
  python, python3, bash, nodejs, typescript, cpp, go

Designed for the Autodl-style host where Docker is unavailable. Each request
is executed in an isolated temp dir as a subprocess with `resource.setrlimit`
CPU + AS caps. There is NO chroot / cgroups / network namespace — this is a
*development* sandbox, not a multi-tenant production sandbox.

Defaults (override via env):
  SANDBOX_HOST=127.0.0.1
  SANDBOX_PORT=8080
  SANDBOX_CPU_SECONDS=30
  SANDBOX_MEM_BYTES=2147483648    (2 GB)
  SANDBOX_WORK_ROOT=/tmp/hwa_sandbox
"""
from __future__ import annotations

import argparse
import base64
import http.server
import io
import json
import os
import resource
import shutil
import socketserver
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
from pathlib import Path
from typing import Any


HOST = os.environ.get("SANDBOX_HOST", "127.0.0.1")
PORT = int(os.environ.get("SANDBOX_PORT", "8080"))
CPU_SECONDS = int(os.environ.get("SANDBOX_CPU_SECONDS", "30"))
MEM_BYTES = int(os.environ.get("SANDBOX_MEM_BYTES", str(2 * 1024 ** 3)))
WORK_ROOT = Path(os.environ.get("SANDBOX_WORK_ROOT", "/tmp/hwa_sandbox"))
WORK_ROOT.mkdir(parents=True, exist_ok=True)


# language -> (source filename, compile cmd or None, run cmd template)
# Each entry is a callable returning (compile_cmd_or_None, run_cmd, stdin_payload)
def _spec_python(workdir: Path, code: str) -> tuple[list[str] | None, list[str], str | None]:
    src = workdir / "main.py"
    src.write_text(code)
    return None, [sys.executable, str(src)], None


def _spec_bash(workdir: Path, code: str) -> tuple[list[str] | None, list[str], str | None]:
    src = workdir / "main.sh"
    src.write_text(code)
    return None, ["bash", str(src)], None


def _spec_nodejs(workdir: Path, code: str) -> tuple[list[str] | None, list[str], str | None]:
    src = workdir / "main.js"
    src.write_text(code)
    return None, ["node", str(src)], None


def _spec_typescript(workdir: Path, code: str) -> tuple[list[str] | None, list[str], str | None]:
    src = workdir / "main.ts"
    src.write_text(code)
    runner = shutil.which("ts-node") or shutil.which("tsx")
    if not runner:
        raise RuntimeError("typescript: install either `ts-node` or `tsx` "
                           "(`npm i -g ts-node typescript` or `npm i -g tsx`)")
    return None, [runner, str(src)], None


def _spec_cpp(workdir: Path, code: str) -> tuple[list[str] | None, list[str], str | None]:
    src = workdir / "main.cpp"
    bin_path = workdir / "a.out"
    src.write_text(code)
    cc = shutil.which("g++") or shutil.which("clang++")
    if not cc:
        raise RuntimeError("cpp: no g++ or clang++ on PATH")
    return [cc, str(src), "-O2", "-std=c++17", "-o", str(bin_path)], [str(bin_path)], None


def _spec_go(workdir: Path, code: str) -> tuple[list[str] | None, list[str], str | None]:
    src = workdir / "main.go"
    bin_path = workdir / "a.out"
    src.write_text(code)
    go = shutil.which("go")
    if not go:
        raise RuntimeError("go: no `go` toolchain on PATH")
    return [go, "build", "-o", str(bin_path), str(src)], [str(bin_path)], None


LANG_SPECS = {
    "python": _spec_python,
    "python3": _spec_python,
    "python_gpu": _spec_python,
    "bash": _spec_bash,
    "nodejs": _spec_nodejs,
    "typescript": _spec_typescript,
    "cpp": _spec_cpp,
    "go": _spec_go,
}


def _apply_rlimit() -> None:
    """Run inside the subprocess via preexec_fn — cap CPU and memory."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECONDS, CPU_SECONDS))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_BYTES, MEM_BYTES))
    except (ValueError, OSError):
        pass


def _run_one(cmd: list[str], *, cwd: Path, timeout: float,
             stdin: str | None) -> dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            input=stdin if stdin is not None else None,
            capture_output=True,
            text=True,
            timeout=max(0.1, float(timeout or 0.1)),
            preexec_fn=_apply_rlimit,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "TimeLimitExceeded",
            "execution_time": time.time() - t0,
            "return_code": -1,
            "stdout": (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, (bytes, bytearray)) else (exc.stdout or ""),
            "stderr": (exc.stderr or b"").decode("utf-8", "replace") if isinstance(exc.stderr, (bytes, bytearray)) else (exc.stderr or ""),
        }
    except FileNotFoundError as exc:
        return {
            "status": "Error",
            "execution_time": time.time() - t0,
            "return_code": -1,
            "stdout": "",
            "stderr": f"executable not found: {exc}",
        }
    except Exception as exc:
        return {
            "status": "Error",
            "execution_time": time.time() - t0,
            "return_code": -1,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
        }
    status = "Finished" if proc.returncode == 0 else "Error"
    return {
        "status": status,
        "execution_time": time.time() - t0,
        "return_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _materialize_files(workdir: Path, files: dict[str, Any] | None) -> None:
    if not files:
        return
    for rel_path, content in files.items():
        if content is None:
            continue
        dest = (workdir / rel_path).resolve()
        # Refuse escapes out of workdir.
        if workdir.resolve() not in dest.parents and dest != workdir.resolve():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            try:
                # SandboxFusion encodes binary as base64; treat values that
                # look like base64 as binary and fall back to text otherwise.
                blob = base64.b64decode(content, validate=True)
                dest.write_bytes(blob)
                continue
            except Exception:
                pass
            dest.write_text(content)
        elif isinstance(content, (bytes, bytearray)):
            dest.write_bytes(content)


def _harvest_fetch_files(workdir: Path, fetch_files: list[str] | None
                        ) -> dict[str, str]:
    if not fetch_files:
        return {}
    out: dict[str, str] = {}
    for rel in fetch_files:
        candidate = (workdir / rel).resolve()
        if workdir.resolve() not in candidate.parents and candidate != workdir.resolve():
            continue
        try:
            raw = candidate.read_bytes()
        except FileNotFoundError:
            continue
        try:
            out[rel] = raw.decode("utf-8")
        except UnicodeDecodeError:
            out[rel] = base64.b64encode(raw).decode("ascii")
    return out


def handle_run_code(body: dict[str, Any]) -> dict[str, Any]:
    language = (body.get("language") or "python").strip()
    code = body.get("code") or ""
    run_timeout = float(body.get("run_timeout") or 10)
    compile_timeout = float(body.get("compile_timeout") or 10)
    stdin = body.get("stdin")
    files = body.get("files") or {}
    fetch_files = body.get("fetch_files") or []

    spec_fn = LANG_SPECS.get(language)
    if spec_fn is None:
        return {
            "status": "Error",
            "message": f"language `{language}` not supported by local sandbox",
            "compile_result": None,
            "run_result": None,
            "executor_pod_name": None,
            "files": {},
        }

    with tempfile.TemporaryDirectory(dir=str(WORK_ROOT)) as tmp:
        workdir = Path(tmp)
        try:
            _materialize_files(workdir, files)
            compile_cmd, run_cmd, _ = spec_fn(workdir, code)
        except Exception as exc:
            return {
                "status": "Error",
                "message": f"{type(exc).__name__}: {exc}",
                "compile_result": None,
                "run_result": None,
                "executor_pod_name": None,
                "files": {},
            }

        compile_result = None
        if compile_cmd:
            compile_result = _run_one(
                compile_cmd, cwd=workdir, timeout=compile_timeout, stdin=None
            )
            if compile_result["status"] != "Finished":
                return {
                    "status": "CompileError",
                    "message": "compile failed",
                    "compile_result": compile_result,
                    "run_result": None,
                    "executor_pod_name": None,
                    "files": _harvest_fetch_files(workdir, fetch_files),
                }

        run_result = _run_one(
            run_cmd, cwd=workdir, timeout=run_timeout, stdin=stdin
        )
        out_files = _harvest_fetch_files(workdir, fetch_files)
        overall = "Success" if run_result["status"] == "Finished" else "Error"
        return {
            "status": overall,
            "message": "",
            "compile_result": compile_result,
            "run_result": run_result,
            "executor_pod_name": None,
            "files": out_files,
        }


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "HWALocalSandbox/0.1"

    def _write_json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/healthz", "/api/v1/list_supported_languages"):
            self._write_json(200, {
                "status": "ok",
                "supported": sorted(LANG_SPECS),
                "cpu_seconds": CPU_SECONDS,
                "mem_bytes": MEM_BYTES,
            })
            return
        self._write_json(404, {"detail": "Not Found"})

    def do_POST(self):  # noqa: N802
        if self.path != "/run_code":
            self._write_json(404, {"detail": "Not Found"})
            return
        length = int(self.headers.get("content-length") or 0)
        try:
            raw = self.rfile.read(length) if length > 0 else b""
            body = json.loads(raw or b"{}")
        except Exception as exc:
            self._write_json(400, {"detail": f"invalid json: {exc}"})
            return
        try:
            result = handle_run_code(body)
        except Exception as exc:
            tb = traceback.format_exc()
            self._write_json(500, {
                "status": "Error",
                "message": f"{type(exc).__name__}: {exc}\n{tb}",
            })
            return
        self._write_json(200, result)

    def log_message(self, format, *args):  # quieter logs
        sys.stderr.write("%s - - [%s] %s\n" % (
            self.address_string(), self.log_date_time_string(), format % args))


class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    httpd = ThreadingServer((args.host, args.port), Handler)
    print(f"[local_sandbox] listening on http://{args.host}:{args.port} "
          f"(cpu={CPU_SECONDS}s mem={MEM_BYTES} workdir={WORK_ROOT})",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
