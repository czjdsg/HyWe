import os
from typing import Any

import requests

from hybrid_utils import compact_observation_text


SUPPORTED_LANGUAGES = {
    "python",
    "cpp",
    "nodejs",
    "go",
    "go_test",
    "java",
    "php",
    "csharp",
    "bash",
    "typescript",
    "sql",
    "rust",
    "cuda",
    "lua",
    "R",
    "perl",
    "D_ut",
    "ruby",
    "scala",
    "julia",
    "pytest",
    "junit",
    "kotlin_script",
    "jest",
    "verilog",
    "python_gpu",
    "lean",
    "swift",
    "racket",
}


class CodeSandboxTool:
    def __init__(
        self,
        base_url: str = "",
        request_timeout: float = 90,
        max_output_chars: int = 0,
    ):
        self.base_url = (
            base_url
            or os.getenv("CODE_SANDBOX_BASE_URL")
            or "http://127.0.0.1:8080"
        ).rstrip("/")
        self.request_timeout = float(
            request_timeout
            or os.getenv("CODE_SANDBOX_REQUEST_TIMEOUT", "90")
        )
        # 0 / None / negative = no truncation. Don't `or` here — that
        # would turn an explicit 0 into the env / 6000 fallback.
        if max_output_chars is None:
            env_val = os.getenv("CODE_SANDBOX_MAX_OUTPUT_CHARS", "0")
            self.max_output_chars = int(env_val or 0)
        else:
            self.max_output_chars = int(max_output_chars)

    def call(
        self,
        code: str,
        language: str = "python",
        stdin: str | None = None,
        run_timeout: float = 10,
        compile_timeout: float = 10,
        files: dict[str, str | None] | None = None,
        fetch_files: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "language": language,
            "code": code,
            "run_timeout": run_timeout,
            "compile_timeout": compile_timeout,
        }
        if stdin is not None:
            payload["stdin"] = stdin
        if files:
            payload["files"] = files
        if fetch_files:
            payload["fetch_files"] = fetch_files

        try:
            response = requests.post(
                f"{self.base_url}/run_code",
                json=payload,
                timeout=self.request_timeout,
            )
        except Exception as exc:
            return f"[code_sandbox] Request failed: {type(exc).__name__}: {exc}"

        if response.status_code != 200:
            return (
                "[code_sandbox] Sandbox returned HTTP "
                f"{response.status_code}: "
                f"{compact_observation_text(response.text, max_chars=self.max_output_chars)}"
            )

        try:
            data = response.json()
        except Exception as exc:
            return (
                f"[code_sandbox] Invalid JSON response: {type(exc).__name__}: {exc}\n"
                f"{compact_observation_text(response.text, max_chars=self.max_output_chars)}"
            )

        return self._format_response(data)

    def _format_command_result(self, title: str, result: dict | None) -> list[str]:
        if not result:
            return [f"{title}: [not applicable]"]

        lines = [
            f"{title}:",
            f"- status: {result.get('status', '')}",
            f"- return_code: {result.get('return_code', '')}",
            f"- execution_time: {result.get('execution_time', '')}",
        ]
        stdout = result.get("stdout")
        stderr = result.get("stderr")
        if stdout:
            lines.extend(["- stdout:", stdout])
        if stderr:
            lines.extend(["- stderr:", stderr])
        return lines

    def _format_response(self, data: dict) -> str:
        lines = [
            "Code sandbox execution result:",
            f"- status: {data.get('status', '')}",
            f"- message: {data.get('message', '')}",
        ]
        lines.extend(self._format_command_result("Compile result", data.get("compile_result")))
        lines.extend(self._format_command_result("Run result", data.get("run_result")))

        files = data.get("files") or {}
        if files:
            lines.append("Fetched files:")
            for path, content in files.items():
                text = content if isinstance(content, str) else str(content)
                lines.append(f"- {path}: {text[:1000]}")

        return compact_observation_text(
            "\n".join(str(item) for item in lines),
            max_chars=self.max_output_chars,
        )
