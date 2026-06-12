import base64
import binascii
import json
import mimetypes
import os
import re
import time
from io import BytesIO
from typing import Callable, Iterable
from urllib.parse import urljoin

import requests
from PIL import Image


DEFAULT_JINA_READER_BASE_URL = "https://r.jina.ai"
KNOWN_JINA_PREFIXES = ("https://r.jina.ai/", "http://r.jina.ai/")


INCREMENTAL_EXTRACTOR_PROMPT = """Please process the following webpage chunk and user goal to update the accumulated extraction.

## Existing Extraction
{existing_summary}

## New Webpage Chunk ({chunk_index}/{chunk_count})
{webpage_content}

## User Goal
{goal}

## Task Guidelines
1. Use the existing extraction as the starting point, then merge in any relevant evidence from the new chunk.
2. Preserve important names, numbers, dates, URLs, titles, table values, and constraints that may help answer the user goal.
3. Remove duplicates and ignore content that is not useful for the goal.
4. If the new chunk has no relevant information, keep the useful existing extraction.
5. Keep the final extraction concise but complete enough for downstream reasoning.
6. Keep "rational" to one short sentence.
7. Keep "evidence" to at most 8 concise items.
8. Keep "summary" to at most 250 words.

Final Output Format:
Return exactly one complete JSON object with keys "rational", "evidence", and "summary". Do not use markdown fences or extra text.
""".strip()


DATA_URL_PATTERN = re.compile(
    r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\n\r]+"
)
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
ALLOWED_RASTER_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/x-icon",
    "image/vnd.microsoft.icon",
}

ALLOWED_PARSE_TYPES = {"auto", "html", "pdf"}


def _build_jina_headers(api_key: str = "") -> dict[str, str]:
    headers: dict[str, str] = {}
    resolved_key = api_key or os.getenv("JINA_API_KEY") or os.getenv("JINA_API_KEYS") or ""
    if resolved_key:
        headers["Authorization"] = f"Bearer {resolved_key}"
    pool_token = os.getenv("JINA_POOL_TOKEN")
    if pool_token:
        headers["X-Pool-Token"] = pool_token
    return headers


def _build_jina_reader_url(url: str) -> str:
    normalized = (url or "").strip()
    for prefix in KNOWN_JINA_PREFIXES:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break
    base_url = os.getenv("JINA_READER_BASE_URL", DEFAULT_JINA_READER_BASE_URL).rstrip("/")
    return f"{base_url}/{normalized}"


def _load_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception:
        return None


def _parse_json_output(raw: str) -> dict | None:
    if not raw:
        return None

    candidates = [raw.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None


def _stringify_summary_field(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            text = _stringify_summary_field(item)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2).strip()
        except Exception:
            return str(value).strip()
    return str(value).strip()


class VisitTool:
    def __init__(
        self,
        summarize_fn: Callable[[str, list[str]], str],
        trace_callback: Callable[[dict], None] | None = None,
        jina_api_key: str = "",
        tokenizer_path: str = "",
        max_webpage_tokens: int = 32000,
        max_webpage_chars: int = 0,
        max_images: int = 4,
        max_image_url_chars: int = 2_000_000,
    ):
        self.summarize_fn = summarize_fn
        self.trace_callback = trace_callback
        self.jina_api_key = jina_api_key or os.getenv("JINA_API_KEY", "")
        self.tokenizer_path = (
            tokenizer_path
            or os.getenv("TOKENIZER_PATH", "Qwen/Qwen2.5-7B-Instruct")
        )
        self.max_webpage_tokens = int(
            max_webpage_tokens or os.getenv("MAX_WEBPAGE_TOKENS", "32000")
        )
        self.max_webpage_chars = max_webpage_chars
        self.max_images = max_images
        self.max_image_url_chars = max_image_url_chars
        self.tokenizer = _load_tokenizer(self.tokenizer_path)
        self.max_image_download_bytes = 5_000_000
        self.max_pdf_download_bytes = int(
            os.getenv("MAX_PDF_DOWNLOAD_BYTES", "50000000")
        )

    def call(self, url: str | Iterable[str], goal: str, parse_type: str = "auto") -> str:
        urls: list[str]
        if isinstance(url, str):
            urls = [url]
        else:
            urls = [
                item.strip()
                for item in url
                if isinstance(item, str) and item.strip()
            ]

        goal = (goal or "").strip()
        if not urls or not goal:
            return "[visit] Missing required parameters: url and goal."

        if parse_type is None:
            parse_type = "auto"
        if not isinstance(parse_type, str):
            return "[visit] Invalid parse_type: expected one of auto, html, or pdf."
        parse_type = (parse_type or "auto").strip().lower()
        if parse_type not in ALLOWED_PARSE_TYPES:
            return "[visit] Invalid parse_type: expected one of auto, html, or pdf."

        outputs = [self._visit_single(item, goal, parse_type) for item in urls[:5]]
        return "\n=======\n".join(outputs)

    def _visit_single(self, url: str, goal: str, parse_type: str = "auto") -> str:
        content = self._jina_fetch(url)
        if (
            (not content or content.startswith("[visit]"))
            and parse_type in {"auto", "pdf"}
            and (parse_type == "pdf" or self._looks_like_pdf_url(url))
        ):
            content = self._pdf_fetch(url)

        if not content or content.startswith("[visit]"):
            if self.trace_callback is not None:
                self.trace_callback(
                    {
                        "url": url,
                        "goal": goal,
                        "parse_type": parse_type,
                        "status": "fetch_failed",
                        "image_count": 0,
                        "image_meta": [],
                        "summary_prompt_text": "",
                        "summary_response_text": "",
                        "truncation_note": "",
                        "fetch_result": content or "",
                    }
                )
            return content or f"[visit] Failed to read page (url: {url}, goal: {goal})."

        raw_content = content
        raw_content_tokens = self._count_tokens(raw_content)
        chunks, chunk_note = self._chunk_content(content)
        raw_response = ""
        parsed: dict | None = None
        accumulated_summary = ""
        chunk_count = len(chunks)

        for chunk_index, chunk in enumerate(chunks, 1):
            summary_prompt_text = INCREMENTAL_EXTRACTOR_PROMPT.format(
                existing_summary=accumulated_summary,
                webpage_content=chunk,
                goal=goal,
                chunk_index=chunk_index,
                chunk_count=chunk_count,
            )
            raw_response = self.summarize_fn(summary_prompt_text, [])
            parsed_candidate = _parse_json_output(raw_response)
            parsed = parsed_candidate if isinstance(parsed_candidate, dict) else None
            accumulated_summary = self._accumulated_summary_text(
                raw_response,
                parsed,
            )

            if self.trace_callback is not None:
                trace = {
                    "url": url,
                    "goal": goal,
                    "parse_type": parse_type,
                    "status": "summarized",
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "image_count": 0,
                    "image_meta": [],
                    "summary_prompt_text": summary_prompt_text,
                    "summary_response_text": raw_response,
                    "summary_parsed_json": parsed,
                    "truncation_note": chunk_note,
                    "raw_content_chars": len(raw_content),
                    "raw_content_tokens": raw_content_tokens,
                }
                if chunk_index == 1:
                    trace["raw_content"] = raw_content
                self.trace_callback(trace)
        useful_info = f"The useful information in '{url}' for user goal '{goal}':\n\n"

        if isinstance(parsed, dict):
            evidence = _stringify_summary_field(parsed.get("evidence"))
            summary = _stringify_summary_field(parsed.get("summary"))
            if evidence:
                useful_info += f"Evidence in page:\n{evidence}\n\n"
            if summary:
                useful_info += f"Summary:\n{summary}\n"
            if evidence or summary:
                if chunk_note:
                    useful_info += f"\nNote:\n{chunk_note}\n"
                return useful_info.strip()

        if chunk_note:
            useful_info += f"Note:\n{chunk_note}\n\n"
        return (useful_info + raw_response.strip()).strip()

    def _jina_fetch(self, url: str) -> str:
        headers = _build_jina_headers(self.jina_api_key)

        direct_headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }

        for attempt in range(3):
            try:
                response = requests.get(
                    _build_jina_reader_url(url),
                    headers=headers,
                    timeout=60,
                )
                if response.status_code == 200 and response.text.strip():
                    return self._parse_jina_response(response.text)
            except Exception:
                pass
            time.sleep(1 + attempt)

        for attempt in range(2):
            try:
                response = requests.get(
                    url,
                    headers=direct_headers,
                    timeout=30,
                )
                response.raise_for_status()
                content_type = (
                    response.headers.get("Content-Type") or ""
                ).split(";", 1)[0].strip().lower()
                if "pdf" in content_type or response.content.startswith(b"%PDF"):
                    return "[visit] Direct fetch returned a PDF document."
                if response.text.strip():
                    return response.text
            except Exception:
                if attempt == 1:
                    return f"[visit] Failed to read page (url: {url})."
                time.sleep(1 + attempt)

        return f"[visit] Failed to read page (url: {url})."

    @staticmethod
    def _looks_like_pdf_url(url: str) -> bool:
        normalized = (url or "").strip().lower()
        if not normalized:
            return False
        return (
            normalized.endswith(".pdf")
            or ".pdf?" in normalized
            or "/pdf/" in normalized
            or "arxiv.org/pdf" in normalized
        )

    def _pdf_fetch(self, url: str) -> str:
        pdf_bytes, error = self._download_pdf_bytes(url)
        if not pdf_bytes:
            return error or f"[visit] Failed to download PDF (url: {url})."
        return self._parse_pdf_bytes(pdf_bytes, url)

    def _download_pdf_bytes(self, url: str) -> tuple[bytes, str]:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        }

        try:
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()

            content_type = (
                response.headers.get("Content-Type") or ""
            ).split(";", 1)[0].strip().lower()
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > self.max_pdf_download_bytes:
                    return b"", (
                        "[visit] PDF download exceeded "
                        f"{self.max_pdf_download_bytes} bytes (url: {url})."
                    )
                chunks.append(chunk)

            data = b"".join(chunks)
            if not data:
                return b"", f"[visit] Empty PDF download (url: {url})."
            if (
                "pdf" not in content_type
                and not data.startswith(b"%PDF")
                and not self._looks_like_pdf_url(url)
            ):
                return b"", f"[visit] Downloaded content is not a PDF (url: {url})."
            return data, ""
        except Exception as exc:
            return b"", f"[visit] Failed to download PDF (url: {url}): {exc}"

    @staticmethod
    def _parse_pdf_bytes(pdf_bytes: bytes, url: str) -> str:
        try:
            import fitz
        except Exception:
            return "[visit] PDF parser unavailable: PyMuPDF is not installed."

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(f"## Page {page_num + 1}\n\n{text}")
            doc.close()
        except Exception as exc:
            return f"[visit] Failed to parse PDF (url: {url}): {exc}"

        content = "\n\n".join(text_parts).strip()
        if not content:
            return f"[visit] PDF contained no extractable text (url: {url})."
        return content

    def _parse_jina_response(self, raw_text: str) -> str:
        try:
            payload = json.loads(raw_text)
        except Exception:
            return raw_text

        if not isinstance(payload, dict):
            return raw_text

        content = (
            payload.get("content")
            or (payload.get("data") or {}).get("content")
            or raw_text
        )
        if not isinstance(content, str):
            content = raw_text
        return content

    def _extract_images(self, content: str, base_url: str) -> tuple[str, list[dict]]:
        found = DATA_URL_PATTERN.findall(content or "")
        image_payloads: list[dict] = []
        seen_data_urls: set[str] = set()
        for item in found:
            normalized = re.sub(r"\s+", "", item)
            if not normalized.startswith("data:image/"):
                continue
            payload = self._payload_from_data_url(
                normalized,
                source_kind="inline_data_url",
                source_url="",
            )
            if not payload:
                continue
            if payload["data_url"] in seen_data_urls:
                continue
            
            image_payloads.append(payload)
            seen_data_urls.add(payload["data_url"])
            if len(image_payloads) >= self.max_images:
                break

        clean_text = DATA_URL_PATTERN.sub("[inline image attached separately]", content or "")
        remaining_slots = self.max_images - len(image_payloads)
        if remaining_slots > 0:
            for raw_image_url in MARKDOWN_IMAGE_PATTERN.findall(clean_text):
                image_url = urljoin(base_url, raw_image_url.strip())
                image_payload = self._download_image_as_data_url(image_url)
                if not image_payload:
                    continue
                data_url = image_payload["data_url"]
                if data_url in seen_data_urls:
                    continue
                image_payloads.append(image_payload)
                seen_data_urls.add(data_url)
                if len(image_payloads) >= self.max_images:
                    break

        return clean_text, image_payloads

    def _download_image_as_data_url(self, image_url: str) -> dict | None:
        try:
            response = requests.get(
                image_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"
                    )
                },
                timeout=30,
            )
            response.raise_for_status()
            content = response.content or b""
        except Exception:
            return None

        if not content or len(content) > self.max_image_download_bytes:
            return None

        content_type = (response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if not content_type.startswith("image/"):
            guessed_type, _ = mimetypes.guess_type(image_url)
            content_type = (guessed_type or "").lower()

        content_type = self._normalize_content_type(content_type)
        if content_type not in ALLOWED_RASTER_CONTENT_TYPES:
            return None

        if not self._is_valid_raster_image(content):
            return None

        data_url = f"data:{content_type};base64,{base64.b64encode(content).decode('ascii')}"
        if len(data_url) > self.max_image_url_chars:
            return None

        return {
            "source_kind": "downloaded_image_url",
            "source_url": image_url,
            "data_url": data_url,
            "length": len(data_url),
        }

    def _payload_from_data_url(
        self,
        data_url: str,
        source_kind: str,
        source_url: str,
    ) -> dict | None:
        if len(data_url) > self.max_image_url_chars or not data_url.startswith("data:image/"):
            return None

        try:
            header, encoded = data_url.split(",", 1)
        except ValueError:
            return None

        if ";base64" not in header:
            return None

        content_type = self._normalize_content_type(
            header[len("data:") :].split(";", 1)[0].strip().lower()
        )
        if content_type not in ALLOWED_RASTER_CONTENT_TYPES:
            return None

        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            return None

        if not self._is_valid_raster_image(image_bytes):
            return None

        return {
            "source_kind": source_kind,
            "source_url": source_url,
            "data_url": data_url,
            "length": len(data_url),
        }

    def _normalize_content_type(self, content_type: str) -> str:
        content_type = (content_type or "").strip().lower()
        if content_type == "image/svg+xml":
            return content_type
        if content_type == "image/jpg":
            return "image/jpeg"
        return content_type

    def _is_valid_raster_image(self, content: bytes) -> bool:
        if not content:
            return False
        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()
                if (image.format or "").upper() == "SVG":
                    return False
            with Image.open(BytesIO(content)) as image:
                return (image.format or "").upper() in {
                    "PNG",
                    "JPEG",
                    "WEBP",
                    "GIF",
                    "BMP",
                    "ICO",
                }
        except Exception:
            return False

    def _chunk_content(self, content: str) -> tuple[list[str], str]:
        text = content or ""
        note_parts: list[str] = []
        chunks: list[str] = [text]

        if self.tokenizer is not None and self.max_webpage_tokens > 0:
            try:
                tokens = self.tokenizer.encode(text)
                if len(tokens) > self.max_webpage_tokens:
                    chunks = [
                        self.tokenizer.decode(
                            tokens[index : index + self.max_webpage_tokens],
                            skip_special_tokens=True,
                        )
                        for index in range(0, len(tokens), self.max_webpage_tokens)
                    ]
                    note_parts.append(
                        "Webpage content processed in token chunks of up to "
                        f"{self.max_webpage_tokens} tokens."
                    )
            except Exception:
                pass

        if self.max_webpage_chars:
            char_chunks: list[str] = []
            for chunk in chunks:
                char_chunks.extend(
                    self._split_text_by_chars(chunk, self.max_webpage_chars)
                )
            chunks = char_chunks or [""]

        chunks = [chunk for chunk in chunks if chunk.strip()]
        if not chunks:
            chunks = [""]

        if len(chunks) > 1:
            if self.max_webpage_chars:
                note_parts.append(
                    "Webpage content summarized incrementally in "
                    f"{len(chunks)} chunks of up to {self.max_webpage_chars} characters."
                )
            else:
                note_parts.append(
                    f"Webpage content summarized incrementally in {len(chunks)} chunks."
                )

        return chunks, " ".join(note_parts).strip()

    def _count_tokens(self, text: str) -> int | None:
        if self.tokenizer is None:
            return None
        try:
            return len(self.tokenizer.encode(text or ""))
        except Exception:
            return None

    @staticmethod
    def _split_text_by_chars(text: str, max_chars: int) -> list[str]:
        if max_chars <= 0 or len(text) <= max_chars:
            return [text]

        chunks: list[str] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + max_chars, text_length)
            if end < text_length:
                split_at = text.rfind("\n\n", start, end)
                if split_at <= start + max_chars // 2:
                    split_at = text.rfind("\n", start, end)
                if split_at <= start + max_chars // 2:
                    split_at = text.rfind(" ", start, end)
                if split_at > start:
                    end = split_at

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end
            while start < text_length and text[start].isspace():
                start += 1

        return chunks

    @staticmethod
    def _accumulated_summary_text(raw_response: str, parsed: dict | None) -> str:
        if isinstance(parsed, dict):
            sections = []
            evidence = _stringify_summary_field(parsed.get("evidence"))
            summary = _stringify_summary_field(parsed.get("summary"))
            if evidence:
                sections.append(f"Evidence:\n{evidence}")
            if summary:
                sections.append(f"Summary:\n{summary}")
            if sections:
                return "\n\n".join(sections).strip()
        return (raw_response or "").strip()


def copy_image_meta(image_payloads: list[dict]) -> list[dict]:
    return [
        {
            "index": index + 1,
            "source_kind": item.get("source_kind", ""),
            "source_url": item.get("source_url", ""),
            "length": item.get("length", 0),
        }
        for index, item in enumerate(image_payloads or [])
    ]
