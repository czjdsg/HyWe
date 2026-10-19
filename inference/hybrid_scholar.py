import os
import time
from typing import Iterable

import requests


DEFAULT_SERPER_BASE_URL = "https://google.serper.dev"


class ScholarTool:
    def __init__(self):
        self.api_key = (
            os.getenv("SERPER_KEY_ID")
            or os.getenv("SERPER_API_KEY")
            or os.getenv("SERPER_KEY")
        )
        self.base_url = (
            os.getenv("SCHOLAR_BASE_URL")
            or os.getenv("SERPER_BASE_URL", DEFAULT_SERPER_BASE_URL)
        ).rstrip("/")

    def _has_scholar_access(self) -> bool:
        return bool(
            self.api_key
            or self.base_url != DEFAULT_SERPER_BASE_URL
        )

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        return headers

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text or "")

    def _build_payload(self, query: str) -> dict:
        if self._contains_chinese(query):
            return {
                "q": query,
                "location": "China",
                "gl": "cn",
                "hl": "zh-cn",
                "num": 10,
            }
        return {
            "q": query,
            "location": "United States",
            "gl": "us",
            "hl": "en",
            "num": 10,
        }

    @staticmethod
    def _extract_publication_info(page: dict) -> str:
        publication_info = page.get("publicationInfo")
        if isinstance(publication_info, str):
            return publication_info

        publication_info = page.get("publication_info")
        if isinstance(publication_info, dict):
            return publication_info.get("summary", "") or ""

        return ""

    @staticmethod
    def _extract_citation_count(page: dict) -> str:
        cited_by = page.get("citedBy")
        if cited_by not in (None, ""):
            return str(cited_by)

        inline_cited_by = (
            (page.get("inline_links") or {}).get("cited_by")
            if isinstance(page.get("inline_links"), dict)
            else {}
        )
        if isinstance(inline_cited_by, dict):
            total = inline_cited_by.get("total")
            if total not in (None, ""):
                return str(total)

        return ""

    @staticmethod
    def _extract_pdf_url(page: dict) -> str:
        pdf_url = page.get("pdfUrl") or page.get("pdf_url")
        if isinstance(pdf_url, str) and pdf_url.strip():
            return pdf_url.strip()

        resources = page.get("resources") or []
        if isinstance(resources, list):
            for resource in resources:
                if not isinstance(resource, dict):
                    continue
                file_format = (resource.get("file_format") or "").strip().lower()
                link = (resource.get("link") or "").strip()
                if file_format == "pdf" and link:
                    return link

        return ""

    def _format_results(self, query: str, results: dict) -> str:
        organic = results.get("organic") or results.get("organic_results") or []
        if not organic:
            return f"No scholar results found for '{query}'."

        lines = [f"Scholar results for '{query}':"]
        for idx, page in enumerate(organic[:10], start=1):
            if not isinstance(page, dict):
                continue

            title = page.get("title", "Untitled scholar result")
            link = page.get("link") or page.get("url") or ""
            snippet = page.get("snippet", "")
            publication_info = self._extract_publication_info(page)
            citation_count = self._extract_citation_count(page)
            pdf_url = self._extract_pdf_url(page)
            year = page.get("year", "")

            parts = [f"{idx}. {title}"]
            if link:
                parts.append(f"URL: {link}")
            if pdf_url:
                parts.append(f"PDF URL: {pdf_url}")
            if publication_info:
                parts.append(f"Publication: {publication_info}")
            if year:
                parts.append(f"Year: {year}")
            if citation_count:
                parts.append(f"Cited by: {citation_count}")
            if snippet:
                parts.append(f"Snippet: {snippet}")
            lines.append("\n".join(parts))

        return "\n\n".join(lines)

    def _search_single(self, query: str) -> str:
        if not self._has_scholar_access():
            return (
                "[scholar] Missing Scholar/Serper access. "
                "Set SERPER_KEY_ID/SERPER_API_KEY/SERPER_KEY, "
                "or point SERPER_BASE_URL or SCHOLAR_BASE_URL to a Serper-compatible endpoint."
            )

        headers = self._build_headers()
        payload = self._build_payload(query)

        for attempt in range(5):
            try:
                response = requests.post(
                    f"{self.base_url}/scholar",
                    json=payload,
                    headers=headers,
                    timeout=30,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Scholar endpoint returned HTTP {response.status_code}: {response.text}"
                    )
                return self._format_results(query, response.json())
            except Exception as exc:
                if attempt == 4:
                    return f"[scholar] Scholar search failed for '{query}': {exc}"
                time.sleep(0.5 * (attempt + 1))

        return f"[scholar] Scholar search failed for '{query}'."

    def call(self, query: str | Iterable[str]) -> str:
        if isinstance(query, str):
            query = [query]

        cleaned = [item.strip() for item in query if isinstance(item, str) and item.strip()]
        if not cleaned:
            return "[scholar] Invalid request: query must be a non-empty string or list of strings."

        return "\n=======\n".join(self._search_single(item) for item in cleaned)
