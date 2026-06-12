import os
import time
from typing import Iterable

import requests


DEFAULT_SERPER_BASE_URL = "https://google.serper.dev"


class SearchTool:
    def __init__(self):
        self.api_key = (
            os.getenv("SERPER_KEY_ID")
            or os.getenv("SERPER_API_KEY")
            or os.getenv("SERPER_KEY")
        )
        self.base_url = os.getenv("SERPER_BASE_URL", DEFAULT_SERPER_BASE_URL).rstrip("/")

    def _has_serper_access(self) -> bool:
        return bool(
            self.api_key
            or os.getenv("SERPER_POOL_TOKEN")
            or self.base_url != DEFAULT_SERPER_BASE_URL
        )

    def _build_headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        pool_token = os.getenv("SERPER_POOL_TOKEN")
        if pool_token:
            headers["X-Pool-Token"] = pool_token
        return headers

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text or "")

    def _build_payload(self, query: str) -> dict:
        if self._contains_chinese(query):
            return {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        return {"q": query, "location": "United States", "gl": "us", "hl": "en"}

    def _format_results(self, query: str, results: dict) -> str:
        organic = results.get("organic") or []
        if not organic:
            return f"No results found for '{query}'."

        lines = [f"Search results for '{query}':"]
        for idx, page in enumerate(organic[:8], start=1):
            title = page.get("title", "Untitled result")
            link = page.get("link", "")
            snippet = page.get("snippet", "")
            source = page.get("source", "")
            date = page.get("date", "")
            parts = [f"{idx}. {title}", f"URL: {link}"]
            if source:
                parts.append(f"Source: {source}")
            if date:
                parts.append(f"Date: {date}")
            if snippet:
                parts.append(f"Snippet: {snippet}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    def _search_single(self, query: str) -> str:
        if not self._has_serper_access():
            return (
                "[search] Missing Serper access. "
                "Set SERPER_KEY_ID/SERPER_API_KEY/SERPER_KEY, "
                "or point SERPER_BASE_URL to the local Serper pool."
            )

        headers = self._build_headers()
        payload = self._build_payload(query)

        for attempt in range(5):
            try:
                response = requests.post(
                    f"{self.base_url}/search",
                    json=payload,
                    headers=headers,
                    timeout=20,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Serper returned HTTP {response.status_code}: {response.text}"
                    )
                return self._format_results(query, response.json())
            except Exception as exc:
                if attempt == 4:
                    return f"[search] Search failed for '{query}': {exc}"
                time.sleep(0.5 * (attempt + 1))

        return f"[search] Search failed for '{query}'."

    def call(self, query: str | Iterable[str]) -> str:
        if isinstance(query, str):
            query = [query]

        cleaned = [item.strip() for item in query if isinstance(item, str) and item.strip()]
        if not cleaned:
            return "[search] Invalid request: query must be a non-empty string or list of strings."

        return "\n=======\n".join(self._search_single(item) for item in cleaned)
