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
            return {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        return {"q": query, "location": "United States", "gl": "us", "hl": "en"}

    def _format_results(self, query: str, results: dict) -> str:
        organic = results.get("organic") or []
        if not organic:
            return f"No results found for '{query}'."

        lines = [f"Search results for '{query}':"]
        for idx, page in enumerate(organic[:10], start=1):
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

    def _format_jina_results(self, query: str, results: dict) -> str:
        """Format s.jina.ai SERP JSON into the exact serper output shape so the
        model-facing observation is provider-agnostic."""
        from urllib.parse import urlsplit

        organic = results.get("data") or []
        if not organic:
            return f"No results found for '{query}'."
        lines = [f"Search results for '{query}':"]
        for idx, page in enumerate(organic[:10], start=1):
            title = page.get("title", "Untitled result")
            link = page.get("url", "")
            snippet = page.get("description", "")
            date = page.get("date", "")
            source = urlsplit(link).netloc if link else ""
            parts = [f"{idx}. {title}", f"URL: {link}"]
            if source:
                parts.append(f"Source: {source}")
            if date:
                parts.append(f"Date: {date}")
            if snippet:
                parts.append(f"Snippet: {snippet}")
            lines.append("\n".join(parts))
        return "\n\n".join(lines)

    def _search_single_jina(self, query: str) -> str | None:
        """SERP via Jina Search (s.jina.ai).

        JINA_SEARCH_BASE_URL may point at a self-hosted gateway that exposes
        `/search?q=` (forwarded to s.jina.ai); otherwise, when the backend is
        explicitly set to jina, the public endpoint is called with JINA_API_KEY.
        Returns None when unavailable so the caller can fall back to serper."""
        base = os.getenv("JINA_SEARCH_BASE_URL", "").rstrip("/")
        backend = os.getenv("HYBRID_SEARCH_BACKEND", "").strip().lower()
        if base:
            url = f"{base}/search"
        elif backend in {"jina", "jina-only", "jina_only"}:
            url = "https://s.jina.ai/"
        else:
            return None
        headers = {
            "Accept": "application/json",
            # SERP only — skip page-content crawling (fast + ~1k tokens/query).
            "X-Respond-With": "no-content",
        }
        jina_key = os.getenv("JINA_API_KEY", "").strip()
        if jina_key:
            headers["Authorization"] = f"Bearer {jina_key}"
        attempts = max(1, int(os.getenv("JINA_SEARCH_ATTEMPTS", "4")))
        timeout = max(1.0, float(os.getenv("JINA_SEARCH_TIMEOUT", "45")))
        for attempt in range(attempts):
            try:
                response = requests.get(
                    url,
                    params={"q": query},
                    headers=headers,
                    timeout=timeout,
                )
                # 422 = s.jina.ai 的「这个查询没有结果」,不是故障:
                #   {"code":422,"name":"AssertionFailureError",
                #    "message":"No search results available for query \"...\""}
                # 当成失败会连锁出三个后果:白重试 4 次(~8s)、再打一次 serper、
                # serper 池干了就把 503 报错当观测喂给模型。2026-09-12 HLE 全集跑到
                # 一半 serper key 耗尽后暴露:42,339 次搜索里 5,976 次(14.1%)变成
                # 「Search failed ... No usable Serper key」,涉及 727 道题;而 serper
                # 还有 key 的时段,同样的查询由 serper 正常返回「没有结果」。
                # 统一成空结果,与 serper 的行为一致。
                if response.status_code == 422 and "No search results available" in response.text:
                    return f"No results found for '{query}'."
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Jina search returned HTTP {response.status_code}: {response.text[:200]}"
                    )
                return self._format_jina_results(query, response.json())
            except Exception:
                if attempt == attempts - 1:
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None

    def _search_single(self, query: str) -> str:
        # SERP 后端优先级。默认 jina 优先(便宜),serper 兜底。
        #
        # 2026-08-03:jina 的实时检索链路坏了 —— **返回 HTTP 200 + 结构完整的 JSON**,
        # 但内容与查询完全无关(查"孤注一掷 票房"返回 Microsoft 官网 / 南瓜派菜谱,
        # 而且每次调用返回的垃圾都不同;缓存命中的高频词条如"北京大学"仍正确)。
        # 102 个 active key 全测过,83 个 HTTP 200 里**零个**结果相关 —— 是服务本身
        # 坏了,不是配额,补 key 无效。
        #
        # 因为它状态码全绿,key 池、本函数、模型三层都发现不了,静默污染了整跑评测
        # (DeepSearch-2510 跑出 0/100)。所以这里加开关强制走 serper。
        # 注意:坏的只有搜索(s.jina.ai),**页面读取 r.jina.ai 正常**,`visit` 不受影响。
        backend = os.getenv("HYBRID_SEARCH_BACKEND", "").strip().lower()
        if backend != "serper":
            jina_result = self._search_single_jina(query)
            if jina_result is not None:
                return jina_result
            if backend in {"jina", "jina-only", "jina_only"}:
                return (
                    f"[search] Jina search failed for '{query}'. "
                    "Retry the query or reformulate it; Serper fallback is disabled."
                )
        if not self._has_serper_access():
            return (
                "[search] Missing Serper access. "
                "Set SERPER_API_KEY (or SERPER_KEY_ID/SERPER_KEY), "
                "or point SERPER_BASE_URL to a Serper-compatible endpoint."
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
