"""Shared helpers for the task-design stage: JSON extraction and a cached
Jina embedding client (used to rank semantically related O/S prototypes)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterator

import httpx
from openai import OpenAI


def clean_text(value: object, limit: int = 4000) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(text)):
            char = text[i]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = not in_string
            elif not in_string and char == "{":
                depth += 1
            elif not in_string and char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def read_jsonl(path: Path) -> Iterator[dict]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


class EmbeddingCache:
    def __init__(self, path: Path, model: str, task: str):
        self.path = path
        self.model = model
        self.task = task
        self.rows: dict[str, list[float]] = {}
        if path.is_file():
            for row in read_jsonl(path):
                if row.get("model") == model and row.get("task") == task:
                    vector = row.get("embedding")
                    if isinstance(vector, list) and vector:
                        self.rows[str(row.get("key"))] = vector

    def key(self, text: str) -> str:
        payload = f"{self.model}\0{self.task}\0{text}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, text: str) -> list[float] | None:
        return self.rows.get(self.key(text))

    def add_many(self, values: list[tuple[str, list[float]]]) -> None:
        if not values:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for text, vector in values:
                key = self.key(text)
                self.rows[key] = vector
                handle.write(
                    json.dumps(
                        {
                            "key": key,
                            "model": self.model,
                            "task": self.task,
                            "embedding": vector,
                        }
                    )
                    + "\n"
                )


class JinaPoolEmbedder:
    """OpenAI-compatible embedding client for Jina embeddings.

    Defaults to the public endpoint (https://api.jina.ai/v1, key from
    JINA_API_KEY); any OpenAI-compatible /v1/embeddings server also works."""

    def __init__(
        self,
        base_url: str,
        model: str,
        cache_path: Path,
        *,
        task: str = "text-matching",
        batch_size: int = 32,
        timeout: float = 180,
        api_key: str = "",
    ):
        base = base_url.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        self.base_url = base
        self.model = model
        self.task = task
        self.batch_size = batch_size
        local = base.startswith(("http://127.0.0.1", "http://localhost"))
        self.client = OpenAI(
            api_key=api_key or os.environ.get("JINA_API_KEY") or "EMPTY",
            base_url=base,
            timeout=timeout,
            max_retries=3,
            # Local servers must bypass any HTTP(S)_PROXY; the public API honours it.
            http_client=httpx.Client(trust_env=not local),
        )
        self.cache = EmbeddingCache(cache_path, model, task)
        self.dimension: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        normalized = [clean_text(text, 7000) for text in texts]
        result: list[list[float] | None] = [self.cache.get(text) for text in normalized]
        missing = [i for i, vector in enumerate(result) if vector is None]
        for start in range(0, len(missing), self.batch_size):
            indices = missing[start : start + self.batch_size]
            payload = [normalized[i] for i in indices]
            response = self.client.embeddings.create(
                model=self.model,
                input=payload,
                encoding_format="float",
                extra_body={"task": self.task},
            )
            ordered = sorted(response.data, key=lambda row: row.index)
            if len(ordered) != len(indices):
                raise RuntimeError(
                    f"embedding endpoint returned {len(ordered)} vectors for {len(indices)} inputs"
                )
            additions: list[tuple[str, list[float]]] = []
            for index, item in zip(indices, ordered):
                vector = list(item.embedding)
                result[index] = vector
                additions.append((normalized[index], vector))
            self.cache.add_many(additions)
        vectors = [vector for vector in result if vector is not None]
        if len(vectors) != len(texts):
            raise RuntimeError("embedding cache/result alignment failed")
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise RuntimeError(f"inconsistent embedding dimensions: {dimensions}")
        self.dimension = next(iter(dimensions))
        return vectors
