"""Embedding adapter, mirroring the shape of recipes/llm.py (SPEC.md §10)."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class EmbeddingResponse:
    vectors: list[list[float]]
    input_tokens: int
    latency_ms: float = 0.0


class Embedder(Protocol):
    def embed(self, texts: list[str], model: str) -> EmbeddingResponse: ...


class OpenAIEmbedder:
    """Production adapter. Requires OPENAI_API_KEY to be set."""

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def embed(self, texts: list[str], model: str) -> EmbeddingResponse:
        start = time.perf_counter()
        response = self._client.embeddings.create(model=model, input=texts)
        latency_ms = (time.perf_counter() - start) * 1000
        vectors = [item.embedding for item in response.data]
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        return EmbeddingResponse(vectors=vectors, input_tokens=input_tokens, latency_ms=latency_ms)


class MockEmbedder:
    """Deterministic adapter for tests and CI. Never makes a network call.

    Produces a fixed-length pseudo-embedding per text by hashing the text
    into floats. Same text always yields the same vector, so retrieval math
    (similarity ranking) is testable without any real embedding model.
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.calls: list[tuple[list[str], str]] = []

    def embed(self, texts: list[str], model: str) -> EmbeddingResponse:
        self.calls.append((texts, model))
        vectors = [self._hash_vector(text) for text in texts]
        input_tokens = sum(max(1, len(t) // 4) for t in texts)
        return EmbeddingResponse(vectors=vectors, input_tokens=input_tokens, latency_ms=0.05)

    def _hash_vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat/trim the digest bytes to `dim` floats in [-1, 1].
        raw = (digest * ((self.dim // len(digest)) + 1))[: self.dim]
        return [(b / 127.5) - 1.0 for b in raw]


def get_embedder() -> Embedder:
    """Factory respecting the RAG_RECIPES_LLM env var (SPEC.md §12/§13)."""
    backend = os.environ.get("RAG_RECIPES_LLM", "openai").lower()
    if backend == "mock":
        return MockEmbedder()
    return OpenAIEmbedder()
