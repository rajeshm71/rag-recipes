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


VOYAGE_EMBEDDINGS_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbedder:
    """Production adapter for A2's embedding-swap study (P5). Requires
    VOYAGE_API_KEY. Uses voyage-4 -- SPEC.md named voyage-3, but that
    family is deprecated as of the 2026-08-12 verification against
    docs.voyageai.com/docs/pricing (listed under "Older models," no free
    tokens offered).

    Calls Voyage's REST API directly via `requests` rather than the
    `voyageai` SDK package: the SDK's current release transitively depends
    on `langchain-core`/`langchain-text-splitters` (confirmed via `uv
    tree` when the SDK was briefly added and immediately reverted) --
    unacceptable for this project regardless of it being transitive, not a
    direct import, given SPEC.md R1's explicit non-choice ("no LangChain,
    no LlamaIndex, no framework abstraction"). `requests` is already a
    project dependency (corpus/build_corpus.py). Endpoint/schema verified
    against docs.voyageai.com/reference/embeddings-api on 2026-08-12, not
    exhaustively confirmed against a real call (no VOYAGE_API_KEY
    available this phase) -- flagged in tasks/todo.md for confirmation
    once a key exists, same treatment as gpt-5.4-mini's temperature
    behavior in P1.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")

    def embed(self, texts: list[str], model: str) -> EmbeddingResponse:
        import requests

        start = time.perf_counter()
        response = requests.post(
            VOYAGE_EMBEDDINGS_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"input": texts, "model": model},
            timeout=60,
        )
        response.raise_for_status()
        latency_ms = (time.perf_counter() - start) * 1000

        payload = response.json()
        # Response items include an "index" field indicating original input
        # position -- sort by it rather than assuming response order matches
        # request order (not guaranteed by the documented schema).
        ordered = sorted(payload["data"], key=lambda item: item["index"])
        vectors = [item["embedding"] for item in ordered]
        input_tokens = payload.get("usage", {}).get("total_tokens", 0)
        return EmbeddingResponse(vectors=vectors, input_tokens=input_tokens, latency_ms=latency_ms)


class LocalEmbedder:
    """Production adapter for A2's bge-large-en-v1.5 leg (P5). No API key
    needed -- local model via sentence-transformers (already a dependency
    since P3). Real, working, zero-cost -- mirrors CrossEncoderReranker's
    (pattern 04, recipes/rerank.py) zero-API-key precedent.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str], model: str) -> EmbeddingResponse:
        start = time.perf_counter()
        vectors = self._model.encode(texts, convert_to_numpy=True).tolist()
        latency_ms = (time.perf_counter() - start) * 1000
        # sentence-transformers doesn't report a token count; approximate
        # the same way MockEmbedder does elsewhere in this project.
        input_tokens = sum(max(1, len(t) // 4) for t in texts)
        return EmbeddingResponse(vectors=vectors, input_tokens=input_tokens, latency_ms=latency_ms)


def get_embedder() -> Embedder:
    """Factory respecting the RAG_RECIPES_LLM env var (SPEC.md §12/§13)."""
    backend = os.environ.get("RAG_RECIPES_LLM", "openai").lower()
    if backend == "mock":
        return MockEmbedder()
    return OpenAIEmbedder()


def get_voyage_embedder() -> Embedder:
    """Factory for A2's Voyage leg, respecting RAG_RECIPES_LLM the same way
    get_embedder()/get_anthropic_llm() do -- R8 (CI is mock-only) applies
    to every provider, not just OpenAI.
    """
    backend = os.environ.get("RAG_RECIPES_LLM", "openai").lower()
    if backend == "mock":
        return MockEmbedder()
    return VoyageEmbedder()
