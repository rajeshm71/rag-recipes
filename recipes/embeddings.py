"""Embedding adapter, mirroring the shape of recipes/llm.py."""

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


def _batch_by_token_budget(texts: list[str], max_tokens: int) -> list[list[str]]:
    """Split `texts` into sub-batches, each under `max_tokens` estimated
    tokens. A single text exceeding the budget on its own still gets its
    own one-item batch (the API's per-request cap, not a per-item one)."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        text_tokens = _estimate_tokens(text)
        if current and current_tokens + text_tokens > max_tokens:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(text)
        current_tokens += text_tokens
    if current:
        batches.append(current)
    return batches


# OpenAI's embeddings endpoint caps requests at 300_000 tokens total. Batch
# under a conservative margin below that (found via a real A1_chunking_study
# run: a re-chunked corpus's full text batch requested 400_506 tokens in one
# call and was rejected with `max_tokens_per_request`).
_OPENAI_MAX_TOKENS_PER_REQUEST = 250_000


def _estimate_tokens(text: str) -> int:
    # Same char/4 approximation MockEmbedder/LocalEmbedder use elsewhere in
    # this module -- good enough to stay safely under the hard API limit.
    return max(1, len(text) // 4)


class OpenAIEmbedder:
    """Production adapter. Requires OPENAI_API_KEY to be set."""

    def __init__(self, api_key: str | None = None) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def embed(self, texts: list[str], model: str) -> EmbeddingResponse:
        start = time.perf_counter()
        vectors: list[list[float]] = []
        input_tokens = 0
        for batch in _batch_by_token_budget(texts, _OPENAI_MAX_TOKENS_PER_REQUEST):
            response = self._client.embeddings.create(model=model, input=batch)
            vectors.extend(item.embedding for item in response.data)
            input_tokens += response.usage.prompt_tokens if response.usage else 0
        latency_ms = (time.perf_counter() - start) * 1000
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
    """Production adapter for A2's embedding-swap study. Requires
    VOYAGE_API_KEY. Uses voyage-4 -- this project originally targeted
    voyage-3, but that family is deprecated as of the 2026-08-12
    verification against docs.voyageai.com/docs/pricing (listed under
    "Older models," no free tokens offered).

    Calls Voyage's REST API directly via `requests` rather than the
    `voyageai` SDK package: the SDK's current release transitively depends
    on `langchain-core`/`langchain-text-splitters` (confirmed via `uv
    tree` when the SDK was briefly added and immediately reverted) --
    unacceptable for a project that deliberately avoids any RAG framework
    dependency (no LangChain, no LlamaIndex, no framework abstraction),
    even a transitive, non-direct-import one. `requests` is already a
    project dependency (corpus/build_corpus.py). Endpoint/schema verified
    against docs.voyageai.com/reference/embeddings-api on 2026-08-12, not
    exhaustively confirmed against a real call (no VOYAGE_API_KEY has been
    available yet) -- still needs confirmation once a key exists, same
    unverified-until-tested status as gpt-5.4-mini's temperature-parameter
    behavior in recipes/llm.py.
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
    """Production adapter for A2's bge-large-en-v1.5 leg. No API key
    needed -- local model via sentence-transformers (already a project
    dependency for pattern 04's cross-encoder reranker). Real, working,
    zero-cost -- mirrors CrossEncoderReranker's (recipes/rerank.py)
    zero-API-key precedent.
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
    """Factory respecting the RAG_RECIPES_LLM env var (mock in CI, openai for
    a real run -- see .env.example)."""
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
