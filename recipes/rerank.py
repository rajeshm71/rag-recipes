"""Reranker abstraction (mirrors LLM/Embedder's Real+Mock shape) plus
pattern 04: cross-encoder rerank on top of BM25 candidates. Single best
add-on to a base retriever -- costs latency, not API money (the reranker
is a local model, no API key needed at all).
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from recipes import AnswerWithCitations
from recipes.bm25 import BM25Index
from recipes.llm import LLM

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
CANDIDATES_K = 20
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"


@dataclass
class RerankResult:
    chunk_id: str
    score: float


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int
    ) -> list[RerankResult]: ...


class CrossEncoderReranker:
    """Production adapter. Lazy-imports sentence_transformers so MockReranker-
    only test runs never need it importable (same convention as OpenAILLM).
    """

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int
    ) -> list[RerankResult]:
        pairs = [(query, text) for _, text in candidates]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        return [RerankResult(chunk_id=cid, score=float(s)) for (cid, _), s in ranked[:top_k]]


class MockReranker:
    """Deterministic, hash-based (same technique as MockEmbedder) so mock
    execution genuinely reorders candidates, not a no-op passthrough.
    """

    def rerank(
        self, query: str, candidates: list[tuple[str, str]], top_k: int
    ) -> list[RerankResult]:
        scored = []
        for cid, text in candidates:
            digest = hashlib.sha256((query + "|" + text).encode("utf-8")).digest()
            # 4 bytes, not 1, to keep collisions negligible across a
            # 20-candidate list (matches MockEmbedder's collision-avoidance
            # rigor rather than an 8-bit-only pseudo-score).
            pseudo_score = int.from_bytes(digest[:4], "big") / 2**32
            scored.append((cid, pseudo_score))
        ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
        return [RerankResult(chunk_id=cid, score=s) for cid, s in ranked[:top_k]]


def get_reranker() -> Reranker:
    backend = os.environ.get("RAG_RECIPES_LLM", "openai").lower()
    return MockReranker() if backend == "mock" else CrossEncoderReranker()


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    llm: LLM,
    reranker: Reranker | None = None,
    generation_model: str = GENERATION_MODEL,
    candidates_k: int = CANDIDATES_K,
) -> Callable[..., AnswerWithCitations]:
    chunk_ids = list(corpus_by_id.keys())
    texts = [corpus_by_id[cid]["text"] for cid in chunk_ids]
    bm25_index = BM25Index()
    bm25_index.build(chunk_ids, texts)
    reranker = reranker or get_reranker()

    prompt_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()

        candidates = bm25_index.search(question, candidates_k)
        candidate_pairs = [(r.chunk_id, corpus_by_id[r.chunk_id]["text"]) for r in candidates]
        reranked = reranker.rerank(question, candidate_pairs, top_k=k)
        retrieved_ids = [r.chunk_id for r in reranked]

        context = "\n\n".join(
            f"[{cid}] {corpus_by_id[cid]['text']}"
            for cid in retrieved_ids
            if cid in corpus_by_id
        )
        prompt = prompt_template.format(context=context, question=question)

        response = llm.complete(prompt=prompt, model=generation_model, temperature=0.0)
        latency_ms = (time.perf_counter() - start) * 1000

        return AnswerWithCitations(
            answer=response.text,
            retrieved_chunk_ids=retrieved_ids,
            latency_ms=latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cached_input_tokens=response.cached_input_tokens,
            cache_creation_input_tokens=response.cache_creation_input_tokens,
        )

    return retrieve_and_answer
