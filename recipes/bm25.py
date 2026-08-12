"""Thin wrapper around rank_bm25 for keyword/sparse retrieval.

Also serves as pattern 02's recipe module (make_retrieve_and_answer below):
SPEC.md's file tree (§9) lists a single recipes/bm25.py serving both roles
(the low-level BM25Index infrastructure from P1, and the pattern-facing
recipe function), so both live here rather than splitting into a file the
spec's tree doesn't name.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from recipes import AnswerWithCitations
from recipes.llm import LLM


@dataclass
class BM25Result:
    chunk_id: str
    score: float


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._bm25 = None

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self._chunk_ids = list(chunk_ids)
        tokenized = [_tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[BM25Result]:
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(
            zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True
        )
        return [BM25Result(chunk_id=cid, score=float(s)) for cid, s in ranked[:k]]


GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    llm: LLM,
    generation_model: str = GENERATION_MODEL,
) -> Callable[..., AnswerWithCitations]:
    """Pattern 02: BM25 (keyword/lexical) retrieval. Builds the index once
    from `corpus_by_id` and returns a bound `retrieve_and_answer(question, k)`
    closure -- same shape as recipes/naive_dense.py's factory, so both
    patterns are drop-in interchangeable for evals.run_pattern().

    Purely lexical: needs no embedding model and no API call for retrieval
    itself, only for the generation step.
    """
    chunk_ids = list(corpus_by_id.keys())
    texts = [corpus_by_id[cid]["text"] for cid in chunk_ids]

    index = BM25Index()
    index.build(chunk_ids, texts)

    prompt_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()

        results = index.search(question, k=k)
        retrieved_ids = [r.chunk_id for r in results]

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
        )

    return retrieve_and_answer
