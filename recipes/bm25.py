"""Thin wrapper around rank_bm25 for keyword/sparse retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass


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
