"""Thin wrapper around rank_bm25 for keyword/sparse retrieval.

Also serves as pattern 02's recipe module (make_retrieve_and_answer below):
the low-level BM25Index infrastructure and the pattern-facing recipe
function both live here rather than splitting into a separate file, since
BM25 itself is the entire pattern -- no other pattern module needs to
import BM25Index directly.
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


# Common English function words, plus the question words queries start with.
_STOPWORD_TEXT = """
a about above after again against all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no nor
not of off on once only or other our ours out over own same she should so some such than that the
their theirs them then there these they this those through to too under until up very was we were
what when where which while who whom why will with would you your yours
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def fold_plural(token: str) -> str:
    """A light English plural folding: "studies" -> "study", "boxes" -> "box",
    "models" -> "model". Leaves short words, numbers, and words ending in
    -ss, -us, -is alone ("loss", "corpus", "analysis"). Applied to the
    index and the query alike, so a rare wrong fold still matches itself."""
    if len(token) <= 3 or not token.isalpha() or token.endswith(("ss", "us", "is")):
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("sses", "xes", "zes", "ches", "shes")):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


class BM25Index:
    """BM25 over lowercase alphanumeric tokens. `stopwords` drops those
    tokens from chunks and queries; `fold_plurals` maps plural forms to
    their singular (see fold_plural). Both are off by default."""

    def __init__(self, stopwords: frozenset[str] | None = None, fold_plurals: bool = False) -> None:
        self._chunk_ids: list[str] = []
        self._bm25 = None
        self._stopwords = stopwords or frozenset()
        self._fold_plurals = fold_plurals

    def _tokens(self, text: str) -> list[str]:
        tokens = [t for t in _tokenize(text) if t not in self._stopwords]
        return [fold_plural(t) for t in tokens] if self._fold_plurals else tokens

    def build(self, chunk_ids: list[str], texts: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        self._chunk_ids = list(chunk_ids)
        tokenized = [self._tokens(t) for t in texts]
        self._token_sets = [set(tokens) for tokens in tokenized]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[BM25Result]:
        """Top-k chunks by BM25 score, best first. Chunks sharing no term
        with the query are excluded, so a query with zero lexical overlap
        returns fewer than k results, possibly none. Returning arbitrary
        non-matching chunks instead would hand them full RRF / reranker
        credit downstream in the hybrid and rerank patterns.

        The test is term overlap, not a positive score: rank_bm25 gives a
        term in half or more of the chunks an IDF of zero or below, so a
        chunk that does match can score <= 0, most often in small corpora.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Index.build() must be called before search()")
        query_tokens = self._tokens(query)
        query_set = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            (
                (cid, float(s))
                for cid, s, tokens in zip(self._chunk_ids, scores, self._token_sets)
                if tokens & query_set
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [BM25Result(chunk_id=cid, score=s) for cid, s in ranked[:k]]


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
            cache_creation_input_tokens=response.cache_creation_input_tokens,
        )

    return retrieve_and_answer
