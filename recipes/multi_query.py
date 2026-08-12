"""Pattern 06: multi-query decomposition. Rewrite the question into up to 3
sub-queries, retrieve for each independently, merge by best rank. Helps
ambiguous queries; roughly doubles retrieval + one extra LLM call's cost.

NOTE on R4 scope: `prompts/generation_prompt.txt` (R4, held constant) is
used ONLY for this pattern's final answer-generation call. The decomposition
prompt (`prompts/multi_query_prompt.txt`) is an auxiliary, pattern-specific
prompt that runs BEFORE retrieval -- it is an input to what gets retrieved,
not "the generation step," so R4 does not apply to it.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

from recipes import AnswerWithCitations
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import Embedder
from recipes.llm import LLM

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"
MULTI_QUERY_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "multi_query_prompt.txt"
PER_SUBQUERY_K = 10


def _parse_subqueries(text: str, fallback_question: str) -> list[str]:
    """Defensive parse: any failure (bad JSON, missing/empty 'queries')
    degrades gracefully to [fallback_question] rather than raising. The key
    claim of this pattern is "helps ambiguous queries," not "may crash on
    bad JSON."
    """
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        parsed = json.loads(stripped)
        queries = parsed.get("queries")
        if not queries or not isinstance(queries, list):
            raise ValueError("empty or missing 'queries'")
        return [str(q) for q in queries]
    except Exception:
        return [fallback_question]


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
    per_subquery_k: int = PER_SUBQUERY_K,
) -> Callable[..., AnswerWithCitations]:
    store = build_dense_index(corpus_by_id, embedder, embedding_model)
    decompose_template = MULTI_QUERY_PROMPT_PATH.read_text(encoding="utf-8")
    gen_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()

        decompose_prompt = decompose_template.format(question=question)
        decompose_response = llm.complete(
            prompt=decompose_prompt, model=generation_model, temperature=0.0
        )
        subqueries = _parse_subqueries(decompose_response.text, fallback_question=question)

        best_rank: dict[str, int] = {}
        for sub_q in subqueries:
            ids = dense_search(store, embedder, embedding_model, sub_q, per_subquery_k)
            for rank, cid in enumerate(ids, start=1):
                if cid not in best_rank or rank < best_rank[cid]:
                    best_rank[cid] = rank
        retrieved_ids = sorted(best_rank, key=best_rank.get)[:k]

        context = "\n\n".join(
            f"[{cid}] {corpus_by_id[cid]['text']}"
            for cid in retrieved_ids
            if cid in corpus_by_id
        )
        final_prompt = gen_template.format(context=context, question=question)
        final_response = llm.complete(prompt=final_prompt, model=generation_model, temperature=0.0)

        latency_ms = (time.perf_counter() - start) * 1000

        return AnswerWithCitations(
            answer=final_response.text,
            retrieved_chunk_ids=retrieved_ids,
            latency_ms=latency_ms,
            input_tokens=decompose_response.input_tokens + final_response.input_tokens,
            output_tokens=decompose_response.output_tokens + final_response.output_tokens,
            cached_input_tokens=decompose_response.cached_input_tokens
            + final_response.cached_input_tokens,
            cache_creation_input_tokens=decompose_response.cache_creation_input_tokens
            + final_response.cache_creation_input_tokens,
        )

    return retrieve_and_answer
