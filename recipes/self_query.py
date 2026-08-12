"""Pattern 08: metadata + self-query. Extracts a structured filter
({"category": ..., "year": ...}) from the question via an LLM call, using
`prompts/self_query_prompt.txt` (auxiliary, NOT covered by R4 -- same
reasoning as hyde_prompt.txt/multi_query_prompt.txt: this runs BEFORE
retrieval, it's an input to what gets searched, not the generation step).
Reports the extracted filter on AnswerWithCitations.extracted_filter, which
evals/run.py's existing filter_accuracy() call already consumes -- no
changes needed there, that plumbing has been in place since P1 and was
simply unused until this pattern populates the field.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

from recipes import AnswerWithCitations
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import Embedder
from recipes.llm import LLM

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"
SELF_QUERY_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "self_query_prompt.txt"


def _parse_filter(text: str) -> dict:
    """Defensive parse, same convention as multi_query.py's
    _parse_subqueries: any failure (bad JSON, non-dict result) degrades to
    {} (= search the whole corpus), never raises.
    """
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
) -> Callable[..., AnswerWithCitations]:
    store = build_dense_index(corpus_by_id, embedder, embedding_model)
    filter_template = SELF_QUERY_PROMPT_PATH.read_text(encoding="utf-8")
    gen_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")
    all_ids = list(corpus_by_id.keys())

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()

        filter_prompt = filter_template.format(question=question)
        filter_response = llm.complete(prompt=filter_prompt, model=generation_model, temperature=0.0)
        extracted_filter = _parse_filter(filter_response.text)

        # Rank the WHOLE corpus, then filter by metadata, then slice to
        # top-k -- one code path whether or not a filter was extracted (an
        # empty filter matches everything). Simple and correct at this
        # corpus scale; a production self-query system would push the
        # filter into the vector DB's WHERE clause instead.
        ranked_ids = dense_search(store, embedder, embedding_model, question, k=len(all_ids))
        if extracted_filter:
            ranked_ids = [
                cid
                for cid in ranked_ids
                if all(corpus_by_id[cid].get(field) == value for field, value in extracted_filter.items())
            ]
        retrieved_ids = ranked_ids[:k]

        context = "\n\n".join(
            f"[{cid}] {corpus_by_id[cid]['text']}" for cid in retrieved_ids if cid in corpus_by_id
        )
        final_prompt = gen_template.format(context=context, question=question)
        final_response = llm.complete(prompt=final_prompt, model=generation_model, temperature=0.0)

        latency_ms = (time.perf_counter() - start) * 1000
        return AnswerWithCitations(
            answer=final_response.text,
            retrieved_chunk_ids=retrieved_ids,
            latency_ms=latency_ms,
            input_tokens=filter_response.input_tokens + final_response.input_tokens,
            output_tokens=filter_response.output_tokens + final_response.output_tokens,
            cached_input_tokens=filter_response.cached_input_tokens + final_response.cached_input_tokens,
            extracted_filter=extracted_filter,
        )

    return retrieve_and_answer
