"""Pattern 08: metadata + self-query. Extracts a structured filter
({"category": ..., "year": ...}) from the question via an LLM call, using
`prompts/self_query_prompt.txt` (auxiliary, not the held-constant generation
prompt -- same reasoning as hyde_prompt.txt/multi_query_prompt.txt: this
runs BEFORE retrieval, it's an input to what gets searched, not the
generation step). Reports the extracted filter on
AnswerWithCitations.extracted_filter, which evals/run.py's existing
filter_accuracy() call already consumes -- that plumbing predates this
pattern and was simply unused until this pattern populates the field.
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
SELF_QUERY_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "self_query_prompt.txt"
# The only metadata fields a filter may constrain. Anything else the LLM
# emits is dropped rather than compared against a field the corpus doesn't
# have, which would silently match zero chunks.
FILTER_FIELDS = ("category", "year")
# sqlite-vec rejects KNN queries with k above this.
_MAX_KNN_K = 4096


def _coerce_filter_value(field: str, value: object) -> str | int | None:
    """Normalize one filter value to the type corpus.jsonl stores, or None
    if it can't be (a null, a list, a non-numeric year...). The comparison
    in retrieve_and_answer and evals/metrics.py's filter_accuracy are both
    exact equality, so {"year": "2024"} would otherwise never match an int
    2024 in the corpus.
    """
    if field == "year":
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None
    return value.strip() if isinstance(value, str) and value.strip() else None


def _parse_filter(text: str) -> dict:
    """Defensive parse, same convention as multi_query.py's
    _parse_subqueries: any failure (bad JSON, non-dict result) degrades to
    {} (= search the whole corpus), never raises. Unknown keys, nulls, and
    uncoercible values are dropped individually, so a plausible-but-wrong
    shape like {"category": "cs.LG", "year": null} still filters on
    category instead of matching nothing.
    """
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            return {}
        cleaned: dict = {}
        for field in FILTER_FIELDS:
            value = _coerce_filter_value(field, parsed.get(field))
            if value is not None:
                cleaned[field] = value
        return cleaned
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
    # Category values as the corpus spells them, keyed case-insensitively,
    # so "cs.lg" from the LLM filters on the corpus's "cs.LG".
    known_categories = {
        chunk["category"].lower(): chunk["category"]
        for chunk in corpus_by_id.values()
        if isinstance(chunk.get("category"), str)
    }

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()

        filter_prompt = filter_template.format(question=question)
        filter_response = llm.complete(prompt=filter_prompt, model=generation_model, temperature=0.0)
        extracted_filter = _parse_filter(filter_response.text)
        if "category" in extracted_filter:
            category = extracted_filter["category"]
            extracted_filter["category"] = known_categories.get(category.lower(), category)

        # Rank the WHOLE corpus, then filter by metadata, then slice to
        # top-k -- one code path whether or not a filter was extracted (an
        # empty filter matches everything). Simple and correct at this
        # corpus scale; a production self-query system would push the
        # filter into the vector DB's WHERE clause instead.
        ranked_ids = dense_search(
            store, embedder, embedding_model, question, k=min(len(all_ids), _MAX_KNN_K)
        )
        if extracted_filter:
            filtered_ids = [
                cid
                for cid in ranked_ids
                if all(corpus_by_id[cid].get(field) == value for field, value in extracted_filter.items())
            ]
            # A filter that matches nothing (a category or year the corpus
            # doesn't have) would leave the answer with no context at all;
            # the unfiltered ranking is the better fallback. extracted_filter
            # still reports the filter as extracted, so filter accuracy
            # scores it as the miss it is.
            if filtered_ids:
                ranked_ids = filtered_ids
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
            cache_creation_input_tokens=filter_response.cache_creation_input_tokens
            + final_response.cache_creation_input_tokens,
            extracted_filter=extracted_filter,
        )

    return retrieve_and_answer
