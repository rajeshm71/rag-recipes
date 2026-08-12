"""Pattern 09: multi-hop retrieval. Unlike pattern 06 (multi-query, which
fires several sub-queries in PARALLEL from one upfront decomposition), this
pattern is SEQUENTIAL: after each search, an LLM call decides whether
another, different search is needed before enough context exists to answer
-- each hop's query depends on what the PREVIOUS hop found. Bounded to
MAX_HOPS to keep cost predictable (roughly hops+1 LLM calls per question).
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
MULTI_HOP_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "multi_hop_prompt.txt"
MAX_HOPS = 2
PER_HOP_K = 5


def _parse_next_query(text: str) -> str | None:
    """Defensive parse, same convention as multi_query.py: any failure
    (bad JSON, missing key) degrades to None (= stop hopping), never raises
    and never loops forever on malformed output.
    """
    try:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.lower().startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        parsed = json.loads(stripped)
        next_query = parsed.get("next_query")
        return str(next_query) if next_query else None
    except Exception:
        return None


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
    max_hops: int = MAX_HOPS,
    per_hop_k: int = PER_HOP_K,
) -> Callable[..., AnswerWithCitations]:
    store = build_dense_index(corpus_by_id, embedder, embedding_model)
    hop_template = MULTI_HOP_PROMPT_PATH.read_text(encoding="utf-8")
    gen_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()
        accumulated_ids: list[str] = []
        total_input = total_output = total_cached = 0
        current_query = question

        for hop in range(max_hops):
            for cid in dense_search(store, embedder, embedding_model, current_query, per_hop_k):
                if cid not in accumulated_ids:
                    accumulated_ids.append(cid)

            if hop == max_hops - 1:
                break  # last hop already searched; no point asking for another

            context_so_far = "\n\n".join(
                f"[{cid}] {corpus_by_id[cid]['text']}" for cid in accumulated_ids if cid in corpus_by_id
            )
            hop_prompt = hop_template.format(question=question, context_so_far=context_so_far)
            hop_response = llm.complete(prompt=hop_prompt, model=generation_model, temperature=0.0)
            total_input += hop_response.input_tokens
            total_output += hop_response.output_tokens
            total_cached += hop_response.cached_input_tokens

            next_query = _parse_next_query(hop_response.text)
            if next_query is None:
                break
            current_query = next_query

        retrieved_ids = accumulated_ids[:k]
        context = "\n\n".join(
            f"[{cid}] {corpus_by_id[cid]['text']}" for cid in retrieved_ids if cid in corpus_by_id
        )
        final_prompt = gen_template.format(context=context, question=question)
        final_response = llm.complete(prompt=final_prompt, model=generation_model, temperature=0.0)
        total_input += final_response.input_tokens
        total_output += final_response.output_tokens
        total_cached += final_response.cached_input_tokens

        latency_ms = (time.perf_counter() - start) * 1000
        return AnswerWithCitations(
            answer=final_response.text,
            retrieved_chunk_ids=retrieved_ids,
            latency_ms=latency_ms,
            input_tokens=total_input,
            output_tokens=total_output,
            cached_input_tokens=total_cached,
        )

    return retrieve_and_answer
