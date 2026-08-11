"""Pattern 01: naive dense retrieval.

Embed the question, embed every corpus chunk, retrieve the top-k nearest by
cosine/L2 distance (via recipes/store.py's sqlite-vec wrapper), stuff the
retrieved chunks into the held-constant generation prompt (R4), and ask the
LLM to answer.

Baseline pattern. Fails on rare terms / exact identifiers that dense
embeddings don't represent distinctly (see 01_naive_dense.ipynb's "Where
this pattern FAILS" section for real, corpus-grounded examples).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from recipes import AnswerWithCitations
from recipes.embeddings import Embedder
from recipes.llm import LLM
from recipes.store import VectorStore

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
) -> Callable[..., AnswerWithCitations]:
    """Builds the dense index once (batch-embedding every chunk in
    `corpus_by_id`) and returns a bound `retrieve_and_answer(question, k)`
    closure matching the §16.3 contract.
    """
    chunk_ids = list(corpus_by_id.keys())
    texts = [corpus_by_id[cid]["text"] for cid in chunk_ids]

    embed_response = embedder.embed(texts, model=embedding_model)
    # Vector dimension is read from the real embedding output, not assumed
    # -- MockEmbedder defaults to 16 dims, text-embedding-3-small produces
    # 1536, and hardcoding either would silently break with the other.
    dim = len(embed_response.vectors[0]) if embed_response.vectors else 0
    store = VectorStore(dim=dim)
    store.add_many(chunk_ids, embed_response.vectors)

    prompt_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()

        query_embed = embedder.embed([question], model=embedding_model)
        results = store.search(query_embed.vectors[0], k=k)
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
