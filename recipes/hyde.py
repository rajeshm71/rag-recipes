"""Pattern 05: HyDE (Hypothetical Document Embeddings). Generate a
hypothetical answer FIRST, embed and search with THAT instead of the raw
question, then generate the real answer from what's retrieved. Two LLM
calls per query -- token accounting sums both, or cost is silently
understated.

NOTE on R4 scope: `prompts/generation_prompt.txt` (R4, held constant) is
used ONLY for this pattern's final answer-generation call. The hypothetical-
document prompt (`prompts/hyde_prompt.txt`) is an auxiliary, pattern-specific
prompt that runs BEFORE retrieval -- it is an input to what gets retrieved,
not "the generation step," so R4 does not apply to it.
"""

from __future__ import annotations

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
HYDE_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "hyde_prompt.txt"


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
) -> Callable[..., AnswerWithCitations]:
    store = build_dense_index(corpus_by_id, embedder, embedding_model)
    hyde_template = HYDE_PROMPT_PATH.read_text(encoding="utf-8")
    gen_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()

        hyde_prompt = hyde_template.format(question=question)
        hyde_response = llm.complete(prompt=hyde_prompt, model=generation_model, temperature=0.0)

        retrieved_ids = dense_search(store, embedder, embedding_model, hyde_response.text, k)
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
            # Sum tokens across BOTH LLM calls, not just the final one --
            # this pattern's real cost is 2 calls, and silently forgetting
            # to sum understates it.
            input_tokens=hyde_response.input_tokens + final_response.input_tokens,
            output_tokens=hyde_response.output_tokens + final_response.output_tokens,
            cached_input_tokens=hyde_response.cached_input_tokens + final_response.cached_input_tokens,
        )

    return retrieve_and_answer
