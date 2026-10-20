"""Pattern 07: Contextual retrieval (Anthropic). One-time indexing-cost
preprocessing: before embedding, prepend a short LLM-generated context blurb
to each chunk, situating it within its full source document. The blurb
generation call uses claude-sonnet-5, with the paper's full text sent as a
cached system-prompt prefix. On this corpus a paper is roughly 4,500 to
9,500 tokens, above the model's 1,024-token minimum cacheable prefix, so
generating blurbs for every chunk of the same paper is cheap after the
first. A model with a higher minimum, or shorter documents, would silently
skip caching and pay full price for every call. Retrieval and final-answer generation are unchanged from pattern 01
(dense over the now-contextualized chunk text) -- this pattern's novelty is
entirely in what gets embedded, not in the retrieval or generation mechanism.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from pathlib import Path

from recipes import AnswerWithCitations
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import Embedder
from recipes.llm import LLM, UnusableResponseError

EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"  # final answer, held constant across every pattern
CONTEXTUALIZE_MODEL = "claude-sonnet-5"  # Anthropic, preprocessing step ONLY
_GENERATION_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "generation_prompt.txt"
_DOC_PREFIX_PATH = Path(__file__).resolve().parent.parent / "prompts" / "contextualize_document_prefix.txt"
_CHUNK_PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "contextualize_chunk_prompt.txt"


def _contextualize_corpus(
    corpus_by_id: dict[str, dict],
    anthropic_llm: LLM,
    model: str,
    cost_log: list[dict] | None,
) -> dict[str, dict]:
    """Groups chunks by paper_id, calls anthropic_llm once per chunk with
    the paper's full text as a cacheable system prefix, and returns a NEW
    corpus_by_id dict whose "text" field is `{blurb}\n\n{original_text}`.
    Does not mutate the input dict.

    If `cost_log` is a list, one dict is appended per LLM call with its
    real token breakdown, so the caller (the 07 notebook) can compute a
    with/without-caching cost comparison afterward. A chunk whose call
    returns no usable text (refused or cut off) keeps its original text, is
    logged with "contextualized": False, and is named in one warning.
    """
    doc_prefix_template = _DOC_PREFIX_PATH.read_text(encoding="utf-8")
    chunk_prompt_template = _CHUNK_PROMPT_PATH.read_text(encoding="utf-8")

    by_paper: dict[str, list[str]] = {}
    for cid, chunk in corpus_by_id.items():
        by_paper.setdefault(chunk["paper_id"], []).append(cid)

    contextualized: dict[str, dict] = {}
    skipped: list[str] = []
    for paper_id, chunk_ids in by_paper.items():
        document_text = "\n\n".join(corpus_by_id[cid]["text"] for cid in chunk_ids)
        cacheable_prefix = doc_prefix_template.format(document_text=document_text)

        for cid in chunk_ids:
            chunk = corpus_by_id[cid]
            prompt = chunk_prompt_template.format(chunk_text=chunk["text"])
            try:
                response = anthropic_llm.complete(
                    prompt=prompt, model=model, temperature=0.0, cacheable_prefix=cacheable_prefix
                )
                usable = True
            except UnusableResponseError as exc:
                # One refused or truncated blurb should not throw away every
                # other (already billed) call: this chunk keeps its original
                # text and the call is still counted.
                response = exc.response
                usable = False
                skipped.append(cid)
            if cost_log is not None:
                cost_log.append(
                    {
                        "chunk_id": cid,
                        "paper_id": paper_id,
                        "input_tokens": response.input_tokens,
                        "cached_input_tokens": response.cached_input_tokens,
                        "cache_creation_input_tokens": response.cache_creation_input_tokens,
                        "output_tokens": response.output_tokens,
                        "contextualized": usable,
                    }
                )
            text = f"{response.text.strip()}\n\n{chunk['text']}" if usable else chunk["text"]
            contextualized[cid] = {**chunk, "text": text}

    if skipped:
        warnings.warn(
            f"{len(skipped)} of {len(corpus_by_id)} chunks kept their original text because the "
            f"contextualization call returned no usable text: {', '.join(skipped)}",
            stacklevel=2,
        )
    return contextualized


def make_retrieve_and_answer(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    llm: LLM,
    anthropic_llm: LLM,
    embedding_model: str = EMBEDDING_MODEL,
    generation_model: str = GENERATION_MODEL,
    contextualize_model: str = CONTEXTUALIZE_MODEL,
    cost_log: list[dict] | None = None,
) -> Callable[..., AnswerWithCitations]:
    contextualized_corpus = _contextualize_corpus(corpus_by_id, anthropic_llm, contextualize_model, cost_log)
    store = build_dense_index(contextualized_corpus, embedder, embedding_model)
    prompt_template = _GENERATION_PROMPT_PATH.read_text(encoding="utf-8")

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations:
        start = time.perf_counter()
        retrieved_ids = dense_search(store, embedder, embedding_model, question, k)
        # Context for the final answer uses the ORIGINAL corpus_by_id text
        # (no blurb) -- the blurb's only job was to improve embedding
        # quality; showing it to the answering LLM would just be noise.
        context = "\n\n".join(
            f"[{cid}] {corpus_by_id[cid]['text']}" for cid in retrieved_ids if cid in corpus_by_id
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
