"""The 6 chunking strategies A1_chunking_study.ipynb compares, holding
retrieval (hybrid+rerank, recipes/hybrid_rerank.py) constant. Each function
takes one paper's full text (from corpus/corpus_fulltext.jsonl) and returns
a list of chunk dicts: {"text": ..., "embed_text": ...} -- embed_text
differs from text only for the late-chunking approximation (see
chunk_late's docstring); every other strategy has embed_text == text.
"""

from __future__ import annotations

import re

from corpus.build_corpus import chunk_text as _fixed_chunk_text

SEMANTIC_BREAKPOINT_PERCENTILE = 20
SEMANTIC_MIN_CHUNK_TOKENS = 100
SEMANTIC_MAX_CHUNK_TOKENS = 1024
LATE_CHUNKING_CONTEXT_CHARS = 400  # ~100 tokens at ~4 chars/token


def chunk_fixed(full_text: str, chunk_tokens: int, tokenizer) -> list[dict]:
    return [
        {"text": t, "embed_text": t}
        for t in _fixed_chunk_text(full_text, tokenizer, chunk_tokens=chunk_tokens)
    ]


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def chunk_semantic(full_text: str, embedder, embedding_model: str, tokenizer) -> list[dict]:
    """Splits into sentences, embeds each, and breaks wherever consecutive-
    sentence cosine similarity is unusually low (bottom
    SEMANTIC_BREAKPOINT_PERCENTILE of observed drops) -- a standard,
    simple semantic-chunking heuristic. Enforces a min/max token floor/cap
    via the fixed-token splitter as a fallback for oversized merges.
    """
    import numpy as np

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(full_text) if s.strip()]
    if len(sentences) <= 1:
        return [{"text": full_text, "embed_text": full_text}]

    vectors = np.array(embedder.embed(sentences, model=embedding_model).vectors)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.clip(norms, 1e-9, None)
    sims = np.sum(unit[:-1] * unit[1:], axis=1)  # consecutive cosine similarities
    threshold = np.percentile(sims, SEMANTIC_BREAKPOINT_PERCENTILE)
    breakpoints = set(int(i) for i in np.where(sims < threshold)[0])  # break AFTER sentence i

    groups: list[list[str]] = [[]]
    for i, sentence in enumerate(sentences):
        groups[-1].append(sentence)
        if i in breakpoints:
            groups.append([])
    groups = [g for g in groups if g]

    chunks: list[dict] = []
    for group in groups:
        text = " ".join(group)
        n_tokens = len(tokenizer.encode(text))
        if n_tokens > SEMANTIC_MAX_CHUNK_TOKENS:
            # Fallback: split an oversized semantic group at the token
            # level, same mechanism build_corpus.py already uses for
            # oversized paragraphs.
            for piece in _fixed_chunk_text(text, tokenizer, chunk_tokens=SEMANTIC_MAX_CHUNK_TOKENS):
                chunks.append({"text": piece, "embed_text": piece})
        elif n_tokens < SEMANTIC_MIN_CHUNK_TOKENS and chunks:
            # Merge an undersized trailing group into the previous chunk
            # rather than shipping a degenerate near-empty chunk.
            chunks[-1]["text"] += " " + text
            chunks[-1]["embed_text"] = chunks[-1]["text"]
        else:
            chunks.append({"text": text, "embed_text": text})
    return chunks


def chunk_document_aware(chunks_for_paper: list[dict]) -> list[dict]:
    """Groups the ALREADY-CHUNKED original corpus.jsonl entries for one
    paper by their existing `section` tag, concatenating same-section text.
    Operates on corpus.jsonl (not full_text) since section labels only
    exist there.
    """
    by_section: dict[str, list[str]] = {}
    for chunk in sorted(chunks_for_paper, key=lambda c: c["chunk_index"]):
        by_section.setdefault(chunk["section"], []).append(chunk["text"])
    return [{"text": "\n\n".join(texts), "embed_text": "\n\n".join(texts)} for texts in by_section.values()]


def chunk_late(full_text: str, chunk_tokens: int, tokenizer) -> list[dict]:
    """Context-window approximation of late chunking: each chunk's citable
    `text` is its own fixed-size span, but `embed_text` includes a window
    of surrounding CHARACTERS on each side, so the embedding call sees
    more of the document than the chunk alone -- NOT the literal
    token-embed-then-pool algorithm (Günther et al.), which needs
    token-level embedding access no provider used in this project exposes
    through its public API. Uses the same get_embedder() call as every
    other variant, so there's no embedding-model confound.

    Uses a character-offset search (str.find), not token-position
    arithmetic: _fixed_chunk_text's chunks OVERLAP by design (64-token
    overlap, same as the main corpus), so naively accumulating "cursor +=
    span_len" between chunks drifts out of sync with each chunk's true
    position after the first overlap. Searching for each chunk's own text
    directly in full_text sidesteps that entirely and is robust regardless
    of overlap.
    """
    base_texts = _fixed_chunk_text(full_text, tokenizer, chunk_tokens=chunk_tokens)

    chunks: list[dict] = []
    search_from = 0
    for text in base_texts:
        pos = full_text.find(text, search_from)
        if pos == -1:
            # Chunk text was reconstructed via tokenizer decode/re-encode
            # and no longer matches full_text byte-for-byte (rare, e.g.
            # whitespace normalization) -- fall back to no extra context
            # rather than guessing a wrong window.
            chunks.append({"text": text, "embed_text": text})
            continue
        window_start = max(0, pos - LATE_CHUNKING_CONTEXT_CHARS)
        window_end = min(len(full_text), pos + len(text) + LATE_CHUNKING_CONTEXT_CHARS)
        chunks.append({"text": text, "embed_text": full_text[window_start:window_end]})
        search_from = pos + 1  # allow the next (overlapping) chunk to match forward from here
    return chunks
