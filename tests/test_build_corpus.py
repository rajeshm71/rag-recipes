"""Regression coverage for corpus/build_corpus.py's chunking logic.

corpus/build_corpus.py is not part of the recipes/evals package (it's a
one-off script gated behind the corpus-build optional dependency group),
so it's imported directly by path here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corpus"))

from build_corpus import CHUNK_TOKENS, chunk_text


class _FakeTokenizer:
    """Whitespace tokenizer so these tests don't need the real tiktoken
    model download, and so token counts are trivially predictable.
    """

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def test_chunk_text_respects_token_budget_across_paragraph_boundaries():
    tokenizer = _FakeTokenizer()
    paragraphs = [f"word{i}" for i in range(2000)]
    text = "\n\n".join(paragraphs)

    chunks = chunk_text(text, tokenizer)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(tokenizer.encode(chunk)) <= CHUNK_TOKENS


def test_chunk_text_splits_a_single_oversized_paragraph():
    # Regression test: PDF extraction often loses blank-line breaks, so an
    # entire page can come back as ONE paragraph with no internal breaks.
    # Before the fix, chunk_text never split within a paragraph, so a
    # single 2000-word "paragraph" produced one chunk 4x over budget.
    tokenizer = _FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(2000))  # one giant paragraph

    chunks = chunk_text(text, tokenizer)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(tokenizer.encode(chunk)) <= CHUNK_TOKENS


def test_chunk_text_small_input_stays_one_chunk():
    tokenizer = _FakeTokenizer()
    text = "This is a short paper.\n\nIt has two small paragraphs."
    chunks = chunk_text(text, tokenizer)
    assert len(chunks) == 1
