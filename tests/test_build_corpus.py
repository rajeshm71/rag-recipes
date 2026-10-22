"""Regression coverage for corpus/build_corpus.py's chunking logic.

corpus/build_corpus.py is not part of the recipes/evals package (it's a
one-off script gated behind the corpus-build optional dependency group),
so it's imported directly by path here.
"""

import sys
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corpus"))

from build_corpus import CHUNK_OVERLAP_TOKENS, CHUNK_TOKENS, chunk_text, parse_arxiv_categories


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


def test_every_chunk_after_the_first_starts_with_the_previous_chunks_last_64_tokens():
    tokenizer = _FakeTokenizer()
    # Paragraphs longer than the overlap: carrying "the whole last paragraph
    # if it is short enough" would carry nothing here.
    paragraphs = [" ".join(f"p{p}w{i}" for i in range(100)) for p in range(20)]
    chunks = chunk_text("\n\n".join(paragraphs), tokenizer)

    assert len(chunks) > 2
    for prev, cur in pairwise(chunks):
        assert cur.split()[:CHUNK_OVERLAP_TOKENS] == prev.split()[-CHUNK_OVERLAP_TOKENS:]


def test_split_paragraph_pieces_rejoin_without_inserted_breaks():
    tokenizer = _FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(2000))  # one giant paragraph

    chunks = chunk_text(text, tokenizer)

    for chunk in chunks:
        assert "\n" not in chunk
        assert chunk in text
    covered = " ".join(c.split(maxsplit=CHUNK_OVERLAP_TOKENS)[-1] if i else c for i, c in enumerate(chunks))
    assert covered == text


def test_fragments_fold_into_the_previous_chunk():
    tokenizer = _FakeTokenizer()
    big = " ".join(f"w{i}" for i in range(500))
    text = big + "\n\n(6)\n\n" + big + "\n\nshort tail"

    chunks = chunk_text(text, tokenizer)

    assert chunks[-1].endswith("short tail")
    for prev, cur in pairwise(chunks):
        new_tokens = len(cur.split()) - CHUNK_OVERLAP_TOKENS
        assert new_tokens >= CHUNK_OVERLAP_TOKENS, (prev[-40:], cur[:40])


class _ByteTokenizer:
    """One token per UTF-8 byte, so token boundaries can fall inside a
    multi-byte character, as they can with tiktoken."""

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")


def test_cuts_never_split_a_multibyte_character():
    tokenizer = _ByteTokenizer()
    text = "αβγδ " * 400  # 2-byte letters: half of all byte offsets are mid-character

    chunks = chunk_text(text, tokenizer, chunk_tokens=100, overlap_tokens=10)

    assert len(chunks) > 1
    assert all("\ufffd" not in c for c in chunks)
    assert all(c in text for c in chunks)


ATOM_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2601.00087v2</id>
    <arxiv:primary_category term="cs.RO" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.RO" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2601.00086v1</id>
    <arxiv:primary_category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


def test_parse_arxiv_categories_lists_every_category_primary_first():
    assert parse_arxiv_categories(ATOM_RESPONSE) == {
        "arxiv:2601.00087": ["cs.RO", "cs.LG", "cs.AI"],
        "arxiv:2601.00086": ["cs.CL"],
    }
