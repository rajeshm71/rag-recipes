"""Coverage for corpus/build_from_fulltext.py: heading detection, section
labels, the References cut, tail merging, and the per-paper cap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corpus"))

from build_from_fulltext import (
    body_text,
    chunk_paper,
    find_headings,
    normalize_section,
    section_for_offset,
)


class _FakeTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


PAPER = "\n".join(
    [
        "A Paper Title",
        "Jane Doe, John Smith",
        "Abstract",
        "We study a thing. " * 20,
        "",
        "1 Introduction",
        "Intro text. " * 300,
        "",
        "2 Method",
        "Method text. " * 300,
        "",
        "2.1 Rule Generation",
        "Sub-method text. " * 300,
        "",
        "3 Experiments",
        "Experiment text. " * 300,
        "",
        "4 Conclusion",
        "Conclusion text. " * 10,
        "",
        "References",
        "[1] Some reference. " * 30,
    ]
)


def test_find_headings_numbered_and_bare():
    titles = [t for _, t, _ in find_headings(PAPER)]
    assert titles == [
        "Abstract",
        "Introduction",
        "Method",
        "Rule Generation",
        "Experiments",
        "Conclusion",
        "References",
    ]


def test_find_headings_ignores_sentences_and_author_lines():
    # "Jane Doe, John Smith" is Title-case but not a conventional bare title
    # and not numbered, so it must not count; neither does a sentence.
    assert "Jane Doe, John Smith" not in [t for _, t, _ in find_headings(PAPER)]
    assert find_headings("This is a sentence.\nAnother one here.") == []


def test_normalize_section_vocabulary():
    assert normalize_section("Related Work") == "related_work"
    assert normalize_section("3.2 Datasets and Baselines"[4:]) == "experiments"
    assert normalize_section("Ablation Study") == "results"
    assert normalize_section("Limitations") == "discussion"
    assert normalize_section("Conclusions and Future Work") == "conclusion"
    assert normalize_section("Something Unusual") == "other"


def test_body_text_cuts_at_references():
    body, headings = body_text(PAPER)
    assert "Some reference" not in body
    assert "Conclusion text" in body
    assert [t for _, t, _ in headings][-1] == "Conclusion"


def test_body_text_keeps_an_early_references_mention():
    # A "References" line in the first 30% of a paper is not the
    # bibliography (e.g. a section discussing reference resolution).
    text = "References\nWe discuss reference resolution. " * 5 + "\n\n1 Introduction\n" + "Body. " * 50
    body, _ = body_text(text)
    assert "Body." in body


def test_section_for_offset_subheading_inherits_parent():
    headings = find_headings(PAPER)
    sub_offset = next(o for o, t, _ in headings if t == "Rule Generation")
    assert section_for_offset(sub_offset + 5, headings) == "methods"
    assert section_for_offset(0, headings) == "abstract"


def test_chunk_paper_labels_sections_and_respects_cap():
    chunks = chunk_paper(PAPER, _FakeTokenizer(), max_chunks=3)
    assert len(chunks) == 3
    assert chunks[0]["section"] == "abstract"
    assert all(c["n_tokens"] == len(c["text"].split()) for c in chunks)
    assert all("Some reference" not in c["text"] for c in chunks)


def test_chunk_paper_merges_tiny_fragments():
    # Two full-size paragraphs with a one-line equation between them, then a
    # 5-token tail: neither fragment may ship as its own chunk.
    big = " ".join(f"w{i}" for i in range(500))
    text = big + "\n\n(6)\n\n" + big + "\n\nshort tail here now ok"
    chunks = chunk_paper(text, _FakeTokenizer(), max_chunks=10)
    assert chunks[-1]["text"].endswith("short tail here now ok")
    assert any("(6)" in c["text"] for c in chunks)
    assert all(c["n_tokens"] >= 64 for c in chunks)
