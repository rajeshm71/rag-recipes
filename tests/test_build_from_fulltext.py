"""Coverage for corpus/build_from_fulltext.py: heading detection, section
labels, the References cut, tail merging, and the per-paper cap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corpus"))

import json

import build_from_fulltext
from build_from_fulltext import (
    body_text,
    build,
    chunk_paper,
    dehyphenate,
    find_headings,
    line_break_vocabulary,
    normalize_section,
    refresh_paper_categories,
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


OUTLINE_LINES = [
    "1 University of Southern California, Los Angeles, CA 90007, USA",
    "Abstract",
    "1 Introduction",
    "2 Reinforcement Learning",
    "2.1 Markov Decision Process",
    "1. Equality and Symmetry:",
    "Method",
    "199 Request ID",
    "3 Method",
    "3.1 Experiment Setup Details",
    "3.2 Something Unusual",
    "Limitations",
    "References",
]
OUTLINE_PAPER = "\n".join(OUTLINE_LINES)


def test_find_headings_keeps_only_the_paper_outline():
    titles = [t for _, t, _ in find_headings(OUTLINE_PAPER)]
    assert titles == [
        "Abstract",
        "Introduction",
        "Reinforcement Learning",
        "Markov Decision Process",
        "Method",
        "Experiment Setup Details",
        "Something Unusual",
        "Limitations",
        "References",
    ]
    # Rejected: the affiliation, the numbered list item inside section 2,
    # a bare "Method" table cell after numbering started, and the figure
    # label "199 Request ID".


def test_section_labels_follow_the_top_level_section():
    labels = {t: label for _, t, label in find_headings(OUTLINE_PAPER)}
    # An unrecognized top-level section is "other", not the previous label.
    assert labels["Reinforcement Learning"] == "other"
    # Its unrecognized sub-section stays "other" too.
    assert labels["Markov Decision Process"] == "other"
    # Sub-sections of a recognized section take that section's label.
    assert labels["Experiment Setup Details"] == "methods"
    assert labels["Something Unusual"] == "methods"
    assert labels["Limitations"] == "discussion"


def test_outline_tolerates_one_missed_heading_but_not_a_jump():
    text = "1 Introduction\n2 Method\n4 Experiments\n9 Figure Label"
    assert [t for _, t, _ in find_headings(text)] == ["Introduction", "Method", "Experiments"]


def test_normalize_section_ignores_spaces_lost_or_added_by_pdf_extraction():
    assert normalize_section("RELATEDWORKS") == "related_work"
    assert normalize_section("The Hybrid Agentic F ramework") == "methods"


def test_dehyphenate_rejoins_words_split_at_a_line_end():
    text = (
        "Candidates help.\nWe score can-\ndidates and orienta-\ntions, using self-\nsupervised "
        "Hu-\nBERT features and a self supervised loss."
    )
    vocabulary = line_break_vocabulary([text])
    # Fragments of split words do not count as words.
    assert "didates" not in vocabulary
    assert dehyphenate(text, vocabulary) == (
        "Candidates help.\nWe score candidates and orientations, using self-supervised "
        "Hu-BERT features and a self supervised loss."
    )


def test_dehyphenate_leaves_paragraph_breaks_and_plain_hyphens_alone():
    text = "A trade-off.\n\nNext para-\n\ngraph starts here."
    assert dehyphenate(text, line_break_vocabulary([text])) == text


def test_refresh_paper_categories_sets_primary_and_every_listing(tmp_path):
    papers = tmp_path / "papers.jsonl"
    papers.write_text(
        json.dumps({"paper_id": "arxiv:1", "title": "T", "authors": ["A"], "year": 2025, "category": "cs.LG"}) + "\n",
        encoding="utf-8",
    )

    refresh_paper_categories(papers, fetch=lambda ids: {"arxiv:1": ["cs.RO", "cs.LG"]})

    record = json.loads(papers.read_text(encoding="utf-8"))
    assert record["category"] == "cs.RO"
    assert record["categories"] == ["cs.RO", "cs.LG"]


def test_build_copies_every_category_into_each_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(build_from_fulltext, "_get_tokenizer", lambda: _FakeTokenizer())
    papers = tmp_path / "papers.jsonl"
    fulltext = tmp_path / "fulltext.jsonl"
    out = tmp_path / "corpus.jsonl"
    papers.write_text(
        json.dumps(
            {"paper_id": "arxiv:1", "title": "T", "authors": ["A"], "year": 2025,
             "category": "cs.RO", "categories": ["cs.RO", "cs.LG"]}
        ) + "\n",
        encoding="utf-8",
    )
    fulltext.write_text(json.dumps({"paper_id": "arxiv:1", "full_text": PAPER}) + "\n", encoding="utf-8")

    chunks = build(fulltext_path=fulltext, papers_path=papers, out_path=out)

    assert chunks and all(c["categories"] == ["cs.RO", "cs.LG"] for c in chunks)
    assert all(c["category"] == "cs.RO" for c in chunks)
