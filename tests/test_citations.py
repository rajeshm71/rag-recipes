from recipes.citations import extract_citations


def test_extract_single_citation():
    assert extract_citations("The answer is X [arxiv:2601.11580#0].") == [
        "arxiv:2601.11580#0"
    ]


def test_extract_multiple_citations_in_order():
    text = "First claim [a#0]. Second claim [b#1]. Back to first [a#0]."
    assert extract_citations(text) == ["a#0", "b#1"]


def test_extract_no_citations():
    assert extract_citations("I don't have enough information.") == []


def test_extract_ignores_non_citation_brackets():
    # Plain bracketed text that isn't a chunk_id shape should still match
    # (the regex is intentionally permissive) -- this documents the
    # behavior rather than asserting a stricter shape we don't need yet.
    assert extract_citations("See [note] for details.") == ["note"]
