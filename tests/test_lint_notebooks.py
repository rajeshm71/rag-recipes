import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lint_notebooks import (
    NOTEBOOKS_DIR,
    extract_headers,
    find_pattern_notebooks,
    lint_notebook,
)

GOOD_SECTIONS = [
    (1, "What this pattern does"),
    (2, "When to use it"),
    (3, "When NOT to use it"),
    (4, "Implementation"),
    (5, "Run on our eval set"),
    (6, "Example query walkthrough"),
    (7, "Where this pattern FAILS"),
    (8, "Copy-paste snippet"),
]


def _write_fixture_notebook(tmp_path: Path, name: str, sections: list[tuple[int, str]]) -> Path:
    cells = [
        {
            "cell_type": "markdown",
            "id": "abc123",
            "metadata": {},
            "source": [f"## {n}. {title}\n"],
        }
        for n, title in sections
    ]
    nb = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = tmp_path / name
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def test_find_pattern_notebooks_includes_01_through_10_excludes_others(tmp_path):
    for name in [
        "00_baseline_no_rag.ipynb",
        "00b_long_context_baseline.ipynb",
        "01_naive_dense.ipynb",
        "09_multi_hop.ipynb",
        "10_agentic.ipynb",
        "11_leaderboard.ipynb",
        "A1_chunking_study.ipynb",
        "A2_embedding_swap.ipynb",
    ]:
        (tmp_path / name).write_text("{}", encoding="utf-8")

    found = {p.name for p in find_pattern_notebooks(tmp_path)}
    assert found == {"01_naive_dense.ipynb", "09_multi_hop.ipynb", "10_agentic.ipynb"}


def test_extract_headers_parses_markdown_cells(tmp_path):
    path = _write_fixture_notebook(tmp_path, "fixture.ipynb", GOOD_SECTIONS)
    headers = extract_headers(path)
    assert headers == GOOD_SECTIONS


def test_lint_notebook_passes_on_all_8_correct_sections(tmp_path):
    path = _write_fixture_notebook(tmp_path, "fixture.ipynb", GOOD_SECTIONS)
    assert lint_notebook(path) == []


def test_lint_notebook_fails_on_missing_section(tmp_path):
    sections = [s for s in GOOD_SECTIONS if s[0] != 7]  # drop "Where this pattern FAILS"
    path = _write_fixture_notebook(tmp_path, "fixture.ipynb", sections)
    problems = lint_notebook(path)
    assert any("missing section 7" in p for p in problems)


def test_lint_notebook_fails_on_out_of_order_sections(tmp_path):
    sections = GOOD_SECTIONS.copy()
    sections[0], sections[1] = sections[1], sections[0]  # swap sections 1 and 2
    path = _write_fixture_notebook(tmp_path, "fixture.ipynb", sections)
    problems = lint_notebook(path)
    assert any("out of order" in p for p in problems)


def test_lint_notebook_fails_on_duplicate_section_number(tmp_path):
    # Regression test found during plan review: a naive dict-from-headers
    # build would silently keep only the LAST "## 4." header, masking a
    # real authoring bug (two sections both numbered 4).
    sections = GOOD_SECTIONS + [(4, "Implementation (duplicate)")]
    path = _write_fixture_notebook(tmp_path, "fixture.ipynb", sections)
    problems = lint_notebook(path)
    assert any("duplicate section number" in p for p in problems)


def test_lint_notebook_fails_on_wrong_keyword(tmp_path):
    sections = GOOD_SECTIONS.copy()
    sections[2] = (3, "Something unrelated")  # section 3 should mention "when not to use"
    path = _write_fixture_notebook(tmp_path, "fixture.ipynb", sections)
    problems = lint_notebook(path)
    assert any("doesn't contain expected keyword" in p and "3" in p for p in problems)


def test_real_pattern_notebooks_all_pass():
    # The check that matters: the linter must actually validate the real,
    # already-shipped 01-10 notebooks, not just internally-consistent
    # fixtures.
    notebooks = find_pattern_notebooks(NOTEBOOKS_DIR)
    assert len(notebooks) == 10, f"expected 10 pattern notebooks, found {len(notebooks)}"
    for path in notebooks:
        problems = lint_notebook(path)
        assert problems == [], f"{path.name}: {problems}"
