"""evals/retrieval_leaderboard.py end to end on a fixture corpus with the
mock embedder and reranker: every row is produced, the table renders, and
--check's comparison ignores the latency column."""

import json
import time

import pytest

from evals.retrieval_leaderboard import (
    README_BLOCKS,
    README_CATEGORY_END,
    README_CATEGORY_START,
    README_END,
    README_START,
    build_retrievers,
    category_table,
    category_table_only,
    readme_table,
    render,
    run,
    score,
    strip_latency,
    table_only,
    write_readme_table,
)
from recipes.embeddings import MockEmbedder
from recipes.rerank import MockReranker

CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
    "c4": {"chunk_id": "c4", "text": "Gradient descent optimizes neural network weights."},
}
QA = [
    {"qid": "q1", "question": "speculative decoding inference", "relevant_chunk_ids": ["c1"]},
    {"qid": "q2", "question": "image classification with convolutional networks", "relevant_chunk_ids": ["c2", "c4"]},
]


def test_every_pattern_row_is_scored():
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    assert list(retrievers) == ["01_naive_dense", "02_bm25", "03_hybrid_rrf", "04_bm25_rerank", "03+04_hybrid_rerank"]
    row = score("02_bm25", retrievers["02_bm25"], QA)
    # BM25 ranks c1 first for q1 and c2 first for q2; c4 shares no terms
    # with q2, so recall@10 for q2 is 0.5 and the mean is 0.75.
    assert row.hit_at_3.mean == 1.0
    assert row.recall_at_10.mean == 0.75
    assert row.p50_latency_ms >= 0.0
    assert [r["first_relevant_rank"] for r in row.per_question] == [1, 1]
    assert row.per_question[1]["recall_at_10"] == 0.5


def test_render_and_strip_latency():
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    rows = [score(name, fn, QA) for name, fn in retrievers.items()]
    md = render(rows, len(QA), len(CORPUS), "mock", "mock")
    assert table_only(md).count("\n| ") == len(rows)  # one line per row after the header
    assert "p50_ms" in md
    stripped = strip_latency(md)
    assert "p50_ms" not in stripped
    # Different latencies, same scores: the comparable part must be equal.
    rows2 = [score(name, fn, QA) for name, fn in retrievers.items()]
    for r in rows2:
        r.p50_latency_ms += 123.4
    assert strip_latency(render(rows2, len(QA), len(CORPUS), "mock", "mock")) == stripped


def test_readme_table_round_trip(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n\n{README_START}\nold table\n{README_END}\n\noutro\n", encoding="utf-8")
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    rows = [score(name, fn, QA) for name, fn in retrievers.items()]
    md = render(rows, len(QA), len(CORPUS), "mock", "mock")

    write_readme_table(readme, table_only(md))

    text = readme.read_text(encoding="utf-8")
    assert text.startswith("intro\n\n") and text.endswith("\n\noutro\n")
    assert "old table" not in text
    assert readme_table(text) == table_only(md)
    assert readme_table("no markers here") is None


def test_run_on_files(tmp_path):
    corpus_path = tmp_path / "corpus.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    corpus_path.write_text("\n".join(json.dumps(c) for c in CORPUS.values()) + "\n", encoding="utf-8")
    qa_path.write_text("\n".join(json.dumps(q) for q in QA) + "\n", encoding="utf-8")
    md, rows = run(corpus_path, qa_path, embedder=MockEmbedder(), reranker=MockReranker())
    assert len(rows) == 5
    assert md.startswith("# Retrieval leaderboard")
    assert "2 questions, 4 chunks" in md


class _CountingReranker(MockReranker):
    def __init__(self):
        self.scored_pairs: list[tuple[str, str]] = []

    def rerank(self, query, candidates, top_k):
        self.scored_pairs += [(query, cid) for cid, _ in candidates]
        return super().rerank(query, candidates, top_k)


def test_rerank_rows_score_each_pair_once_and_keep_the_reranker_order():
    counting = _CountingReranker()
    retrievers = build_retrievers(CORPUS, MockEmbedder(), counting, "mock")
    question = QA[1]["question"]
    candidates = retrievers["03_hybrid_rrf"](question, 20)
    pairs = [(cid, CORPUS[cid]["text"]) for cid in candidates]
    expected = [r.chunk_id for r in MockReranker().rerank(question, pairs, top_k=3)]

    retrievers["04_bm25_rerank"](question, 3)
    assert retrievers["03+04_hybrid_rerank"](question, 3) == expected
    retrievers["03+04_hybrid_rerank"](question, 3)

    assert len(counting.scored_pairs) == len(set(counting.scored_pairs)) == len(candidates)


class _SlowReranker(MockReranker):
    SECONDS_PER_PAIR = 0.02

    def rerank(self, query, candidates, top_k):
        time.sleep(self.SECONDS_PER_PAIR * len(candidates))
        return super().rerank(query, candidates, top_k)


def test_reused_pairs_are_charged_to_the_rows_latency(tmp_path):
    # Hybrid offers every chunk as a candidate (4 per question); some of
    # those pairs were already scored by the BM25 + rerank row. The hybrid
    # row must still be charged for scoring all 4, not just the new ones.
    corpus_path = tmp_path / "corpus.jsonl"
    qa_path = tmp_path / "qa.jsonl"
    corpus_path.write_text("\n".join(json.dumps(c) for c in CORPUS.values()) + "\n", encoding="utf-8")
    qa_path.write_text("\n".join(json.dumps(q) for q in QA) + "\n", encoding="utf-8")

    _, rows = run(corpus_path, qa_path, embedder=MockEmbedder(), reranker=_SlowReranker())

    hybrid_rerank = next(r for r in rows if r.name == "03+04_hybrid_rerank")
    assert hybrid_rerank.p50_latency_ms >= len(CORPUS) * _SlowReranker.SECONDS_PER_PAIR * 1000


def test_p50_latency_uses_the_shared_percentile_definition(monkeypatch):
    # Four questions taking 1, 2, 3 and 4 ms: the median of an even count is
    # the mean of the two middle values (2.5), as evals/metrics.py reports
    # p50 everywhere else, not the upper middle value (3).
    ticks = iter([0.0, 0.001, 1.0, 1.002, 2.0, 2.003, 3.0, 3.004])
    monkeypatch.setattr("evals.retrieval_leaderboard.time.perf_counter", lambda: next(ticks))
    qa = [{"qid": f"q{i}", "question": f"q{i}", "relevant_chunk_ids": ["c1"]} for i in range(4)]

    row = score("fixed", lambda q, k: ["c1"], qa)

    assert row.p50_latency_ms == pytest.approx(2.5)


def test_render_shows_mrr_vs_top_with_ties_marked():
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    rows = [score(name, fn, QA) for name, fn in retrievers.items()]
    md = render(rows, len(QA), len(CORPUS), "mock", "mock")
    body = [line for line in table_only(md).splitlines() if line.startswith("| ") and not line.startswith("| rank")]

    assert "mrr_vs_top" in md
    assert body[0].split(" | ")[6] == "--"
    for line in body[1:]:
        cell = line.split(" | ")[6]
        assert cell.startswith(("+", "-"))
        # Two questions cannot separate any pattern from the top one.
        assert cell.endswith("(tie)")
    # The comparable part still drops only the latency column.
    assert "mrr_vs_top" in strip_latency(md) and "p50_ms" not in strip_latency(md)


QA_WITH_CATEGORIES = [
    {"qid": "q1", "category": "keyword", "question": "speculative decoding inference", "relevant_chunk_ids": ["c1"]},
    {"qid": "q2", "category": "keyword", "question": "gradient descent weights", "relevant_chunk_ids": ["c4"]},
    {"qid": "q3", "category": "paraphrase", "question": "pictures sorted by machines", "relevant_chunk_ids": ["c2"]},
]


def test_category_table_groups_questions_by_category():
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    row = score("02_bm25", retrievers["02_bm25"], QA_WITH_CATEGORIES)

    lines = category_table([row])

    # Known categories come first, in the fixed order, with their question counts.
    assert lines[0] == "| pattern | keyword (2) | paraphrase (1) |"
    # BM25 ranks both keyword targets first; the paraphrase shares no term with its target.
    assert lines[2] == "| 02_bm25 | 1.000 / 1.000 | 0.000 / 0.000 |"


def test_rendered_leaderboard_has_both_tables_and_keeps_the_category_table_comparable():
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    rows = [score(name, fn, QA_WITH_CATEGORIES) for name, fn in retrievers.items()]
    md = render(rows, len(QA_WITH_CATEGORIES), len(CORPUS), "mock", "mock")

    assert "## By question category" in md
    assert table_only(md).startswith("| rank |")
    assert category_table_only(md).startswith("| pattern | keyword (2) | paraphrase (1) |")
    # Only the main table loses its latency column; the category table's last
    # column (a category) must survive the comparison.
    stripped = strip_latency(md)
    assert "p50_ms" not in stripped
    assert category_table_only(md) in stripped


def test_readme_round_trip_for_both_tables(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(
        f"intro\n{README_START}\nold main\n{README_END}\nmiddle\n"
        f"{README_CATEGORY_START}\nold categories\n{README_CATEGORY_END}\noutro\n",
        encoding="utf-8",
    )
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    rows = [score(name, fn, QA_WITH_CATEGORIES) for name, fn in retrievers.items()]
    md = render(rows, len(QA_WITH_CATEGORIES), len(CORPUS), "mock", "mock")

    for start, end, fresh in README_BLOCKS:
        write_readme_table(readme, fresh(md), start, end)

    text = readme.read_text(encoding="utf-8")
    assert "old main" not in text and "old categories" not in text
    assert "\nmiddle\n" in text and text.endswith("outro\n")
    assert readme_table(text) == table_only(md)
    assert readme_table(text, README_CATEGORY_START, README_CATEGORY_END) == category_table_only(md)


def test_category_mrr_weighted_by_question_count_equals_the_main_mrr():
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    rows = [score(name, fn, QA_WITH_CATEGORIES) for name, fn in retrievers.items()]
    header, _, *body = category_table(rows)

    def cells(line):
        return [cell.strip() for cell in line.strip("|").split("|")][1:]

    counts = [int(h.rsplit("(", 1)[1].rstrip(")")) for h in cells(header)]
    for row, line in zip(rows, body):
        mrrs = [float(cell.split(" / ")[0]) for cell in cells(line)]
        weighted = sum(m * n for m, n in zip(mrrs, counts)) / sum(counts)
        assert weighted == pytest.approx(row.mrr.mean, abs=1e-3)


def test_category_table_shows_a_dash_for_a_row_without_that_category():
    retrievers = build_retrievers(CORPUS, MockEmbedder(), MockReranker(), "mock")
    full = score("02_bm25", retrievers["02_bm25"], QA_WITH_CATEGORIES)
    partial = score("01_naive_dense", retrievers["01_naive_dense"], QA_WITH_CATEGORIES[:2])

    lines = category_table([full, partial])

    assert lines[3] == "| 01_naive_dense | " + lines[3].split(" | ")[1] + " | -- |"
