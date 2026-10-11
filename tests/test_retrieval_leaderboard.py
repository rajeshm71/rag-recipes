"""evals/retrieval_leaderboard.py end to end on a fixture corpus with the
mock embedder and reranker: every row is produced, the table renders, and
--check's comparison ignores the latency column."""

import json

from evals.retrieval_leaderboard import (
    README_END,
    README_START,
    build_retrievers,
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
    assert md.count("\n| ") == 1 + len(rows)  # header + one line per row
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
