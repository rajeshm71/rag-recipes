"""Retrieval-only leaderboard: every retrieval pattern that needs no LLM,
scored on the eval set with local models. No API key, no network beyond
the one-time model download, reproducible on any machine:

    uv run python evals/retrieval_leaderboard.py            # write outputs/retrieval_leaderboard.md + README tables
    uv run python evals/retrieval_leaderboard.py --check    # fail if the output file or a README table differs
    uv run python evals/retrieval_leaderboard.py --per-question out.jsonl   # also dump per-question ranks

Rows: naive dense, BM25, hybrid (RRF), BM25 + cross-encoder rerank, and
hybrid + rerank. Metrics: hit@3, hit@10, MRR, recall@10 (fraction of a
question's relevant chunks in the top 10), each with a 95% bootstrap CI,
plus each row's paired MRR difference from the top row and p50 latency.
A second table breaks MRR and hit@10 down by question category. Both
tables are written to the output file and to their own marker blocks in
the README. Latency depends on hardware and is excluded from --check.

Models: BAAI/bge-small-en-v1.5 embeddings (local, 33M parameters) and
BAAI/bge-reranker-v2-m3 (local). RAG_RECIPES_LLM=mock swaps both for the
hash-based mock backends so the script itself can run in a unit test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.metrics import (
    ConfidenceInterval,
    bootstrap_ci,
    hit_at_k,
    latency_percentiles,
    mrr,
    paired_difference_ci,
    recall_at_k,
)
from evals.run import load_corpus_by_id, load_qa_set
from recipes.bm25 import BM25Index
from recipes.dense_index import build_dense_index, dense_search
from recipes.embeddings import Embedder, LocalEmbedder, MockEmbedder
from recipes.hybrid import rrf_fuse
from recipes.rerank import CrossEncoderReranker, MockReranker, Reranker

LOCAL_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CANDIDATES = 20
DEFAULT_OUTPUT = ROOT / "outputs" / "retrieval_leaderboard.md"
README = ROOT / "README.md"
README_START = "<!-- retrieval-leaderboard:start -->"
README_END = "<!-- retrieval-leaderboard:end -->"
README_CATEGORY_START = "<!-- retrieval-by-category:start -->"
README_CATEGORY_END = "<!-- retrieval-by-category:end -->"
# Column order of the per-category table; categories not listed follow, sorted.
CATEGORY_ORDER = ("keyword", "paraphrase", "multi_hop", "filter")

Retriever = Callable[[str, int], list[str]]


@dataclass
class Row:
    name: str
    hit_at_3: ConfidenceInterval
    hit_at_10: ConfidenceInterval
    mrr: ConfidenceInterval
    recall_at_10: ConfidenceInterval
    p50_latency_ms: float
    # One record per question: qid, category, rank of the first relevant
    # chunk (None if outside the top k), recall@10, and the top-10 ids.
    per_question: list[dict]
    # Per-question reciprocal ranks in eval-set order, for paired comparisons
    # between rows scored on the same questions.
    reciprocal_ranks: list[float] = field(default_factory=list)


class PairScoreCache:
    """Cross-encoder scores keyed by (question, chunk_id). Both rerank rows
    score mostly the same candidates and the cross-encoder dominates the
    script's runtime, so each pair is scored once. Each pair also keeps the
    time it took to score (its batch's time split evenly), and a reused
    pair charges that time to the caller through `take_reused_ms()`, so a
    row's latency still reflects scoring all of its candidates."""

    def __init__(self, reranker: Reranker, corpus_by_id: dict[str, dict]) -> None:
        self._reranker = reranker
        self._corpus_by_id = corpus_by_id
        self._scores: dict[tuple[str, str], float] = {}
        self._ms: dict[tuple[str, str], float] = {}
        self._reused_ms = 0.0

    def rerank(self, q: str, candidates: list[str], k: int) -> list[str]:
        missing = [cid for cid in candidates if (q, cid) not in self._scores]
        self._reused_ms += sum(self._ms[(q, cid)] for cid in candidates if (q, cid) in self._scores)
        if missing:
            pairs = [(cid, self._corpus_by_id[cid]["text"]) for cid in missing]
            start = time.perf_counter()
            results = self._reranker.rerank(q, pairs, top_k=len(pairs))
            per_pair_ms = (time.perf_counter() - start) * 1000 / len(pairs)
            for r in results:
                self._scores[(q, r.chunk_id)] = r.score
                self._ms[(q, r.chunk_id)] = per_pair_ms
        # Stable sort: ties keep candidate order, as Reranker.rerank does.
        return sorted(candidates, key=lambda cid: self._scores[(q, cid)], reverse=True)[:k]

    def take_reused_ms(self) -> float:
        """Scoring time of the pairs reused since the last call."""
        reused, self._reused_ms = self._reused_ms, 0.0
        return reused


def build_retrievers(
    corpus_by_id: dict[str, dict],
    embedder: Embedder,
    reranker: Reranker,
    embedding_model: str,
    pair_cache: PairScoreCache | None = None,
) -> dict[str, Retriever]:
    pair_cache = pair_cache or PairScoreCache(reranker, corpus_by_id)
    chunk_ids = list(corpus_by_id.keys())
    store = build_dense_index(corpus_by_id, embedder, embedding_model)
    bm25 = BM25Index()
    bm25.build(chunk_ids, [corpus_by_id[cid]["text"] for cid in chunk_ids])

    def dense(q: str, k: int) -> list[str]:
        return dense_search(store, embedder, embedding_model, q, k)

    def bm25_only(q: str, k: int) -> list[str]:
        return [r.chunk_id for r in bm25.search(q, k)]

    def hybrid(q: str, k: int) -> list[str]:
        return rrf_fuse([dense(q, CANDIDATES), bm25_only(q, CANDIDATES)], k)

    def bm25_rerank(q: str, k: int) -> list[str]:
        return pair_cache.rerank(q, bm25_only(q, CANDIDATES), k)

    def hybrid_rerank(q: str, k: int) -> list[str]:
        return pair_cache.rerank(q, hybrid(q, CANDIDATES), k)

    return {
        "01_naive_dense": dense,
        "02_bm25": bm25_only,
        "03_hybrid_rrf": hybrid,
        "04_bm25_rerank": bm25_rerank,
        "03+04_hybrid_rerank": hybrid_rerank,
    }


def score(
    name: str,
    retrieve: Retriever,
    qa_set: list[dict],
    k: int = 10,
    extra_latency_ms: Callable[[], float] | None = None,
) -> Row:
    """`extra_latency_ms`, if given, is called after each retrieval and its
    result added to that question's latency (cached work charged back)."""
    h3, h10, rr, rec, lat = [], [], [], [], []
    per_question = []
    for record in qa_set:
        relevant = record["relevant_chunk_ids"]
        start = time.perf_counter()
        ids = retrieve(record["question"], k)
        lat.append((time.perf_counter() - start) * 1000 + (extra_latency_ms() if extra_latency_ms else 0.0))
        h3.append(hit_at_k(ids, relevant, 3))
        h10.append(hit_at_k(ids, relevant, 10))
        rr.append(mrr(ids, relevant))
        rec.append(recall_at_k(ids, relevant, 10))
        first_rank = next((i for i, cid in enumerate(ids, start=1) if cid in set(relevant)), None)
        per_question.append(
            {
                "pattern": name,
                "qid": record.get("qid"),
                "category": record.get("category"),
                "first_relevant_rank": first_rank,
                "recall_at_10": rec[-1],
                "relevant_chunk_ids": relevant,
                "retrieved": ids[:k],
            }
        )
    return Row(
        name=name,
        hit_at_3=bootstrap_ci(h3),
        hit_at_10=bootstrap_ci(h10),
        mrr=bootstrap_ci(rr),
        recall_at_10=bootstrap_ci(rec),
        p50_latency_ms=latency_percentiles(lat)[0],
        per_question=per_question,
        reciprocal_ranks=rr,
    )


def _ci(ci: ConfidenceInterval) -> str:
    return f"{ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]"


def _vs_top(row: Row, top: Row) -> str:
    """MRR difference from the top row with its paired bootstrap interval,
    marked "(tie)" when the interval contains 0."""
    if row is top:
        return "--"
    ci = paired_difference_ci(row.reciprocal_ranks, top.reciprocal_ranks)
    tie = " (tie)" if ci.lower <= 0 <= ci.upper else ""
    return f"{ci.mean:+.3f} [{ci.lower:+.3f}, {ci.upper:+.3f}]{tie}"


def render(rows: list[Row], n_questions: int, n_chunks: int, embedding_model: str, reranker_model: str) -> str:
    ranked = sorted(rows, key=lambda r: (r.mrr.mean, r.recall_at_10.mean, r.hit_at_3.mean), reverse=True)
    lines = [
        "# Retrieval leaderboard",
        "",
        (
            f"{n_questions} questions, {n_chunks} chunks, k=10. Embeddings: `{embedding_model}`. "
            f"Reranker: `{reranker_model}`. Ranked by MRR, then recall@10, then hit@3. "
            "95% bootstrap confidence intervals (1000 resamples, seed 42). mrr_vs_top is each "
            "row's MRR minus the top row's, bootstrapped over per-question differences; "
            '"(tie)" marks an interval that contains 0.'
        ),
        "",
        "| rank | pattern | hit@3 | hit@10 | mrr | recall@10 | mrr_vs_top | p50_ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    top = ranked[0] if ranked else None
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r.name} | {_ci(r.hit_at_3)} | {_ci(r.hit_at_10)} | {_ci(r.mrr)} | "
            f"{_ci(r.recall_at_10)} | {_vs_top(r, top)} | {r.p50_latency_ms:.1f} |"
        )
    lines += [
        "",
        "## By question category",
        "",
        "MRR / hit@10 for each question category; the header gives the number of questions.",
        "",
        *category_table(ranked),
    ]
    return "\n".join(lines) + "\n"


def category_table(ranked: list[Row]) -> list[str]:
    """Markdown lines of a per-category table: one row per pattern in the
    given order, one column per question category, cells "MRR / hit@10"
    from each row's per-question first relevant rank."""
    def category(q: dict) -> str:
        return q.get("category") or "uncategorized"

    seen = {category(q) for r in ranked for q in r.per_question}
    categories = [c for c in CATEGORY_ORDER if c in seen] + sorted(seen - set(CATEGORY_ORDER))
    first = ranked[0].per_question if ranked else []
    counts = {c: sum(1 for q in first if category(q) == c) for c in categories}
    lines = [
        "| pattern | " + " | ".join(f"{c} ({counts[c]})" for c in categories) + " |",
        "|---|" + "---|" * len(categories),
    ]
    for r in ranked:
        cells = []
        for c in categories:
            ranks = [q["first_relevant_rank"] for q in r.per_question if category(q) == c]
            if not ranks:
                cells.append("--")
                continue
            mean_rr = sum(1.0 / rank for rank in ranks if rank) / len(ranks)
            hit_10 = sum(1 for rank in ranks if rank and rank <= 10) / len(ranks)
            cells.append(f"{mean_rr:.3f} / {hit_10:.3f}")
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
    return lines


def _tables(markdown: str) -> list[list[str]]:
    """The markdown tables in `markdown`, each a list of its `|` lines."""
    tables: list[list[str]] = []
    previous_was_table = False
    for line in markdown.splitlines():
        is_table = line.startswith("|")
        if is_table and not previous_was_table:
            tables.append([])
        if is_table:
            tables[-1].append(line)
        previous_was_table = is_table
    return tables


def strip_latency(markdown: str) -> str:
    """The comparable part of a rendered leaderboard: everything except the
    p50_ms column, which depends on hardware. Only tables whose header has a
    p50_ms column lose their last column."""
    out = []
    previous_was_table = False
    strip = False
    for line in markdown.splitlines():
        is_table = line.startswith("|")
        if is_table and not previous_was_table:
            strip = "p50_ms" in line
        if is_table and strip:
            line = "|".join(line.split("|")[:-2]) + "|"
        out.append(line)
        previous_was_table = is_table
    return "\n".join(out)


def table_only(markdown: str) -> str:
    """The main leaderboard table (header + body) of a rendered leaderboard."""
    tables = _tables(markdown)
    return "\n".join(tables[0]) if tables else ""


def category_table_only(markdown: str) -> str:
    """The per-category table of a rendered leaderboard."""
    tables = _tables(markdown)
    return "\n".join(tables[1]) if len(tables) > 1 else ""


def readme_table(readme_text: str, start_marker: str = README_START, end_marker: str = README_END) -> str | None:
    """The table currently embedded in the README between the markers, or
    None if the markers are absent."""
    if start_marker not in readme_text or end_marker not in readme_text:
        return None
    start = readme_text.index(start_marker) + len(start_marker)
    end = readme_text.index(end_marker)
    return readme_text[start:end].strip()


def write_readme_table(
    readme_path: Path, table: str, start_marker: str = README_START, end_marker: str = README_END
) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if start_marker not in text or end_marker not in text:
        return
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker)
    readme_path.write_text(text[:start] + "\n" + table + "\n" + text[end:], encoding="utf-8")


# README marker pairs and the table each one holds.
README_BLOCKS = (
    (README_START, README_END, table_only),
    (README_CATEGORY_START, README_CATEGORY_END, category_table_only),
)


def run(
    corpus_path: Path = ROOT / "corpus" / "corpus.jsonl",
    qa_path: Path = ROOT / "evals" / "qa_set.jsonl",
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
) -> tuple[str, list[Row]]:
    mock = os.environ.get("RAG_RECIPES_LLM", "openai").lower() == "mock"
    if embedder is None:
        embedder = MockEmbedder() if mock else LocalEmbedder(LOCAL_EMBEDDING_MODEL)
    if reranker is None:
        reranker = MockReranker() if mock else CrossEncoderReranker()
    embedding_model = "mock" if isinstance(embedder, MockEmbedder) else LOCAL_EMBEDDING_MODEL
    reranker_model = "mock" if isinstance(reranker, MockReranker) else "BAAI/bge-reranker-v2-m3"

    corpus_by_id = load_corpus_by_id(corpus_path)
    qa_set = load_qa_set(qa_path)
    pair_cache = PairScoreCache(reranker, corpus_by_id)
    retrievers = build_retrievers(corpus_by_id, embedder, reranker, embedding_model, pair_cache)
    rows = [
        score(name, fn, qa_set, extra_latency_ms=pair_cache.take_reused_ms) for name, fn in retrievers.items()
    ]
    return render(rows, len(qa_set), len(corpus_by_id), embedding_model, reranker_model), rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check", action="store_true", help="compare against the committed table instead of writing it"
    )
    parser.add_argument("--per-question", type=Path, default=None, help="also write per-question ranks (JSONL)")
    args = parser.parse_args()

    markdown, rows = run()
    print(markdown)
    if args.per_question is not None:
        with open(args.per_question, "w", encoding="utf-8") as f:
            for row in rows:
                f.writelines(json.dumps(record) + "\n" for record in row.per_question)
        print(f"wrote {args.per_question}")
    if args.check:
        committed = args.output.read_text(encoding="utf-8")
        if strip_latency(committed) != strip_latency(markdown):
            print(f"MISMATCH: {args.output} differs from a fresh run (latency column ignored).", file=sys.stderr)
            sys.exit(1)
        readme_text = README.read_text(encoding="utf-8") if README.exists() else ""
        for start, end, fresh in README_BLOCKS:
            embedded = readme_table(readme_text, start, end)
            if embedded is not None and strip_latency(embedded) != strip_latency(fresh(markdown)):
                print(f"MISMATCH: the table between {start} and {end} in {README} differs.", file=sys.stderr)
                sys.exit(1)
        print(f"OK: {args.output} and the README tables match a fresh run.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.output}")
    if README.exists():
        for start, end, fresh in README_BLOCKS:
            write_readme_table(README, fresh(markdown), start, end)
        print(f"updated the tables in {README}")


if __name__ == "__main__":
    main()
