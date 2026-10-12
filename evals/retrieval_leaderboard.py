"""Retrieval-only leaderboard: every retrieval pattern that needs no LLM,
scored on the eval set with local models. No API key, no network beyond
the one-time model download, reproducible on any machine:

    uv run python evals/retrieval_leaderboard.py            # write outputs/retrieval_leaderboard.md + README table
    uv run python evals/retrieval_leaderboard.py --check    # fail if either committed table differs
    uv run python evals/retrieval_leaderboard.py --per-question out.jsonl   # also dump per-question ranks

Rows: naive dense, BM25, hybrid (RRF), BM25 + cross-encoder rerank, and
hybrid + rerank. Metrics: hit@3, hit@10, MRR, recall@10 (fraction of a
question's relevant chunks in the top 10), each with a 95% bootstrap CI,
plus p50 latency. Latency depends on hardware and is excluded from --check.

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
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.metrics import ConfidenceInterval, bootstrap_ci, hit_at_k, mrr, recall_at_k
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
    lat_sorted = sorted(lat)
    return Row(
        name=name,
        hit_at_3=bootstrap_ci(h3),
        hit_at_10=bootstrap_ci(h10),
        mrr=bootstrap_ci(rr),
        recall_at_10=bootstrap_ci(rec),
        p50_latency_ms=lat_sorted[len(lat_sorted) // 2] if lat_sorted else 0.0,
        per_question=per_question,
    )


def _ci(ci: ConfidenceInterval) -> str:
    return f"{ci.mean:.3f} [{ci.lower:.3f}, {ci.upper:.3f}]"


def render(rows: list[Row], n_questions: int, n_chunks: int, embedding_model: str, reranker_model: str) -> str:
    ranked = sorted(rows, key=lambda r: (r.mrr.mean, r.recall_at_10.mean, r.hit_at_3.mean), reverse=True)
    lines = [
        "# Retrieval leaderboard",
        "",
        (
            f"{n_questions} questions, {n_chunks} chunks, k=10. Embeddings: `{embedding_model}`. "
            f"Reranker: `{reranker_model}`. Ranked by MRR, then recall@10, then hit@3. "
            "95% bootstrap confidence intervals (1000 resamples, seed 42)."
        ),
        "",
        "| rank | pattern | hit@3 | hit@10 | mrr | recall@10 | p50_ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"| {i} | {r.name} | {_ci(r.hit_at_3)} | {_ci(r.hit_at_10)} | {_ci(r.mrr)} | "
            f"{_ci(r.recall_at_10)} | {r.p50_latency_ms:.1f} |"
        )
    return "\n".join(lines) + "\n"


def strip_latency(markdown: str) -> str:
    """The comparable part of a rendered table: every column except p50_ms."""
    out = []
    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = line.split("|")
            line = "|".join(cells[:-2]) + "|"
        out.append(line)
    return "\n".join(out)


def table_only(markdown: str) -> str:
    """Just the markdown table rows (header + body) of a rendered leaderboard."""
    return "\n".join(line for line in markdown.splitlines() if line.startswith("|"))


def readme_table(readme_text: str) -> str | None:
    """The table currently embedded in the README between the markers, or
    None if the markers are absent."""
    if README_START not in readme_text or README_END not in readme_text:
        return None
    start = readme_text.index(README_START) + len(README_START)
    end = readme_text.index(README_END)
    return readme_text[start:end].strip()


def write_readme_table(readme_path: Path, table: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        return
    start = text.index(README_START) + len(README_START)
    end = text.index(README_END)
    readme_path.write_text(text[:start] + "\n" + table + "\n" + text[end:], encoding="utf-8")


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
        embedded = readme_table(README.read_text(encoding="utf-8")) if README.exists() else None
        if embedded is not None and strip_latency(embedded) != strip_latency(table_only(markdown)):
            print(f"MISMATCH: the table in {README} differs from a fresh run.", file=sys.stderr)
            sys.exit(1)
        print(f"OK: {args.output} and the README table match a fresh run.")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"wrote {args.output}")
    if README.exists():
        write_readme_table(README, table_only(markdown))
        print(f"updated the table in {README}")


if __name__ == "__main__":
    main()
