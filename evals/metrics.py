"""Deterministic metrics for scoring a RAG pattern against the eval set.

All randomness (bootstrap resampling) is seeded to 42 for reproducible
results across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 1000
CONFIDENCE_LEVEL = 0.95


@dataclass
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1.0 if any relevant_id appears in the top-k retrieved_ids, else 0.0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal rank of the first relevant id in retrieved_ids, else 0.0."""
    relevant = set(relevant_ids)
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant_ids that appear in the top-k retrieved_ids. Unlike
    hit@k, a multi-chunk question only scores 1.0 when every relevant chunk
    was retrieved."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def paper_hit_at_k(
    retrieved_ids: list[str], relevant_paper_ids: set[str], corpus_by_id: dict[str, dict], k: int
) -> float:
    """Coarser than hit_at_k: 1.0 if ANY of the top-k retrieved chunks
    belongs to one of the relevant PAPERS (not necessarily the exact
    ground-truth CHUNK). Needed for the chunking-strategy and embedding-swap
    appendix studies: re-chunking the corpus produces entirely different
    chunk_ids, so qa_set.jsonl's relevant_chunk_ids can't be matched exactly
    across chunking variants -- paper_id is the one thing stable across
    every strategy.
    """
    top_k_papers = {corpus_by_id[cid]["paper_id"] for cid in retrieved_ids[:k] if cid in corpus_by_id}
    return 1.0 if top_k_papers & relevant_paper_ids else 0.0


def filter_accuracy(extracted_filter: dict | None, requires_filter: dict | None) -> float | None:
    """For pattern 08 (self-query): 1.0 if the extracted filter exactly
    matches the ground-truth filter, else 0.0.

    Returns None (excluded from the aggregate) in two cases: the question
    doesn't require a filter, or the pattern made no filter claim at all
    (`extracted_filter is None`, the default every pattern except 08 leaves
    it at). The second case is "not applicable", not a wrong answer: scoring
    it 0.0 would print a measured-looking 0.000 for patterns that never try
    to filter. A pattern that DID try and extracted nothing reports {} and
    is scored 0.0.

    Comparison is exact dict equality, including value types. Pattern 08's
    parser normalizes types (year to int) before reporting, so a mismatch
    here is a genuinely wrong filter.
    """
    if requires_filter is None or extracted_filter is None:
        return None
    return 1.0 if extracted_filter == requires_filter else 0.0


def bootstrap_ci(
    values: list[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> ConfidenceInterval:
    """95% bootstrap confidence interval over a list of per-question scores.

    Reported for every leaderboard metric, since point estimates on 60 (or
    fewer, pilot-scale) samples are easy to over-interpret.
    """
    if not values:
        return ConfidenceInterval(mean=0.0, lower=0.0, upper=0.0)

    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    resample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        resample_means[i] = sample.mean()

    alpha = 1.0 - confidence
    lower = float(np.quantile(resample_means, alpha / 2))
    upper = float(np.quantile(resample_means, 1 - alpha / 2))
    return ConfidenceInterval(mean=float(arr.mean()), lower=lower, upper=upper)


def paired_difference_ci(
    values: list[float],
    baseline: list[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> ConfidenceInterval:
    """Bootstrap interval for mean(values) - mean(baseline) when both were
    scored on the same questions, in the same order. Resampling the
    per-question differences keeps the pairing, which two separate
    per-row intervals ignore: two patterns can have overlapping intervals
    and still differ reliably on the same questions, and vice versa.
    """
    if len(values) != len(baseline):
        raise ValueError("paired scores must cover the same questions")
    differences = [v - b for v, b in zip(values, baseline)]
    return bootstrap_ci(differences, n_resamples=n_resamples, confidence=confidence, seed=seed)


def latency_percentiles(latencies_ms: list[float]) -> tuple[float, float]:
    """Returns (p50, p95) wall-clock latency in milliseconds."""
    if not latencies_ms:
        return (0.0, 0.0)
    arr = np.asarray(latencies_ms, dtype=float)
    return (float(np.percentile(arr, 50)), float(np.percentile(arr, 95)))
