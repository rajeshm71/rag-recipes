"""Deterministic metrics for scoring a RAG pattern against the eval set.

All randomness (bootstrap resampling) is seeded to 42 per SPEC.md §11
(Reproducibility rules).
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


def filter_accuracy(extracted_filter: dict | None, requires_filter: dict | None) -> float | None:
    """For pattern 08 (self-query): 1.0 if the extracted filter exactly
    matches the ground-truth filter, else 0.0. Returns None if the question
    doesn't require a filter (so it can be excluded from the aggregate).

    NOTE (review #7, documented not fixed): this is exact dict equality,
    including value types -- {"year": "2024"} != {"year": 2024}. No
    consumer exists yet (pattern 08 self-query lands in P4), so adding
    type-coercion/normalization now would be speculative. Revisit this
    once pattern 08's actual filter-extraction output shape is known,
    rather than guessing a normalization scheme ahead of time.
    """
    if requires_filter is None:
        return None
    return 1.0 if extracted_filter == requires_filter else 0.0


def bootstrap_ci(
    values: list[float],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = BOOTSTRAP_SEED,
) -> ConfidenceInterval:
    """95% bootstrap confidence interval over a list of per-question scores.

    Required by SPEC.md §5 ("Statistical honesty") for every leaderboard
    metric, since point estimates on 60 (or fewer, pilot-scale) samples are
    easy to over-interpret.
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


def latency_percentiles(latencies_ms: list[float]) -> tuple[float, float]:
    """Returns (p50, p95) wall-clock latency in milliseconds."""
    if not latencies_ms:
        return (0.0, 0.0)
    arr = np.asarray(latencies_ms, dtype=float)
    return (float(np.percentile(arr, 50)), float(np.percentile(arr, 95)))
