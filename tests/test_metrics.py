from evals.metrics import (
    bootstrap_ci,
    filter_accuracy,
    hit_at_k,
    latency_percentiles,
    mrr,
    paper_hit_at_k,
)

PAPER_CORPUS = {
    "arxiv:1#0": {"chunk_id": "arxiv:1#0", "paper_id": "arxiv:1"},
    "arxiv:1#1": {"chunk_id": "arxiv:1#1", "paper_id": "arxiv:1"},
    "arxiv:2#0": {"chunk_id": "arxiv:2#0", "paper_id": "arxiv:2"},
}


def test_hit_at_k_hit():
    assert hit_at_k(["a", "b", "c"], ["c", "z"], k=3) == 1.0


def test_hit_at_k_miss():
    assert hit_at_k(["a", "b", "c"], ["z"], k=3) == 0.0


def test_hit_at_k_respects_k():
    # relevant id is retrieved but outside the top-k window
    assert hit_at_k(["a", "b", "c", "d"], ["d"], k=2) == 0.0


def test_mrr_first_position():
    assert mrr(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_third_position():
    assert mrr(["a", "b", "c"], ["c"]) == 1.0 / 3


def test_mrr_not_found():
    assert mrr(["a", "b", "c"], ["z"]) == 0.0


def test_paper_hit_at_k_hit():
    # Retrieved chunk_id is a DIFFERENT id than any relevant_chunk_id, but
    # shares the same paper_id -- this is the exact case the metric exists
    # for (re-chunked corpora produce entirely new chunk_ids).
    assert paper_hit_at_k(["arxiv:1#1"], {"arxiv:1"}, PAPER_CORPUS, k=3) == 1.0


def test_paper_hit_at_k_miss():
    assert paper_hit_at_k(["arxiv:2#0"], {"arxiv:1"}, PAPER_CORPUS, k=3) == 0.0


def test_paper_hit_at_k_respects_k():
    assert paper_hit_at_k(["arxiv:2#0", "arxiv:1#0"], {"arxiv:1"}, PAPER_CORPUS, k=1) == 0.0


def test_paper_hit_at_k_ignores_ids_not_in_corpus_by_id():
    # A chunk_id not present in corpus_by_id (shouldn't normally happen,
    # but must not crash) is simply skipped, not treated as a match.
    assert paper_hit_at_k(["not-a-real-chunk-id"], {"arxiv:1"}, PAPER_CORPUS, k=3) == 0.0


def test_filter_accuracy_none_when_not_required():
    assert filter_accuracy({"year": 2024}, None) is None


def test_filter_accuracy_match():
    assert filter_accuracy({"year": 2024}, {"year": 2024}) == 1.0


def test_filter_accuracy_mismatch():
    assert filter_accuracy({"year": 2023}, {"year": 2024}) == 0.0


def test_bootstrap_ci_deterministic_with_seed():
    values = [1.0, 0.0, 1.0, 1.0, 0.0]
    ci1 = bootstrap_ci(values, seed=42)
    ci2 = bootstrap_ci(values, seed=42)
    assert ci1.mean == ci2.mean
    assert ci1.lower == ci2.lower
    assert ci1.upper == ci2.upper


def test_bootstrap_ci_bounds_contain_mean():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    ci = bootstrap_ci(values)
    assert ci.lower <= ci.mean <= ci.upper


def test_bootstrap_ci_empty_list():
    ci = bootstrap_ci([])
    assert ci.mean == 0.0
    assert ci.lower == 0.0
    assert ci.upper == 0.0


def test_latency_percentiles():
    p50, p95 = latency_percentiles([100.0, 200.0, 300.0, 400.0, 500.0])
    assert p50 == 300.0
    assert p95 > p50


def test_latency_percentiles_empty():
    assert latency_percentiles([]) == (0.0, 0.0)
