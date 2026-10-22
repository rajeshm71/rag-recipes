import pytest

from evals.metrics import (
    bootstrap_ci,
    filter_accuracy,
    hit_at_k,
    latency_percentiles,
    mrr,
    paired_difference_ci,
    paper_hit_at_k,
    recall_at_k,
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


def test_recall_at_k_partial_and_full():
    assert recall_at_k(["a", "b", "c"], ["a", "z"], k=3) == 0.5
    assert recall_at_k(["a", "b", "z"], ["a", "z"], k=3) == 1.0
    assert recall_at_k(["a", "b", "z"], ["a", "z"], k=2) == 0.5
    assert recall_at_k(["a"], [], k=3) == 0.0


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


def test_filter_accuracy_none_when_pattern_made_no_claim():
    # Every pattern except 08 leaves extracted_filter at None: not
    # applicable, not a miss.
    assert filter_accuracy(None, {"year": 2024}) is None


def test_filter_accuracy_empty_claim_is_a_miss():
    # The pattern tried to extract a filter and found nothing: a real 0.
    assert filter_accuracy({}, {"year": 2024}) == 0.0


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


def test_paired_difference_ci_uses_per_question_differences():
    # Two rows whose per-row intervals overlap heavily, but b beats a by
    # exactly 0.1 on every question: the paired interval excludes 0.
    a = [0.1, 0.5, 0.9, 0.2, 0.6, 1.0, 0.3, 0.7]
    b = [x + 0.1 for x in a]
    ci = paired_difference_ci(b, a)
    assert ci.mean == pytest.approx(0.1)
    assert ci.lower == pytest.approx(0.1) and ci.upper == pytest.approx(0.1)

    same = paired_difference_ci(a, a)
    assert (same.mean, same.lower, same.upper) == (0.0, 0.0, 0.0)


def test_paired_difference_ci_requires_the_same_questions():
    with pytest.raises(ValueError):
        paired_difference_ci([1.0, 0.0], [1.0])


def test_recall_counts_an_evidence_piece_found_by_any_one_of_its_chunks():
    # Two pieces of evidence; the first appears in two near-identical chunks.
    evidence = [["a#1", "a#2"], ["b#4"]]
    relevant = ["a#1", "a#2", "b#4"]
    # Finding one copy of the first piece is enough for that piece.
    assert recall_at_k(["a#2", "x", "y"], relevant, 10, evidence) == 0.5
    assert recall_at_k(["a#2", "b#4"], relevant, 10, evidence) == 1.0
    # Without evidence groups every chunk is its own piece, as before.
    assert recall_at_k(["a#2", "b#4"], relevant, 10) == pytest.approx(2 / 3)
    # Pieces outside the top k do not count.
    assert recall_at_k(["x", "a#1", "b#4"], relevant, 2, evidence) == 0.5
