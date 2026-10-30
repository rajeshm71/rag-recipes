"""evals/validate_qa.py: each check fails on a deliberately broken fixture,
the fixture passes as written, and the committed eval set passes."""

import copy

import pytest

from evals.run import load_corpus_by_id, load_qa_set
from evals.validate_qa import ROOT, load_papers, main, validate

CORPUS = {
    "arxiv:1#0": {
        "chunk_id": "arxiv:1#0",
        "text": "The model reaches 91.5% accuracy after 40 epochs.",
    },
    "arxiv:1#1": {"chunk_id": "arxiv:1#1", "text": "Training used 8 GPUs."},
    "arxiv:2#0": {"chunk_id": "arxiv:2#0", "text": "The benchmark has 120 tasks."},
    "arxiv:3#0": {"chunk_id": "arxiv:3#0", "text": "A survey of 57 systems."},
}
PAPERS = {
    "arxiv:1": {"paper_id": "arxiv:1", "categories": ["cs.CL", "cs.LG"]},
    "arxiv:2": {"paper_id": "arxiv:2", "categories": ["cs.AI"]},
    "arxiv:3": {"paper_id": "arxiv:3", "categories": ["cs.LG"]},
}
QA = [
    {
        "qid": "q1",
        "category": "keyword",
        "relevant_chunk_ids": ["arxiv:1#0"],
        "reference_answer": "91.5% after 40 epochs.",
        "evidence": [["arxiv:1#0"]],
    },
    {
        "qid": "q2",
        "category": "multi_hop",
        "relevant_chunk_ids": ["arxiv:1#1", "arxiv:2#0"],
        "reference_answer": "8 GPUs, and 120 tasks.",
        "evidence": [["arxiv:1#1"], ["arxiv:2#0"]],
    },
    {
        "qid": "q3",
        "category": "filter",
        "relevant_chunk_ids": ["arxiv:3#0"],
        "reference_answer": "It surveys 57 systems.",
        "requires_filter": {"category": "cs.LG"},
        "distractor_paper_ids": ["arxiv:2"],
        "evidence": [["arxiv:3#0"]],
    },
]


def _broken(target: str, /, **changes) -> list[dict]:
    qa = copy.deepcopy(QA)
    record = next(r for r in qa if r["qid"] == target)
    record.update(changes)
    return qa


def test_fixture_is_valid():
    assert validate(QA, CORPUS, PAPERS) == []


@pytest.mark.parametrize(
    ("qa", "expected"),
    [
        (_broken("q2", qid="q1"), "duplicate qid"),
        (_broken("q1", category="trivia"), "unknown category"),
        (_broken("q1", evidence=[]), "non-empty evidence"),
        (
            _broken("q1", evidence=[["arxiv:9#0"]], relevant_chunk_ids=["arxiv:9#0"]),
            "unknown chunk id",
        ),
        (_broken("q1", relevant_chunk_ids=["arxiv:1#0", "arxiv:1#1"]), "not the union"),
        (_broken("q2", evidence=[["arxiv:1#1", "arxiv:2#0"]]), "2+ evidence groups from 2+ papers"),
        (
            _broken(
                "q2",
                evidence=[["arxiv:1#0"], ["arxiv:1#1"]],
                relevant_chunk_ids=["arxiv:1#0", "arxiv:1#1"],
            ),
            "2+ evidence groups from 2+ papers",
        ),
        (_broken("q3", requires_filter=None), "without a requires_filter category"),
        (_broken("q3", distractor_paper_ids=[]), "without distractor papers"),
        (
            _broken("q3", requires_filter={"category": "cs.AI"}, distractor_paper_ids=["arxiv:1"]),
            "is not listed in",
        ),
        (
            _broken("q3", distractor_paper_ids=["arxiv:1"]),
            "distractor paper arxiv:1 is listed in cs.LG",
        ),
        (_broken("q3", distractor_paper_ids=["arxiv:7"]), "unknown distractor paper"),
        (
            _broken("q1", evidence=[["arxiv:1#1"]], relevant_chunk_ids=["arxiv:1#1"]),
            "distinctive terms",
        ),
    ],
)
def test_each_check_catches_its_problem(qa, expected):
    problems = validate(qa, CORPUS, PAPERS)
    assert any(expected in p for p in problems), problems


def test_paraphrased_answer_with_partial_support_passes():
    # two of three distinctive terms (91.5%, 40) are in the chunk: above the minimum
    qa = _broken("q1", reference_answer="About 91.5% once 40 epochs had passed, on 3 seeds.")
    assert validate(qa, CORPUS, PAPERS) == []


def test_committed_eval_set_is_valid():
    qa = load_qa_set(ROOT / "evals" / "qa_set.jsonl")
    corpus = load_corpus_by_id(ROOT / "corpus" / "corpus.jsonl")
    papers = load_papers(ROOT / "corpus" / "papers.jsonl")
    assert validate(qa, corpus, papers) == []


def test_main_prints_category_counts(capsys):
    main()
    out = capsys.readouterr().out
    assert "questions: keyword" in out and "multi_hop" in out and "OK" in out
