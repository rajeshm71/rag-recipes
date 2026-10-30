"""Structural checks on the eval set, run as a unit test against the real data:

    uv run python evals/validate_qa.py      # print problems and category counts; exit 1 on any problem

Checks:
- every question has a unique qid, a known category and non-empty evidence
  groups; every chunk id exists; relevant_chunk_ids is the union of the groups;
- multi_hop questions have 2+ evidence groups drawn from 2+ papers;
- filter questions name a category that every target paper is listed in, and
  at least one distractor paper, none of which is listed in that category;
- answer support: the labelled chunks contain at least MIN_ANSWER_SUPPORT of
  the reference answer's distinctive terms (see answer_terms), a share low
  enough that paraphrased answers pass.

Category counts are printed, not enforced.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.run import load_corpus_by_id, load_qa_set

CATEGORIES = ("keyword", "paraphrase", "multi_hop", "filter")
MIN_ANSWER_SUPPORT = 0.4
# a term is distinctive if it holds a digit or appears in at most this share of chunks
DISTINCTIVE_MAX_DF = 0.03

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.%-]*[A-Za-z0-9%]|[0-9]")


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(text)}


def document_frequency(corpus: dict[str, dict]) -> Counter:
    df: Counter = Counter()
    for chunk in corpus.values():
        df.update(_tokens(chunk["text"]))
    return df


def answer_terms(answer: str, df: Counter, n_chunks: int) -> set[str]:
    """The reference answer's distinctive terms: numbers, and words rare in the corpus."""
    limit = DISTINCTIVE_MAX_DF * n_chunks
    return {t for t in _tokens(answer) if any(c.isdigit() for c in t) or 0 < df[t] <= limit}


def answer_support(record: dict, corpus: dict[str, dict], df: Counter) -> float:
    """Share of the answer's distinctive terms found in the question's labelled chunks."""
    terms = answer_terms(record.get("reference_answer") or "", df, len(corpus))
    if not terms:
        return 1.0
    text = " ".join(corpus[c]["text"] for c in record["relevant_chunk_ids"] if c in corpus).lower()
    return sum(1 for t in terms if t in text) / len(terms)


def _paper(chunk_id: str) -> str:
    return chunk_id.split("#")[0]


def validate(qa_set: list[dict], corpus: dict[str, dict], papers: dict[str, dict]) -> list[str]:
    """Every problem found, one message per problem; empty when the set is valid."""
    problems: list[str] = []
    df = document_frequency(corpus)
    seen: set[str] = set()
    for record in qa_set:
        qid = record.get("qid", "?")
        if qid in seen:
            problems.append(f"{qid}: duplicate qid")
        seen.add(qid)
        category = record.get("category")
        if category not in CATEGORIES:
            problems.append(f"{qid}: unknown category {category!r}")
        evidence = record.get("evidence") or []
        if not evidence or not all(evidence):
            problems.append(f"{qid}: needs non-empty evidence groups")
        ids = {c for group in evidence for c in group}
        for chunk_id in sorted(ids - corpus.keys()):
            problems.append(f"{qid}: unknown chunk id {chunk_id}")
        if sorted(ids) != sorted(record.get("relevant_chunk_ids", [])):
            problems.append(f"{qid}: relevant_chunk_ids is not the union of the evidence groups")

        if category == "multi_hop":
            group_papers = {_paper(c) for group in evidence for c in group}
            if len(evidence) < 2 or len(group_papers) < 2:
                problems.append(f"{qid}: multi_hop needs 2+ evidence groups from 2+ papers")

        if category == "filter":
            wanted = (record.get("requires_filter") or {}).get("category")
            distractors = record.get("distractor_paper_ids") or []
            if not wanted:
                problems.append(f"{qid}: filter question without a requires_filter category")
            if not distractors:
                problems.append(f"{qid}: filter question without distractor papers")
            for paper_id in sorted({_paper(c) for c in ids}):
                if wanted not in papers.get(paper_id, {}).get("categories", []):
                    problems.append(f"{qid}: target paper {paper_id} is not listed in {wanted}")
            for paper_id in distractors:
                if paper_id not in papers:
                    problems.append(f"{qid}: unknown distractor paper {paper_id}")
                elif wanted in papers[paper_id].get("categories", []):
                    problems.append(f"{qid}: distractor paper {paper_id} is listed in {wanted}")

        support = answer_support(record, corpus, df)
        if support < MIN_ANSWER_SUPPORT:
            problems.append(
                f"{qid}: labelled chunks hold {support:.2f} of the answer's distinctive terms "
                f"(minimum {MIN_ANSWER_SUPPORT})"
            )
    return problems


def load_papers(path: str | Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {p["paper_id"]: p for p in map(json.loads, filter(str.strip, f))}


def main() -> None:
    qa_set = load_qa_set(ROOT / "evals" / "qa_set.jsonl")
    corpus = load_corpus_by_id(ROOT / "corpus" / "corpus.jsonl")
    papers = load_papers(ROOT / "corpus" / "papers.jsonl")
    problems = validate(qa_set, corpus, papers)
    counts = Counter(r.get("category") for r in qa_set)
    print(f"{len(qa_set)} questions: " + ", ".join(f"{c} {counts.get(c, 0)}" for c in CATEGORIES))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        sys.exit(1)
    print("OK: no problems found.")


if __name__ == "__main__":
    main()
