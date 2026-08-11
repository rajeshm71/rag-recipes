"""Eval harness orchestrator. Scores one RAG pattern's recipe function
against the eval set and returns every metric from SPEC.md §5, each with a
95% bootstrap confidence interval.

This is the file whose behavior is the literal P1 acceptance criterion
(SPEC.md §18): "evals/run.py can score a dummy retrieval function end to
end and print all metrics with CIs."
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from evals.judges import (
    judge_answer_relevance,
    judge_citation_accuracy,
    judge_faithfulness,
)
from evals.metrics import (
    ConfidenceInterval,
    bootstrap_ci,
    filter_accuracy,
    hit_at_k,
    latency_percentiles,
    mrr,
)
from recipes import AnswerWithCitations
from recipes.llm import LLM
from recipes.pricing import cost_usd

GENERATION_MODEL = "gpt-4.1-mini-2025-04-14"


@dataclass
class PatternResult:
    pattern_name: str
    n_questions: int
    hit_at_3: ConfidenceInterval
    hit_at_10: ConfidenceInterval
    mrr: ConfidenceInterval
    faithfulness: ConfidenceInterval | None
    answer_relevance: ConfidenceInterval | None
    citation_accuracy: ConfidenceInterval | None
    filter_accuracy: ConfidenceInterval | None
    p50_latency_ms: float
    p95_latency_ms: float
    usd_per_query: float
    eval_usd: float
    # FIX (review #3): count of questions skipped entirely because
    # recipe_fn itself raised (no data at all for that question). Defaults
    # to 0 so existing call sites/tests that don't reference this field are
    # unaffected.
    n_errors: int = 0
    # FIX (P2, evals/run.py bug found while building 02_bm25.ipynb): a
    # judge-call failure is NOT the same situation as a recipe_fn failure --
    # retrieval already succeeded and its scores are real, valid data. This
    # counts questions where retrieval succeeded but one or more judge
    # calls failed, so faithfulness/answer_relevance/citation_accuracy may
    # be computed over fewer questions than hit@k/mrr. Previously this case
    # was folded into n_errors with a misleading "failed and was skipped"
    # message, even though the retrieval score for that question was kept.
    n_judge_errors: int = 0
    # FIX (code review, same bug class as n_judge_errors): cost accounting
    # (cost_usd) and filter-accuracy scoring happened in the SAME try block
    # as the hit@k/mrr/latency appends, so a failure in either of those
    # later steps would still print "no retrieval data" even though the
    # retrieval scores were already appended and kept. Not reachable today
    # (both P2 patterns hardcode a valid, priced generation model), but the
    # same latent-bug shape the judge split already fixed elsewhere.
    n_accounting_errors: int = 0


def load_qa_set(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_corpus_by_id(path: str | Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunk = json.loads(line)
                by_id[chunk["chunk_id"]] = chunk
    return by_id


def _build_context(chunk_ids: list[str], corpus_by_id: dict[str, dict]) -> str:
    parts = []
    for cid in chunk_ids:
        chunk = corpus_by_id.get(cid)
        if chunk is not None:
            parts.append(f"[{cid}] {chunk['text']}")
    return "\n\n".join(parts)


def run_pattern(
    recipe_fn: Callable[..., AnswerWithCitations],
    qa_set: list[dict],
    corpus_by_id: dict[str, dict],
    llm: LLM,
    pattern_name: str = "pattern",
    generation_model: str = GENERATION_MODEL,
    judges_enabled: bool = True,
    k: int = 10,
    verbose: bool = True,
) -> PatternResult:
    """Run `recipe_fn` over every question in `qa_set`, score it against
    every metric in SPEC.md §5, and return the aggregated result.

    `recipe_fn` must match the AnswerWithCitations contract from
    recipes/__init__.py: `recipe_fn(question: str, k: int) -> AnswerWithCitations`.
    """
    hit3_scores: list[float] = []
    hit10_scores: list[float] = []
    mrr_scores: list[float] = []
    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []
    citation_scores: list[float] = []
    filter_scores: list[float] = []
    latencies: list[float] = []
    generation_usd_total = 0.0
    judge_usd_total = 0.0
    n_errors = 0
    n_judge_errors = 0
    n_accounting_errors = 0

    for record in qa_set:
        question = record["question"]
        qid = record.get("qid", question[:40])
        relevant_ids = record["relevant_chunk_ids"]
        requires_filter = record.get("requires_filter")

        # FIX (review #3): isolate each question so one bad recipe_fn call
        # doesn't lose every score already collected for prior questions in
        # this run. Previously an uncaught exception here would propagate
        # straight out of run_pattern and discard the whole run -- costly
        # for a real, paid eval run against real APIs. This block covers
        # ONLY recipe_fn itself plus the minimal hit@k/mrr/latency scoring
        # that depends solely on its return value -- cost accounting, filter
        # scoring, and judge calls are each their own try/except below,
        # since a failure in any of those is a fundamentally different
        # situation from recipe_fn itself failing (retrieval data already
        # exists and is real in all of those cases).
        try:
            result = recipe_fn(question, k=k)

            hit3_scores.append(hit_at_k(result.retrieved_chunk_ids, relevant_ids, 3))
            hit10_scores.append(hit_at_k(result.retrieved_chunk_ids, relevant_ids, 10))
            mrr_scores.append(mrr(result.retrieved_chunk_ids, relevant_ids))
            latencies.append(result.latency_ms)
        except Exception as exc:  # noqa: BLE001 -- deliberately broad, see comment above
            n_errors += 1
            print(f"  WARNING: question {qid!r} failed and was skipped (no retrieval data): {exc}", file=sys.stderr)
            continue

        # FIX (code review): cost accounting and filter scoring are their
        # own try/except, separate from the hit@k/mrr appends above. A
        # failure here (e.g. an unpriced generation_model) must not be
        # reported as "no retrieval data" -- the retrieval scores above
        # were already appended and are kept regardless.
        try:
            generation_usd_total += cost_usd(
                generation_model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_input_tokens=result.cached_input_tokens,
            )

            fa = filter_accuracy(result.extracted_filter, requires_filter)
            if fa is not None:
                filter_scores.append(fa)
        except Exception as exc:  # noqa: BLE001
            n_accounting_errors += 1
            print(f"  WARNING: question {qid!r} retrieval succeeded but cost/filter "
                  f"accounting failed: {exc}", file=sys.stderr)

        if not judges_enabled:
            continue

        # FIX (P2, evals/run.py bug found while building 02_bm25.ipynb):
        # separate try/except so a judge failure only drops THIS question's
        # judge scores, not the retrieval scores already appended above.
        # The old single try/except wrapped both, so a judge-only failure
        # incremented n_errors and printed "failed and was skipped" even
        # though hit@k/mrr for that question were real, valid, and kept --
        # a misleading message for what actually happened.
        try:
            context = _build_context(result.retrieved_chunk_ids, corpus_by_id)

            fr = judge_faithfulness(llm, question, context, result.answer)
            faithfulness_scores.append(fr.score)
            judge_usd_total += fr.usd_cost

            rr = judge_answer_relevance(llm, question, result.answer)
            relevance_scores.append(rr.score)
            judge_usd_total += rr.usd_cost

            cr = judge_citation_accuracy(llm, context, result.answer)
            citation_scores.append(cr.score)
            judge_usd_total += cr.usd_cost
        except Exception as exc:  # noqa: BLE001
            n_judge_errors += 1
            print(f"  WARNING: question {qid!r} retrieval succeeded but judging failed: {exc}", file=sys.stderr)

    p50, p95 = latency_percentiles(latencies)
    n = len(qa_set)
    total_usd = generation_usd_total + judge_usd_total

    pattern_result = PatternResult(
        pattern_name=pattern_name,
        n_questions=n,
        hit_at_3=bootstrap_ci(hit3_scores),
        hit_at_10=bootstrap_ci(hit10_scores),
        mrr=bootstrap_ci(mrr_scores),
        faithfulness=bootstrap_ci(faithfulness_scores) if judges_enabled else None,
        answer_relevance=bootstrap_ci(relevance_scores) if judges_enabled else None,
        citation_accuracy=bootstrap_ci(citation_scores) if judges_enabled else None,
        filter_accuracy=bootstrap_ci(filter_scores) if filter_scores else None,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        usd_per_query=total_usd / n if n else 0.0,
        eval_usd=total_usd,
        n_errors=n_errors,
        n_judge_errors=n_judge_errors,
        n_accounting_errors=n_accounting_errors,
    )

    if verbose:
        print_result(pattern_result)

    return pattern_result


def print_result(result: PatternResult) -> None:
    def fmt_ci(name: str, ci: ConfidenceInterval | None) -> str:
        if ci is None:
            return f"  {name}: (not computed)"
        return f"  {name}: {ci.mean:.3f}  [95% CI {ci.lower:.3f}, {ci.upper:.3f}]"

    print(f"=== {result.pattern_name} (n={result.n_questions}) ===")
    print(fmt_ci("hit@3", result.hit_at_3))
    print(fmt_ci("hit@10", result.hit_at_10))
    print(fmt_ci("mrr", result.mrr))
    print(fmt_ci("faithfulness", result.faithfulness))
    print(fmt_ci("answer_relevance", result.answer_relevance))
    print(fmt_ci("citation_accuracy", result.citation_accuracy))
    print(fmt_ci("filter_accuracy", result.filter_accuracy))
    print(f"  p50_latency_ms: {result.p50_latency_ms:.1f}")
    print(f"  p95_latency_ms: {result.p95_latency_ms:.1f}")
    print(f"  usd_per_query: ${result.usd_per_query:.5f}")
    print(f"  eval_usd: ${result.eval_usd:.4f}")
    if result.n_errors:
        print(f"  WARNING: {result.n_errors} question(s) skipped entirely due to a "
              f"recipe_fn failure (see stderr) -- hit@k/mrr above are computed over "
              f"{result.n_questions - result.n_errors} of {result.n_questions} questions")
    if result.n_judge_errors:
        print(f"  WARNING: {result.n_judge_errors} question(s) had retrieval succeed but "
              f"judging fail (see stderr) -- faithfulness/answer_relevance/citation_accuracy "
              f"above are computed over fewer questions than hit@k/mrr")
    if result.n_accounting_errors:
        print(f"  WARNING: {result.n_accounting_errors} question(s) had retrieval succeed but "
              f"cost/filter accounting fail (see stderr) -- usd_per_query/eval_usd/filter_accuracy "
              f"may be understated")
