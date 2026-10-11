"""Shared contract every pattern recipe (recipes/<pattern>.py) implements.

Each pattern module exposes a factory:

    def make_retrieve_and_answer(corpus_by_id, llm, ...) -> Callable

which builds its index once and returns the bound closure

    def retrieve_and_answer(question: str, k: int = 5) -> AnswerWithCitations

that evals/run.py's run_pattern() and every notebook call per question.
This lives in recipes/__init__.py rather than a pattern-specific file
because it's the interface the eval harness and every pattern module both
depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AnswerWithCitations:
    answer: str
    # Rank-ordered, best first, no duplicates: hit@k slices the top k and
    # MRR uses the position of the first relevant id, so a pattern that
    # returns an unordered set here is scored on an arbitrary order.
    retrieved_chunk_ids: list[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    # Anthropic-only: tokens written to a fresh cache entry, billed at a
    # premium (distinct tier from cached_input_tokens' discounted read
    # rate). Always 0 for OpenAI/Mock, which have no separate write tier.
    # Every two-call pattern (hyde/multi_query/self_query/multi_hop/agentic)
    # must sum this across all its LLM calls into AnswerWithCitations, and
    # evals/run.py's cost_usd() call must pass it -- without either, a
    # pattern that eventually uses Anthropic for its per-question call
    # would silently underprice its cost by whatever the cache-write cost
    # actually was.
    cache_creation_input_tokens: int = 0
    # Only populated by pattern 08 (self-query); the filter the pattern
    # extracted from the question, for comparison against qa_set.jsonl's
    # `requires_filter` ground truth.
    extracted_filter: dict | None = None
    # Only populated by pattern 10 (agentic); logged to
    # outputs/agentic_traces.jsonl so readers can see the agent's reasoning.
    tool_call_trace: list[dict] = field(default_factory=list)


__all__ = ["AnswerWithCitations"]
