"""Per-token USD pricing for the models this project uses.

Prices verified against platform.openai.com/docs/models and
platform.openai.com/docs/pricing on 2026-08-11. Re-verify before relying on
these for a real spend decision if this file is more than a few weeks old,
per SKILL.md's Currency and Research rule.

All prices are USD per 1,000,000 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    cached_input_per_1m: float | None
    output_per_1m: float


# Chat/completion models (generation + judging).
CHAT_PRICING: dict[str, ModelPricing] = {
    "gpt-4.1-mini-2025-04-14": ModelPricing(
        input_per_1m=0.40, cached_input_per_1m=0.10, output_per_1m=1.60
    ),
    "gpt-5.4-mini-2026-03-17": ModelPricing(
        input_per_1m=0.75, cached_input_per_1m=0.075, output_per_1m=4.50
    ),
}

# Embedding models. No output tokens; cached_input_per_1m is None because
# OpenAI does not offer prompt-caching discounts on embedding calls.
EMBEDDING_PRICING: dict[str, ModelPricing] = {
    "text-embedding-3-small": ModelPricing(
        input_per_1m=0.02, cached_input_per_1m=None, output_per_1m=0.0
    ),
}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    cached_input_tokens: int = 0,
) -> float:
    """Compute the USD cost of one LLM or embedding call.

    `cached_input_tokens` must be <= input_tokens; the non-cached portion of
    input_tokens is billed at the regular input rate.
    """
    pricing = CHAT_PRICING.get(model) or EMBEDDING_PRICING.get(model)
    if pricing is None:
        raise ValueError(
            f"No pricing entry for model {model!r}. Add it to recipes/pricing.py "
            "after verifying the current rate at platform.openai.com/docs/pricing."
        )

    if cached_input_tokens > input_tokens:
        raise ValueError("cached_input_tokens cannot exceed input_tokens")

    uncached_input_tokens = input_tokens - cached_input_tokens
    cost = (uncached_input_tokens / 1_000_000) * pricing.input_per_1m
    cost += (output_tokens / 1_000_000) * pricing.output_per_1m

    if cached_input_tokens:
        if pricing.cached_input_per_1m is None:
            raise ValueError(
                f"Model {model!r} has no cached-input rate; caller passed "
                f"cached_input_tokens={cached_input_tokens} in error."
            )
        cost += (cached_input_tokens / 1_000_000) * pricing.cached_input_per_1m

    return cost
