import pytest

from recipes.pricing import cost_usd


def test_gpt_4_1_mini_cost_unchanged():
    # Regression check: existing OpenAI pricing behavior must not shift
    # when the cache_creation_input_tokens tier is added for Anthropic.
    cost = cost_usd("gpt-4.1-mini-2025-04-14", input_tokens=1000, output_tokens=500)
    expected = (1000 / 1_000_000) * 0.40 + (500 / 1_000_000) * 1.60
    assert cost == pytest.approx(expected)


def test_gpt_4_1_mini_cost_with_cached_input_unchanged():
    cost = cost_usd(
        "gpt-4.1-mini-2025-04-14", input_tokens=1000, output_tokens=0, cached_input_tokens=400
    )
    expected = (600 / 1_000_000) * 0.40 + (400 / 1_000_000) * 0.10
    assert cost == pytest.approx(expected)


def test_claude_sonnet_5_cost_across_all_three_tiers():
    # 200 tokens plain input, 300 cached (read), 500 cache-creation (write),
    # 100 output -- exercises all three input tiers in one call.
    cost = cost_usd(
        "claude-sonnet-5",
        input_tokens=1000,
        output_tokens=100,
        cached_input_tokens=300,
        cache_creation_input_tokens=500,
    )
    expected = (
        (200 / 1_000_000) * 2.00
        + (100 / 1_000_000) * 10.00
        + (300 / 1_000_000) * 0.20
        + (500 / 1_000_000) * 2.50
    )
    assert cost == pytest.approx(expected)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        cost_usd("totally-unpriced-model-xyz", input_tokens=100)


def test_cache_creation_tokens_on_model_without_that_tier_raises():
    with pytest.raises(ValueError):
        cost_usd(
            "gpt-4.1-mini-2025-04-14",
            input_tokens=1000,
            cache_creation_input_tokens=100,
        )


def test_cached_plus_creation_exceeding_input_tokens_raises():
    with pytest.raises(ValueError):
        cost_usd(
            "claude-sonnet-5",
            input_tokens=100,
            cached_input_tokens=60,
            cache_creation_input_tokens=60,
        )
