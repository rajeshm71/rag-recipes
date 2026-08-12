"""LLM adapter. All LLM calls in this project go through this module.

Contract fixed by SPEC.md §10: `LLM.complete()` returns an `LLMResponse` with
text, token counts, cached-token count (for prompt-caching cost tracking),
and latency. Three concrete implementations ship here: `OpenAILLM` (real
API, used for generation/judging/long-context per SPEC.md §3), `AnthropicLLM`
(real API, used ONLY by pattern 07's contextualization preprocessing step),
and `MockLLM` (deterministic, used in CI per SPEC.md R8 -- CI never touches
a real API key, for any provider).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    # Anthropic-only: tokens written to a FRESH cache entry, billed at a
    # premium (distinct tier from cached_input_tokens' discounted-read
    # price). Always 0 for OpenAI/Mock, which have no separate write tier.
    cache_creation_input_tokens: int = 0
    latency_ms: float = 0.0


class LLM(Protocol):
    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        # Hint for providers with explicit prompt caching (Anthropic): the
        # portion of the request that's shared/repeated across calls and
        # should be cached. No-op for OpenAILLM (its caching is automatic,
        # no explicit control at our usage tier) and MockLLM (recorded for
        # test assertions only, doesn't change the mock response).
        cacheable_prefix: str | None = None,
    ) -> LLMResponse: ...


class OpenAILLM:
    """Production adapter. Requires OPENAI_API_KEY to be set."""

    def __init__(self, api_key: str | None = None) -> None:
        # Imported lazily so MockLLM-only test runs never need the `openai`
        # package installed to import this module.
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,  # no-op: see LLM Protocol docstring
    ) -> LLMResponse:
        start = time.perf_counter()
        base_kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
        )
        try:
            response = self._client.chat.completions.create(
                **base_kwargs, temperature=temperature
            )
        except Exception as exc:
            # Some reasoning-capable models (e.g. gpt-5.4-mini family) reject
            # `temperature` entirely rather than clamping/ignoring it. This
            # behavior was not confirmed against current docs at write time
            # (see SPEC.md P1 plan "Known technical risks"), so we handle it
            # defensively: on the specific "unsupported parameter" error,
            # retry once without temperature instead of failing the call.
            # FIX (review #6): build a fresh call from base_kwargs (which
            # never had temperature) instead of mutating the first call's
            # kwargs dict via .pop() -- clearer that the two calls are
            # independent and there's no risk of a stale mutated dict being
            # reused if this function is ever refactored further.
            message = str(exc).lower()
            if "temperature" in message and (
                "unsupported" in message or "not supported" in message
            ):
                response = self._client.chat.completions.create(**base_kwargs)
            else:
                raise
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        usage = response.usage
        cached = 0
        if usage is not None and getattr(usage, "prompt_tokens_details", None):
            cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        return LLMResponse(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            cached_input_tokens=cached,
            latency_ms=latency_ms,
        )


@dataclass
class _RecordedCall:
    prompt: str
    model: str
    temperature: float
    max_tokens: int
    cacheable_prefix: str | None = None


class MockLLM:
    """Deterministic adapter for tests and CI. Never makes a network call.

    Returns a fixed string by default, or a caller-supplied mapping from
    prompt substrings to canned responses (checked in insertion order, first
    match wins) for tests that need different answers to different prompts.
    Every call is recorded in `.calls` for test assertions.
    """

    def __init__(
        self,
        default_response: str = "This is a mock response.",
        canned: dict[str, str] | None = None,
    ) -> None:
        self.default_response = default_response
        self.canned = canned or {}
        self.calls: list[_RecordedCall] = []
        # Tracks which cacheable_prefix strings this instance has already
        # "seen", to simulate real cache write-then-read behavior (see
        # complete()'s cache-tier logic below).
        self._seen_prefixes: set[str] = set()

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,
    ) -> LLMResponse:
        self.calls.append(_RecordedCall(prompt, model, temperature, max_tokens, cacheable_prefix))

        text = self.default_response
        for substring, canned_text in self.canned.items():
            if substring in prompt:
                text = canned_text
                break

        # Deterministic pseudo token counts so cost/metric code has something
        # sane to work with in tests without needing a real tokenizer.
        output_tokens = max(1, len(text) // 4)

        # FIX: without this, MockLLM never populated cached_input_tokens/
        # cache_creation_input_tokens, so pattern 07's with-vs-without-
        # caching cost comparison was always identical under mock (found
        # while executing 07_contextual.ipynb -- $0.00 savings every time,
        # even though the formula itself was correct). Simulate real cache
        # semantics deterministically instead: the FIRST time a given
        # cacheable_prefix is seen on this instance, its tokens are billed
        # as a cache write; every subsequent call with the SAME prefix
        # bills them as a cache read. This only activates when a caller
        # passes cacheable_prefix (only recipes/contextual.py does), so
        # every other pattern's MockLLM behavior is unaffected.
        if cacheable_prefix:
            prefix_tokens = max(1, len(cacheable_prefix) // 4)
            suffix_tokens = max(1, len(prompt) // 4)
            input_tokens = prefix_tokens + suffix_tokens
            if cacheable_prefix in self._seen_prefixes:
                cached_input_tokens = prefix_tokens
                cache_creation_input_tokens = 0
            else:
                cached_input_tokens = 0
                cache_creation_input_tokens = prefix_tokens
                self._seen_prefixes.add(cacheable_prefix)
        else:
            input_tokens = max(1, len(prompt) // 4)
            cached_input_tokens = 0
            cache_creation_input_tokens = 0

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            latency_ms=0.1,
        )


class AnthropicLLM:
    """Production adapter for pattern 07's contextualization step ONLY.
    Requires ANTHROPIC_API_KEY. Per SPEC.md §3, every other LLM call in this
    project (generation, judging, long-context baseline) uses OpenAI --
    Anthropic is not a general-purpose swap-in here.
    """

    def __init__(self, api_key: str | None = None) -> None:
        # Imported lazily, same convention as OpenAILLM, so MockLLM-only
        # test runs never need the `anthropic` package importable.
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        cacheable_prefix: str | None = None,
    ) -> LLMResponse:
        start = time.perf_counter()
        kwargs = dict(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        if cacheable_prefix:
            kwargs["system"] = [
                {"type": "text", "text": cacheable_prefix, "cache_control": {"type": "ephemeral"}}
            ]
        response = self._client.messages.create(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # Anthropic's usage.input_tokens excludes BOTH cache tiers (unlike
        # OpenAI's prompt_tokens, which already includes the cached
        # subset) -- recombine to the total here so LLMResponse.input_tokens
        # keeps one consistent meaning across providers: the full input
        # token count, with cached_input_tokens/cache_creation_input_tokens
        # as named subsets/tiers within it. Verified against
        # platform.claude.com/docs/en/build-with-claude/prompt-caching on
        # 2026-08-12: total_input = input_tokens + cache_read + cache_creation.
        total_input = usage.input_tokens + cache_read + cache_creation
        text = "".join(block.text for block in response.content if block.type == "text")

        return LLMResponse(
            text=text,
            input_tokens=total_input,
            output_tokens=usage.output_tokens,
            cached_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            latency_ms=latency_ms,
        )


def get_llm() -> LLM:
    """Factory respecting the RAG_RECIPES_LLM env var (SPEC.md §12/§13)."""
    backend = os.environ.get("RAG_RECIPES_LLM", "openai").lower()
    if backend == "mock":
        return MockLLM()
    return OpenAILLM()


def get_anthropic_llm() -> LLM:
    """Factory for pattern 07's Anthropic step, respecting RAG_RECIPES_LLM
    the same way get_llm() does -- R8 (CI is mock-only) applies to every
    provider, not just OpenAI.
    """
    backend = os.environ.get("RAG_RECIPES_LLM", "openai").lower()
    if backend == "mock":
        return MockLLM()
    return AnthropicLLM()
