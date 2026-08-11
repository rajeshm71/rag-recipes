"""LLM adapter. All LLM calls in this project go through this module.

Contract fixed by SPEC.md §10: `LLM.complete()` returns an `LLMResponse` with
text, token counts, cached-token count (for prompt-caching cost tracking),
and latency. Two concrete implementations ship here: `OpenAILLM` (real API)
and `MockLLM` (deterministic, used in CI per SPEC.md R8 -- CI never touches
a real API key).
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
    latency_ms: float = 0.0


class LLM(Protocol):
    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
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
    ) -> LLMResponse:
        start = time.perf_counter()
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Some reasoning-capable models (e.g. gpt-5.4-mini family) reject
            # `temperature` entirely rather than clamping/ignoring it. This
            # behavior was not confirmed against current docs at write time
            # (see SPEC.md P1 plan "Known technical risks"), so we handle it
            # defensively: on the specific "unsupported parameter" error,
            # retry once without temperature instead of failing the call.
            message = str(exc).lower()
            if "temperature" in message and (
                "unsupported" in message or "not supported" in message
            ):
                kwargs.pop("temperature")
                response = self._client.chat.completions.create(**kwargs)
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

    def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        self.calls.append(_RecordedCall(prompt, model, temperature, max_tokens))

        text = self.default_response
        for substring, canned_text in self.canned.items():
            if substring in prompt:
                text = canned_text
                break

        # Deterministic pseudo token counts so cost/metric code has something
        # sane to work with in tests without needing a real tokenizer.
        input_tokens = max(1, len(prompt) // 4)
        output_tokens = max(1, len(text) // 4)

        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=0,
            latency_ms=0.1,
        )


def get_llm() -> LLM:
    """Factory respecting the RAG_RECIPES_LLM env var (SPEC.md §12/§13)."""
    backend = os.environ.get("RAG_RECIPES_LLM", "openai").lower()
    if backend == "mock":
        return MockLLM()
    return OpenAILLM()
