from recipes.llm import AnthropicLLM, MockLLM, get_anthropic_llm


def test_mock_llm_default_response():
    llm = MockLLM(default_response="hello")
    resp = llm.complete("any prompt", model="gpt-4.1-mini-2025-04-14")
    assert resp.text == "hello"
    assert resp.input_tokens > 0
    assert resp.output_tokens > 0
    assert resp.cached_input_tokens == 0


def test_mock_llm_canned_response_matches_substring():
    llm = MockLLM(canned={"weather": "It is sunny."})
    resp = llm.complete("What is the weather today?", model="gpt-4.1-mini-2025-04-14")
    assert resp.text == "It is sunny."


def test_mock_llm_records_calls():
    llm = MockLLM()
    llm.complete("first", model="m1", temperature=0.2, max_tokens=10)
    llm.complete("second", model="m2")
    assert len(llm.calls) == 2
    assert llm.calls[0].prompt == "first"
    assert llm.calls[0].model == "m1"
    assert llm.calls[0].temperature == 0.2
    assert llm.calls[1].prompt == "second"


def test_mock_llm_is_deterministic():
    llm = MockLLM(default_response="same every time")
    r1 = llm.complete("x", model="m")
    r2 = llm.complete("x", model="m")
    assert r1.text == r2.text


def test_mock_llm_records_cacheable_prefix():
    llm = MockLLM()
    llm.complete("prompt", model="m", cacheable_prefix="the shared document text")
    llm.complete("prompt2", model="m")
    assert llm.calls[0].cacheable_prefix == "the shared document text"
    assert llm.calls[1].cacheable_prefix is None


def test_mock_llm_simulates_cache_write_then_read():
    # FIX regression test: without this, MockLLM never populated
    # cached_input_tokens/cache_creation_input_tokens, so pattern 07's
    # with-vs-without-caching cost comparison was always identical under
    # mock. First call with a given cacheable_prefix must be a cache
    # WRITE (cache_creation_input_tokens > 0, cached_input_tokens == 0);
    # a later call with the SAME prefix must be a cache READ (the
    # opposite).
    llm = MockLLM()
    doc = "a fairly long shared document, repeated across many chunk calls"

    first = llm.complete("chunk one prompt", model="m", cacheable_prefix=doc)
    assert first.cache_creation_input_tokens > 0
    assert first.cached_input_tokens == 0

    second = llm.complete("chunk two prompt", model="m", cacheable_prefix=doc)
    assert second.cached_input_tokens > 0
    assert second.cache_creation_input_tokens == 0

    # A DIFFERENT prefix must be treated as a fresh cache write again, not
    # a read -- proves the simulation keys on the prefix's actual content.
    other_doc = "a completely different document"
    third = llm.complete("chunk three prompt", model="m", cacheable_prefix=other_doc)
    assert third.cache_creation_input_tokens > 0
    assert third.cached_input_tokens == 0


def test_mock_llm_input_tokens_include_prefix_when_given():
    llm = MockLLM()
    doc = "x" * 400  # a long-ish cacheable prefix
    result = llm.complete("short prompt", model="m", cacheable_prefix=doc)
    prefix_tokens = max(1, len(doc) // 4)
    suffix_tokens = max(1, len("short prompt") // 4)
    assert result.input_tokens == prefix_tokens + suffix_tokens


def test_get_anthropic_llm_respects_env_var(monkeypatch):
    monkeypatch.setenv("RAG_RECIPES_LLM", "mock")
    assert isinstance(get_anthropic_llm(), MockLLM)

    # Avoid actually constructing AnthropicLLM here -- its __init__
    # lazy-imports the anthropic SDK and would try to build a real client.
    # Stub it out to prove get_anthropic_llm() picks the real-backend class
    # without paying for that.
    class _StubAnthropicLLM:
        pass

    monkeypatch.setattr("recipes.llm.AnthropicLLM", _StubAnthropicLLM)
    monkeypatch.setenv("RAG_RECIPES_LLM", "openai")
    assert isinstance(get_anthropic_llm(), _StubAnthropicLLM)


class _FakeContentBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens, cache_read, cache_creation, output_tokens):
        self.input_tokens = input_tokens
        self.cache_read_input_tokens = cache_read
        self.cache_creation_input_tokens = cache_creation
        self.output_tokens = output_tokens


class _FakeAnthropicResponse:
    def __init__(self, text, usage):
        self.content = [_FakeContentBlock(text)]
        self.usage = usage


class _FakeAnthropicMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response):
        self.messages = _FakeAnthropicMessages(response)


def test_anthropic_llm_recombines_cache_token_tiers_into_total_input():
    # Anthropic's usage.input_tokens EXCLUDES cache_read/cache_creation
    # (unlike OpenAI's prompt_tokens, which includes the cached subset) --
    # AnthropicLLM.complete() must recombine them so LLMResponse.input_tokens
    # means "total input tokens" consistently across providers.
    fake_response = _FakeAnthropicResponse(
        text="a context blurb",
        usage=_FakeUsage(input_tokens=50, cache_read=1000, cache_creation=200, output_tokens=20),
    )
    llm = AnthropicLLM.__new__(AnthropicLLM)  # bypass __init__'s lazy SDK import
    llm._client = _FakeAnthropicClient(fake_response)

    result = llm.complete(
        prompt="chunk-specific prompt", model="claude-sonnet-5", cacheable_prefix="the document"
    )

    assert result.text == "a context blurb"
    assert result.input_tokens == 50 + 1000 + 200
    assert result.cached_input_tokens == 1000
    assert result.cache_creation_input_tokens == 200
    assert result.output_tokens == 20

    # The cacheable_prefix must actually be sent as a cached system block.
    sent_kwargs = llm._client.messages.last_kwargs
    assert sent_kwargs["system"] == [
        {"type": "text", "text": "the document", "cache_control": {"type": "ephemeral"}}
    ]
    assert sent_kwargs["messages"] == [{"role": "user", "content": "chunk-specific prompt"}]


def test_anthropic_llm_omits_system_block_when_no_cacheable_prefix():
    fake_response = _FakeAnthropicResponse(
        text="text", usage=_FakeUsage(input_tokens=10, cache_read=0, cache_creation=0, output_tokens=5)
    )
    llm = AnthropicLLM.__new__(AnthropicLLM)
    llm._client = _FakeAnthropicClient(fake_response)

    result = llm.complete(prompt="a prompt", model="claude-sonnet-5")

    assert result.input_tokens == 10
    assert result.cached_input_tokens == 0
    assert result.cache_creation_input_tokens == 0
    assert "system" not in llm._client.messages.last_kwargs
