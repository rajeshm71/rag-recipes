from recipes.llm import MockLLM


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
