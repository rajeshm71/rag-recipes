from recipes.agentic import _parse_agent_action, make_retrieve_and_answer
from recipes.embeddings import MockEmbedder
from recipes.llm import MockLLM

FIXTURE_CORPUS = {
    "c1": {"chunk_id": "c1", "text": "Speculative decoding speeds up LLM inference."},
    "c2": {"chunk_id": "c2", "text": "Convolutional networks are used for image classification."},
    "c3": {"chunk_id": "c3", "text": "Reinforcement learning aligns model outputs with feedback."},
}

# Unique substring that only appears in agentic_prompt.txt, never in
# generation_prompt.txt -- distinguishes agent-decision calls from the
# final R4-templated generation call.
AGENT_MARKER = "retrieval agent searching a corpus"


def test_parse_agent_action_valid_search():
    result = _parse_agent_action('{"thought": "t", "action": "search_dense", "action_input": "q"}')
    assert result == {"thought": "t", "action": "search_dense", "action_input": "q"}


def test_parse_agent_action_valid_finish():
    result = _parse_agent_action('{"thought": "done", "action": "finish", "action_input": ""}')
    assert result["action"] == "finish"


def test_parse_agent_action_invalid_action_name_falls_back_to_finish():
    result = _parse_agent_action('{"thought": "t", "action": "delete_everything", "action_input": ""}')
    assert result["action"] == "finish"


def test_parse_agent_action_garbage_falls_back_to_finish():
    result = _parse_agent_action("not json at all")
    assert result["action"] == "finish"


def test_loop_terminates_on_finish_before_max_iterations():
    llm = MockLLM(
        canned={AGENT_MARKER: '{"thought": "enough", "action": "finish", "action_input": ""}'},
        default_response="final answer",
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm, max_iterations=4)
    fn("some question", k=2)

    # 1 decision call (immediately finishes) + 1 final generation call.
    assert len(llm.calls) == 2


def test_loop_terminates_after_max_iterations_without_finish():
    llm = MockLLM(
        canned={AGENT_MARKER: '{"thought": "keep going", "action": "search_dense", "action_input": "q"}'},
        default_response="final answer",
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm, max_iterations=3)
    fn("some question", k=2)

    # 3 decision calls (never finishes, hits max_iterations) + 1 final
    # generation call.
    assert len(llm.calls) == 3 + 1


def test_tool_call_trace_populated_with_one_entry_per_iteration():
    llm = MockLLM(
        canned={AGENT_MARKER: '{"thought": "searching", "action": "search_bm25", "action_input": "q"}'},
        default_response="final answer",
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm, max_iterations=2)
    result = fn("some question", k=2)

    assert len(result.tool_call_trace) == 2
    for step in result.tool_call_trace:
        assert step["action"] == "search_bm25"
        assert step["observation"] is not None


def test_final_answer_never_comes_from_finish_action_input():
    llm = MockLLM(
        canned={
            AGENT_MARKER: '{"thought": "done", "action": "finish", "action_input": "DISTINCTIVE_FINISH_MARKER"}'
        },
        default_response="the real final answer",
    )
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm)
    result = fn("some question", k=2)

    assert result.answer == "the real final answer"
    assert "DISTINCTIVE_FINISH_MARKER" not in result.answer


def test_malformed_response_mid_loop_forces_finish_without_crashing():
    llm = MockLLM(canned={AGENT_MARKER: "this is not valid JSON"}, default_response="final answer")
    fn = make_retrieve_and_answer(FIXTURE_CORPUS, embedder=MockEmbedder(), llm=llm, max_iterations=4)
    result = fn("some question", k=2)

    # Malformed response -> forced finish on the first iteration -> 1
    # decision call + 1 final generation call, not a crash or infinite loop.
    assert len(llm.calls) == 2
    assert result.answer == "final answer"
