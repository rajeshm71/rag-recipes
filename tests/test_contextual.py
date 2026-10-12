import pytest

from recipes.contextual import _contextualize_corpus, make_retrieve_and_answer
from recipes.embeddings import MockEmbedder
from recipes.llm import MockLLM, UnusableResponseError

FIXTURE_CORPUS = {
    "p1#0": {"chunk_id": "p1#0", "paper_id": "p1", "text": "Paper one, chunk zero: speculative decoding."},
    "p1#1": {"chunk_id": "p1#1", "paper_id": "p1", "text": "Paper one, chunk one: draft model verification."},
    "p2#0": {"chunk_id": "p2#0", "paper_id": "p2", "text": "Paper two, chunk zero: convolutional networks."},
}


def test_contextualize_groups_by_paper_id_for_cacheable_prefix():
    llm = MockLLM(default_response="a context blurb")
    _contextualize_corpus(FIXTURE_CORPUS, llm, model="claude-sonnet-5", cost_log=None)

    # p1 has 2 chunks -- both calls' cacheable_prefix must contain BOTH of
    # p1's chunk texts (the full paper), not p2's text.
    p1_calls = [c for c in llm.calls if "Paper one" in (c.cacheable_prefix or "")]
    assert len(p1_calls) == 2
    for call in p1_calls:
        assert "chunk zero: speculative decoding" in call.cacheable_prefix
        assert "chunk one: draft model verification" in call.cacheable_prefix
        assert "convolutional networks" not in call.cacheable_prefix

    p2_calls = [c for c in llm.calls if "Paper two" in (c.cacheable_prefix or "")]
    assert len(p2_calls) == 1


def test_contextualized_text_is_blurb_plus_original_and_input_unmutated():
    llm = MockLLM(default_response="BLURB")
    original_p1_0_text = FIXTURE_CORPUS["p1#0"]["text"]

    result = _contextualize_corpus(FIXTURE_CORPUS, llm, model="claude-sonnet-5", cost_log=None)

    assert result["p1#0"]["text"] == f"BLURB\n\n{original_p1_0_text}"
    # Original dict untouched.
    assert FIXTURE_CORPUS["p1#0"]["text"] == original_p1_0_text


def test_cost_log_accumulates_one_entry_per_chunk():
    llm = MockLLM(default_response="blurb text")
    cost_log: list[dict] = []
    _contextualize_corpus(FIXTURE_CORPUS, llm, model="claude-sonnet-5", cost_log=cost_log)

    assert len(cost_log) == len(FIXTURE_CORPUS)
    for entry in cost_log:
        assert entry["chunk_id"] in FIXTURE_CORPUS
        assert entry["input_tokens"] > 0
        assert entry["output_tokens"] > 0


def test_cost_log_none_is_safe():
    llm = MockLLM()
    # Must not raise when cost_log is not provided.
    _contextualize_corpus(FIXTURE_CORPUS, llm, model="claude-sonnet-5", cost_log=None)


def test_make_retrieve_and_answer_end_to_end():
    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS,
        embedder=MockEmbedder(),
        llm=MockLLM(default_response="final answer text"),
        anthropic_llm=MockLLM(default_response="a blurb"),
    )
    result = fn("What speeds up inference?", k=2)

    assert result.answer == "final answer text"
    assert len(result.retrieved_chunk_ids) == 2
    assert all(cid in FIXTURE_CORPUS for cid in result.retrieved_chunk_ids)


def test_final_generation_context_uses_original_text_not_blurbed_text():
    anthropic_llm = MockLLM(default_response="DISTINCTIVE_BLURB_MARKER")
    gen_llm = MockLLM(default_response="answer")
    fn = make_retrieve_and_answer(
        FIXTURE_CORPUS, embedder=MockEmbedder(), llm=gen_llm, anthropic_llm=anthropic_llm
    )
    fn("What speeds up inference?", k=3)

    # The generation call's prompt must contain original chunk text but
    # NEVER the blurb marker -- the blurb is embedding-only, not shown to
    # the answering LLM.
    assert len(gen_llm.calls) == 1
    final_prompt = gen_llm.calls[0].prompt
    assert "DISTINCTIVE_BLURB_MARKER" not in final_prompt
    assert any(chunk["text"] in final_prompt for chunk in FIXTURE_CORPUS.values())


class _RefusesOneChunk(MockLLM):
    """Returns a blurb for every chunk except the one whose text contains
    `refused_text`, which comes back unusable (but still billed)."""

    def __init__(self, refused_text):
        super().__init__(default_response="BLURB")
        self.refused_text = refused_text

    def complete(self, prompt, model, temperature=0.0, max_tokens=1024, cacheable_prefix=None):
        response = super().complete(prompt, model, temperature, max_tokens, cacheable_prefix)
        if self.refused_text in prompt:
            raise UnusableResponseError("no usable text (stop_reason='refusal')", response)
        return response


def test_unusable_response_keeps_original_text_and_is_still_counted():
    llm = _RefusesOneChunk(refused_text="draft model verification")
    cost_log: list[dict] = []

    with pytest.warns(UserWarning, match="1 of 3 chunks kept their original text"):
        result = _contextualize_corpus(FIXTURE_CORPUS, llm, model="claude-sonnet-5", cost_log=cost_log)

    assert result["p1#1"]["text"] == FIXTURE_CORPUS["p1#1"]["text"]
    assert result["p1#0"]["text"].startswith("BLURB\n\n")
    assert result["p2#0"]["text"].startswith("BLURB\n\n")
    by_id = {e["chunk_id"]: e for e in cost_log}
    assert len(by_id) == 3
    assert by_id["p1#1"]["contextualized"] is False
    assert by_id["p1#1"]["input_tokens"] > 0
    assert by_id["p1#0"]["contextualized"] is True
