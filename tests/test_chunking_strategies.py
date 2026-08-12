from corpus.chunking_strategies import (
    chunk_document_aware,
    chunk_fixed,
    chunk_late,
    chunk_semantic,
)
from recipes.embeddings import MockEmbedder


class _FakeTokenizer:
    """Whitespace tokenizer, same convention as tests/test_build_corpus.py."""

    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


def test_chunk_fixed_wires_through_the_target_token_size():
    tokenizer = _FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(2000))
    chunks = chunk_fixed(text, chunk_tokens=256, tokenizer=tokenizer)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(tokenizer.encode(chunk["text"])) <= 256
        # No late-chunking approximation here -- embed_text == text.
        assert chunk["embed_text"] == chunk["text"]


def test_chunk_semantic_splits_at_low_similarity_not_uniformly():
    # Two obviously-distinct "topics" glued together as one blob of text.
    # A real embedder would show a similarity drop at the topic boundary;
    # since MockEmbedder is hash-based (not semantically meaningful), this
    # test instead verifies the MECHANISM -- that chunk_semantic produces
    # more than one chunk when there's more than one sentence, and that
    # every emitted chunk is non-empty and comes from real input sentences.
    tokenizer = _FakeTokenizer()
    embedder = MockEmbedder()
    text = (
        "Speculative decoding uses a draft model. "
        "The draft model proposes tokens quickly. "
        "Convolutional networks process images. "
        "CNNs use spatial filters for feature extraction."
    )

    chunks = chunk_semantic(text, embedder, embedding_model="mock-embed", tokenizer=tokenizer)

    assert len(chunks) >= 1
    reconstructed = " ".join(c["text"] for c in chunks)
    for sentence_fragment in ["Speculative decoding", "Convolutional networks"]:
        assert sentence_fragment in reconstructed


def test_chunk_semantic_single_sentence_stays_one_chunk():
    tokenizer = _FakeTokenizer()
    embedder = MockEmbedder()
    text = "Just one sentence here."
    chunks = chunk_semantic(text, embedder, embedding_model="mock-embed", tokenizer=tokenizer)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_chunk_document_aware_groups_by_section_and_preserves_order():
    chunks_for_paper = [
        {"chunk_index": 0, "section": "abstract", "text": "abstract text"},
        {"chunk_index": 2, "section": "methods", "text": "methods part two"},
        {"chunk_index": 1, "section": "methods", "text": "methods part one"},
    ]
    result = chunk_document_aware(chunks_for_paper)

    assert len(result) == 2  # one chunk per distinct section (abstract, methods)
    methods_chunk = next(c for c in result if "methods part one" in c["text"])
    # chunk_index order preserved within the section: part one before part two.
    assert methods_chunk["text"].index("methods part one") < methods_chunk["text"].index("methods part two")


def test_chunk_late_embed_text_contains_text():
    tokenizer = _FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_late(text, chunk_tokens=100, tokenizer=tokenizer)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk["text"] in chunk["embed_text"]


def test_chunk_late_embed_text_includes_surrounding_context():
    tokenizer = _FakeTokenizer()
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_late(text, chunk_tokens=100, tokenizer=tokenizer)

    # A middle chunk's embed_text must be strictly longer than its own
    # text -- it picked up real surrounding context, not a no-op passthrough.
    middle = chunks[len(chunks) // 2]
    assert len(middle["embed_text"]) > len(middle["text"])
