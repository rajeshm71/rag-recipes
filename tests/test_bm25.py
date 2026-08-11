from recipes.bm25 import BM25Index


def test_bm25_exact_keyword_match_ranks_first():
    index = BM25Index()
    index.build(
        chunk_ids=["a", "b", "c"],
        texts=[
            "Speculative decoding speeds up LLM inference significantly.",
            "This paper discusses image classification with CNNs.",
            "Reinforcement learning from human feedback aligns models.",
        ],
    )
    results = index.search("speculative decoding inference", k=2)
    assert results[0].chunk_id == "a"
    assert results[0].score > 0


def test_bm25_respects_k():
    index = BM25Index()
    index.build(
        chunk_ids=["a", "b", "c", "d"],
        texts=["cat dog", "cat bird", "cat fish", "cat mouse"],
    )
    results = index.search("cat", k=2)
    assert len(results) == 2


def test_bm25_raises_before_build():
    index = BM25Index()
    try:
        index.search("anything")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
