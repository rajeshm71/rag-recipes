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


def test_bm25_zero_overlap_returns_no_results():
    index = BM25Index()
    index.build(chunk_ids=["a", "b"], texts=["cat dog", "bird fish"])
    assert index.search("zebra", k=2) == []


def test_bm25_raises_before_build():
    index = BM25Index()
    try:
        index.search("anything")
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_bm25_keeps_matches_whose_score_is_not_positive():
    # Matching chunks can score <= 0 in rank_bm25. They still match the
    # query and must be returned; only chunks sharing no term are excluded.
    # A term in exactly half the chunks has an IDF of exactly 0.
    half = BM25Index()
    half.build(chunk_ids=["a", "b", "c", "d"], texts=["rag one", "rag two", "three", "four"])
    results = half.search("rag", k=4)
    assert {r.chunk_id for r in results} == {"a", "b"}
    assert all(r.score == 0 for r in results)

    # When the corpus-average IDF is negative, so is the floor rank_bm25
    # substitutes for negative IDFs: every match scores below zero.
    small = BM25Index()
    small.build(chunk_ids=["a", "b"], texts=["rag retrieval", "rag reranking"])
    results = small.search("rag", k=2)
    assert {r.chunk_id for r in results} == {"a", "b"}
    assert all(r.score < 0 for r in results)
    assert small.search("zebra", k=2) == []


def test_fold_plural():
    from recipes.bm25 import fold_plural

    assert fold_plural("models") == "model"
    assert fold_plural("studies") == "study"
    assert fold_plural("boxes") == "box"
    assert fold_plural("classes") == "class"
    assert fold_plural("approaches") == "approach"
    # Left alone: -ss/-us/-is endings, short words, and tokens with digits.
    for token in ("loss", "corpus", "analysis", "gas", "gpt4s", "model"):
        assert fold_plural(token) == token


def test_default_tokenization_is_unchanged():
    # Without the options, "models" does not match "model" and "the" still counts.
    index = BM25Index()
    index.build(chunk_ids=["a", "b"], texts=["the model", "a dataset"])
    assert index.search("models", k=2) == []
    assert [r.chunk_id for r in index.search("the", k=2)] == ["a"]


def test_plural_folding_matches_singular_and_plural():
    index = BM25Index(fold_plurals=True)
    index.build(chunk_ids=["a", "b"], texts=["one language model", "two image datasets"])
    assert [r.chunk_id for r in index.search("language models", k=2)] == ["a"]
    assert [r.chunk_id for r in index.search("which dataset", k=2)] == ["b"]


def test_stopwords_do_not_count_as_overlap():
    from recipes.bm25 import STOPWORDS

    index = BM25Index(stopwords=STOPWORDS)
    index.build(chunk_ids=["a", "b"], texts=["the model of the paper", "a speculative decoder"])
    # Only stopwords in common with chunk "a": no match at all.
    assert index.search("what is the", k=2) == []
    assert [r.chunk_id for r in index.search("what is the decoder", k=2)] == ["b"]
