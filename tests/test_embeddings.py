from recipes.embeddings import (
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
    get_voyage_embedder,
)


class _FakeRequestsResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_voyage_embedder_maps_response_fields(monkeypatch):
    # Real Voyage response schema (verified 2026-08-12 against
    # docs.voyageai.com/reference/embeddings-api): data items are NOT
    # guaranteed to be in request order, each carries its own "index".
    fake_payload = {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
            {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
        ],
        "model": "voyage-4",
        "usage": {"total_tokens": 42},
    }
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return _FakeRequestsResponse(fake_payload)

    monkeypatch.setattr("requests.post", fake_post)

    embedder = VoyageEmbedder(api_key="test-key")
    result = embedder.embed(["hello", "world"], model="voyage-4")

    # Sorted by "index" -- "hello" (index 0) must map to [0.1, 0.2], not
    # the response's raw (out-of-order) list position.
    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.input_tokens == 42
    assert captured["url"] == "https://api.voyageai.com/v1/embeddings"
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"] == {"input": ["hello", "world"], "model": "voyage-4"}


class _FakeSentenceTransformerModel:
    def __init__(self, vectors):
        self._vectors = vectors

    def encode(self, texts, convert_to_numpy=True):
        import numpy as np

        return np.array(self._vectors[: len(texts)])


def test_local_embedder_maps_response_fields():
    # Bypass __init__'s lazy import/download of the real sentence-transformers
    # model -- real correctness is verified by the A2 notebook's own
    # papermill execution, not by pytest (same precedent as CrossEncoderReranker).
    embedder = LocalEmbedder.__new__(LocalEmbedder)
    embedder._model = _FakeSentenceTransformerModel([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    result = embedder.embed(["a", "b"], model="BAAI/bge-large-en-v1.5")

    assert result.vectors == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert result.input_tokens > 0


class _FakeOpenAIEmbeddingItem:
    def __init__(self, embedding):
        self.embedding = embedding


class _FakeOpenAIUsage:
    def __init__(self, prompt_tokens):
        self.prompt_tokens = prompt_tokens


class _FakeOpenAIEmbeddingResponse:
    def __init__(self, vectors, prompt_tokens):
        self.data = [_FakeOpenAIEmbeddingItem(v) for v in vectors]
        self.usage = _FakeOpenAIUsage(prompt_tokens)


class _FakeOpenAIEmbeddingsEndpoint:
    def __init__(self):
        self.calls: list[list[str]] = []

    def create(self, model, input):
        self.calls.append(list(input))
        return _FakeOpenAIEmbeddingResponse([[float(len(t))] for t in input], prompt_tokens=len(input))


class _FakeOpenAIClient:
    def __init__(self):
        self.embeddings = _FakeOpenAIEmbeddingsEndpoint()


def test_openai_embedder_batches_under_token_budget(monkeypatch):
    # Real bug (found via A1_chunking_study's real-key run): OpenAI's
    # embeddings endpoint rejects a single request over 300_000 tokens
    # (`max_tokens_per_request`). Force a tiny budget here so 5 texts of
    # ~100 estimated tokens each (400 chars // 4) must split into multiple
    # sub-requests, not one.
    monkeypatch.setattr("recipes.embeddings._OPENAI_MAX_TOKENS_PER_REQUEST", 250)

    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    fake_client = _FakeOpenAIClient()
    embedder._client = fake_client

    texts = ["a" * 400 for _ in range(5)]
    result = embedder.embed(texts, model="text-embedding-3-small")

    assert len(fake_client.embeddings.calls) > 1, "should have split into multiple requests"
    assert all(len(call) <= 2 for call in fake_client.embeddings.calls)
    assert len(result.vectors) == 5
    assert result.input_tokens == sum(len(call) for call in fake_client.embeddings.calls)


def test_get_voyage_embedder_respects_env_var(monkeypatch):
    monkeypatch.setenv("RAG_RECIPES_LLM", "mock")
    assert isinstance(get_voyage_embedder(), MockEmbedder)

    class _StubVoyageEmbedder:
        pass

    monkeypatch.setattr("recipes.embeddings.VoyageEmbedder", _StubVoyageEmbedder)
    monkeypatch.setenv("RAG_RECIPES_LLM", "openai")
    assert isinstance(get_voyage_embedder(), _StubVoyageEmbedder)
