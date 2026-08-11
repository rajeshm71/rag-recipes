from recipes.store import VectorStore


def test_store_round_trip_returns_nearest():
    store = VectorStore(dim=4)
    store.add("a", [1.0, 0.0, 0.0, 0.0])
    store.add("b", [0.0, 1.0, 0.0, 0.0])
    store.add("c", [0.9, 0.1, 0.0, 0.0])

    results = store.search([1.0, 0.0, 0.0, 0.0], k=2)

    assert results[0].chunk_id == "a"
    assert {r.chunk_id for r in results} == {"a", "c"}
    store.close()


def test_store_add_many():
    store = VectorStore(dim=2)
    store.add_many(["x", "y"], [[1.0, 0.0], [0.0, 1.0]])
    results = store.search([1.0, 0.0], k=1)
    assert results[0].chunk_id == "x"
    store.close()


def test_store_rejects_wrong_dim():
    store = VectorStore(dim=3)
    try:
        store.add("bad", [1.0, 0.0])
        assert False, "expected ValueError"
    except ValueError:
        pass
    store.close()
