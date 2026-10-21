import pytest

from src.embeddings.keyword_embedder import text_to_demo_embedding
from src.vector_db.demo_data import DEMO_VECTORS, load_demo
from src.vector_db.vector_db import VectorDB


@pytest.fixture
def db():
    db = VectorDB(16)
    load_demo(db)
    return db


def test_demo_loaded(db):
    assert len(db) == len(DEMO_VECTORS) == 20


@pytest.mark.parametrize("algo", ["bruteforce", "kdtree", "hnsw"])
@pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
def test_search_returns_right_category(db, algo, metric):
    query = text_to_demo_embedding("sushi ramen noodle", jitter=0)
    out = db.search(query, 5, metric, algo)
    assert [item.category for _, item in out.hits] == ["food"] * 5
    assert out.latency_us >= 0


def test_all_algorithms_agree_on_demo(db):
    query = text_to_demo_embedding("binary tree", jitter=0)
    results = {a: [i.id for _, i in db.search(query, 5, "cosine", a).hits]
               for a in ("bruteforce", "kdtree", "hnsw")}
    assert results["bruteforce"] == results["kdtree"] == results["hnsw"]


def test_insert_then_search_and_remove(db):
    vec = [0.5] * 16
    new_id = db.insert("Middle point", "test", vec)
    assert new_id == 21 and len(db) == 21
    for algo in ("bruteforce", "kdtree", "hnsw"):
        assert db.search(vec, 1, "euclidean", algo).hits[0][1].id == new_id
    assert db.remove(new_id) is True
    assert db.remove(new_id) is False
    for algo in ("bruteforce", "kdtree", "hnsw"):
        assert new_id not in [i.id for _, i in db.search(vec, 20, "euclidean", algo).hits]


def test_insert_wrong_dims(db):
    with pytest.raises(ValueError):
        db.insert("bad", "x", [1.0, 2.0])


def test_benchmark(db):
    result = db.benchmark([0.1] * 16, 5)
    assert set(result) == {"bruteforce", "kdtree", "hnsw", "item_count"}
    assert result["item_count"] == 20
