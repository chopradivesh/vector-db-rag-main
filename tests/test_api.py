"""End-to-end tests of the HTTP API (same routes as the C++ server)."""

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.embeddings.keyword_embedder import text_to_demo_embedding
from src.llm.ollama_client import OllamaError


class FakeOllamaForApi:
    embed_model, gen_model = "fake-embed", "fake-gen"

    def __init__(self, fake_llm, up=True):
        self.llm, self.up = fake_llm, up

    def is_available(self):
        return self.up

    def embed(self, text):
        if not self.up:
            raise OllamaError("Ollama unavailable")
        return self.llm.embed(text)

    def generate(self, prompt):
        return self.llm.generate(prompt)


@pytest.fixture
def client(fake_llm):
    return TestClient(create_app(FakeOllamaForApi(fake_llm)))


def vec_param(text):
    return ",".join(str(x) for x in text_to_demo_embedding(text, jitter=0))


def test_stats_and_items(client):
    assert client.get("/stats").json()["count"] == 20
    items = client.get("/items").json()
    assert len(items) == 20 and len(items[0]["embedding"]) == 16


@pytest.mark.parametrize("algo", ["hnsw", "kdtree", "bruteforce"])
def test_search(client, algo):
    data = client.get(f"/search?v={vec_param('basketball tennis')}&k=3&metric=cosine&algo={algo}").json()
    assert [r["category"] for r in data["results"]] == ["sports"] * 3
    assert data["algo"] == algo and "latencyUs" in data


def test_search_rejects_wrong_dims(client):
    r = client.get("/search?v=1,2,3")
    assert r.status_code == 400 and "16D" in r.json()["error"]


def test_insert_delete(client):
    new_id = client.post("/insert", json={"metadata": "New", "category": "cs", "embedding": [0.3] * 16}).json()["id"]
    assert client.get("/stats").json()["count"] == 21
    assert client.delete(f"/delete/{new_id}").json() == {"ok": True}
    assert client.delete(f"/delete/{new_id}").json() == {"ok": False}
    assert client.post("/insert", json={"metadata": "x", "embedding": [1]}).status_code == 400


def test_benchmark_and_hnsw_info(client):
    bench = client.get(f"/benchmark?v={vec_param('pizza')}").json()
    assert set(bench) == {"bruteforceUs", "kdtreeUs", "hnswUs", "itemCount"}
    info = client.get("/hnsw-info").json()
    assert info["nodeCount"] == 20 and info["nodesPerLayer"][0] == 20


def test_document_rag_flow(client):
    r = client.post("/doc/insert", json={"title": "Pizza", "text": "pizza cheese tomato oven"}).json()
    assert r == {"ids": [1], "chunks": 1, "dims": 16}
    client.post("/doc/insert", json={"title": "Space", "text": "planet orbit star"})
    assert len(client.get("/doc/list").json()) == 2

    ctx = client.post("/doc/search", json={"question": "cheese pizza", "k": 3}).json()["contexts"]
    assert ctx[0]["title"] == "Pizza"

    ans = client.post("/doc/ask", json={"question": "cheese pizza", "k": 3}).json()
    assert ans["answer"] == "fake answer" and ans["model"] == "fake-gen"
    assert ans["contexts"][0]["text"] == "pizza cheese tomato oven" and ans["docCount"] == 2

    assert client.delete("/doc/delete/1").json() == {"ok": True}
    assert client.get("/status").json()["docCount"] == 1


def test_doc_validation(client):
    assert client.post("/doc/insert", json={"title": "", "text": "x"}).status_code == 400
    assert client.post("/doc/ask", json={"question": ""}).json()["error"] == "need question"


def test_ollama_down(fake_llm):
    client = TestClient(create_app(FakeOllamaForApi(fake_llm, up=False)))
    r = client.post("/doc/insert", json={"title": "t", "text": "x"})
    assert r.status_code == 503 and "error" in r.json()
    assert client.get("/status").json()["ollamaAvailable"] is False


def test_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200 and "<html" in r.text.lower()
