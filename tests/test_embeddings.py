"""Embedding integration: the keyword demo embedder and the Ollama client
(with a mocked HTTP transport, plus an optional live test)."""

import json

import httpx
import pytest

from src import config
from src.embeddings.keyword_embedder import text_to_demo_embedding
from src.llm.ollama_client import OllamaClient, OllamaError


def test_keyword_embedder_topics():
    emb = text_to_demo_embedding("binary tree algorithm", jitter=0)
    assert len(emb) == 16
    assert min(emb[0:4]) > 0.7                 # CS block lit up
    assert max(emb[4:16]) == pytest.approx(0.08)  # everything else at baseline


def test_keyword_embedder_unknown_text():
    assert text_to_demo_embedding("xyzzy", jitter=0) == [0.08] * 16


def mock_client(handler):
    return OllamaClient("http://ollama.test", "embed-m", "gen-m", transport=httpx.MockTransport(handler))


def test_ollama_embed_request_and_parse():
    def handler(request):
        assert request.url.path == "/api/embeddings"
        assert json.loads(request.content) == {"model": "embed-m", "prompt": "hi"}
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    assert mock_client(handler).embed("hi") == [0.1, 0.2, 0.3]


def test_ollama_generate_request_and_parse():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/generate"
        assert body == {"model": "gen-m", "prompt": "Q?", "stream": False}
        return httpx.Response(200, json={"response": "A!"})

    assert mock_client(handler).generate("Q?") == "A!"


def test_ollama_errors():
    with pytest.raises(OllamaError):
        mock_client(lambda r: httpx.Response(404, text="model not found")).embed("x")
    with pytest.raises(OllamaError):
        mock_client(lambda r: httpx.Response(200, json={})).embed("x")

    def down(request):
        raise httpx.ConnectError("refused")

    client = mock_client(down)
    assert client.is_available() is False
    with pytest.raises(OllamaError):
        client.generate("x")


# ---- optional: real Ollama (skipped when it is not running) ----
live = OllamaClient(config.OLLAMA_URL, config.EMBED_MODEL, config.GEN_MODEL)


@pytest.mark.skipif(not live.is_available(), reason="Ollama is not running")
def test_live_ollama_embedding():
    emb = live.embed("vector databases use HNSW")
    assert len(emb) > 100
