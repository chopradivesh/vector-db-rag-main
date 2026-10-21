"""Thin HTTP client for a local Ollama server (https://ollama.com).

Ollama serves both models this project needs:
  * embed()    -> POST /api/embeddings  (nomic-embed-text: text -> 768 floats)
  * generate() -> POST /api/generate    (llama3.2: prompt -> answer text)
They share one client because they share a host and connection settings,
but the RAG pipeline calls them as two separate steps.

CHANGE vs. C++: failures raise OllamaError instead of returning an empty
vector / an "ERROR: ..." string disguised as an answer. The API layer turns
the exception into a proper {"error": ...} response.
"""

import httpx


class OllamaError(RuntimeError):
    """Ollama is not running, the model is missing, or the reply was malformed."""


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        embed_model: str = "nomic-embed-text",
        gen_model: str = "llama3.2",
        transport: httpx.BaseTransport | None = None,  # tests inject a fake transport
    ) -> None:
        self.embed_model = embed_model
        self.gen_model = gen_model
        self._http = httpx.Client(base_url=base_url, transport=transport)

    def is_available(self) -> bool:
        try:
            return self._http.get("/api/tags", timeout=2.0).status_code == 200
        except httpx.HTTPError:
            return False

    def _post(self, path: str, payload: dict, read_timeout: float) -> dict:
        timeout = httpx.Timeout(read_timeout, connect=3.0)
        try:
            response = self._http.post(path, json=payload, timeout=timeout)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama unavailable ({exc}). Run: ollama serve") from exc
        if response.status_code != 200:
            raise OllamaError(f"Ollama returned HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    def embed(self, text: str) -> list[float]:
        data = self._post("/api/embeddings", {"model": self.embed_model, "prompt": text}, read_timeout=30.0)
        embedding = data.get("embedding")
        if not embedding:
            raise OllamaError(
                f"No embedding returned. Is the model pulled? Run: ollama pull {self.embed_model}"
            )
        return [float(x) for x in embedding]

    def generate(self, prompt: str) -> str:
        # LLMs on a laptop CPU can take a while, hence the long read timeout.
        data = self._post(
            "/api/generate",
            {"model": self.gen_model, "prompt": prompt, "stream": False},
            read_timeout=180.0,
        )
        return data.get("response", "")
