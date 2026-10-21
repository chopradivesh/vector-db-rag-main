"""REST API — same routes and JSON shapes as the C++ httplib server, so the
original web UI (static/index.html) works unchanged.

C++ built JSON by hand with string streams; FastAPI + Pydantic do the
parsing/serialisation for us. Handlers are plain `def` (not async): FastAPI
runs them in a thread pool, which is why the databases use locks.
"""

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from src import config
from src.llm.ollama_client import OllamaClient, OllamaError
from src.rag.pipeline import RAGPipeline
from src.vector_db.demo_data import load_demo
from src.vector_db.document_db import DocumentDB
from src.vector_db.vector_db import VectorDB

# This file is src/api/server.py, so parents[2] is the project root folder.
INDEX_HTML = Path(__file__).resolve().parents[2] / "static" / "index.html"


class InsertVectorRequest(BaseModel):
    metadata: str = ""
    category: str = ""
    embedding: list[float] = []


class InsertDocRequest(BaseModel):
    title: str = ""
    text: str = ""


class QuestionRequest(BaseModel):
    question: str = ""
    k: int = 3


def error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def parse_vector(text: str) -> list[float] | None:
    """'0.1,0.2,...' -> [0.1, 0.2, ...]; None if any value is not a number."""
    try:
        return [float(part) for part in text.split(",") if part.strip()]
    except ValueError:
        return None


def create_app(ollama: OllamaClient | None = None) -> FastAPI:
    """Build the app with fresh databases. Tests pass in a fake `ollama`."""
    ollama = ollama or OllamaClient(config.OLLAMA_URL, config.EMBED_MODEL, config.GEN_MODEL)
    db = VectorDB(config.DEMO_DIMS)
    load_demo(db)
    doc_db = DocumentDB()
    rag = RAGPipeline(doc_db, ollama)

    app = FastAPI(title="VectorDB (Python)")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )
    # Exposed for tests / debugging.
    app.state.db = db
    app.state.doc_db = doc_db
    app.state.ollama = ollama

    # ── Demo vector endpoints (16-D) ─────────────────────────────────
    @app.get("/search")
    def search(v: str = "", k: int = Query(5, ge=1), metric: str = "cosine", algo: str = "hnsw"):
        query = parse_vector(v)
        if query is None or len(query) != db.dims:
            return error(f"need {db.dims}D vector")
        out = db.search(query, k, metric, algo)
        return {
            "results": [
                {"id": item.id, "metadata": item.metadata, "category": item.category,
                 "distance": dist, "embedding": item.embedding}
                for dist, item in out.hits
            ],
            "latencyUs": out.latency_us,
            "algo": algo,
            "metric": metric,
        }

    @app.post("/insert")
    def insert(body: InsertVectorRequest):
        if not body.metadata or len(body.embedding) != db.dims:
            return error("invalid body")
        return {"id": db.insert(body.metadata, body.category, body.embedding)}

    @app.delete("/delete/{item_id}")
    def delete(item_id: int):
        return {"ok": db.remove(item_id)}

    @app.get("/items")
    def items():
        return [
            {"id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.embedding}
            for v in db.all()
        ]

    @app.get("/benchmark")
    def benchmark(v: str = "", k: int = Query(5, ge=1), metric: str = "cosine"):
        query = parse_vector(v)
        if query is None or len(query) != db.dims:
            return error(f"need {db.dims}D vector")
        t = db.benchmark(query, k, metric)
        return {"bruteforceUs": t["bruteforce"], "kdtreeUs": t["kdtree"],
                "hnswUs": t["hnsw"], "itemCount": t["item_count"]}

    @app.get("/hnsw-info")
    def hnsw_info():
        return db.hnsw_info()

    @app.get("/stats")
    def stats():
        return {"count": len(db), "dims": db.dims,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"]}

    # ── Document + RAG endpoints ─────────────────────────────────────
    @app.post("/doc/insert")
    def doc_insert(body: InsertDocRequest):
        if not body.title or not body.text:
            return error("need title and text")
        try:
            ids = rag.ingest_document(body.title, body.text)
        except OllamaError as exc:
            return error(
                f"{exc}. Install from https://ollama.com then run: "
                f"ollama pull {ollama.embed_model} && ollama pull {ollama.gen_model}",
                503,
            )
        except ValueError as exc:  # embedding dimension mismatch
            return error(str(exc))
        return {"ids": ids, "chunks": len(ids), "dims": doc_db.dims}

    @app.delete("/doc/delete/{doc_id}")
    def doc_delete(doc_id: int):
        return {"ok": doc_db.remove(doc_id)}

    @app.get("/doc/list")
    def doc_list():
        return [
            {"id": d.id, "title": d.title,
             "preview": d.text[:120] + ("…" if len(d.text) > 120 else ""),
             "words": len(d.text.split())}
            for d in doc_db.all()
        ]

    @app.post("/doc/search")
    def doc_search(body: QuestionRequest):
        """Retrieval only (no LLM) — the UI uses it to highlight hits on the plot."""
        if not body.question:
            return error("need question")
        try:
            hits = rag.retrieve(body.question, body.k)
        except (OllamaError, ValueError) as exc:
            return error(str(exc), 503)
        return {"contexts": [{"id": d.id, "title": d.title, "distance": dist} for dist, d in hits]}

    @app.post("/doc/ask")
    def doc_ask(body: QuestionRequest):
        """Full RAG: embed question -> retrieve -> prompt -> generate."""
        if not body.question:
            return error("need question")
        try:
            result = rag.ask(body.question, body.k)
        except (OllamaError, ValueError) as exc:
            return error(str(exc), 503)
        return {
            "answer": result.answer,
            "model": result.model,
            "contexts": [{"id": d.id, "title": d.title, "text": d.text, "distance": dist}
                         for dist, d in result.contexts],
            "docCount": len(doc_db),
        }

    @app.get("/status")
    def status():
        return {"ollamaAvailable": ollama.is_available(),
                "embedModel": ollama.embed_model, "genModel": ollama.gen_model,
                "docCount": len(doc_db), "docDims": doc_db.dims,
                "demoDims": db.dims, "demoCount": len(db)}

    @app.get("/")
    def index():
        if not INDEX_HTML.exists():
            return error("index.html not found", 404)
        return FileResponse(INDEX_HTML)

    return app
