import pytest

from src.rag.pipeline import RAGPipeline, build_prompt
from src.vector_db.document_db import DocumentDB
from src.vector_db.models import DocItem

DOCS = {
    "Python notes": "python is a programming language . python code is easy",
    "Pizza recipe": "pizza needs cheese tomato and a hot oven",
    "Football": "a football team scores a goal in a match",
    "Astronomy": "a planet moves in orbit around a star in space",
}


@pytest.fixture
def rag(fake_llm):
    pipeline = RAGPipeline(DocumentDB(), fake_llm)
    for title, text in DOCS.items():
        pipeline.ingest_document(title, text)
    return pipeline


def test_ingest_stores_chunks(rag):
    assert len(rag.doc_db) == 4
    assert rag.doc_db.dims == 16


def test_ingest_long_document_is_chunked(fake_llm):
    pipeline = RAGPipeline(DocumentDB(), fake_llm, chunk_words=10, overlap_words=2)
    ids = pipeline.ingest_document("Long", " ".join(["python code"] * 20))  # 40 words
    assert len(ids) == 5
    titles = [d.title for d in pipeline.doc_db.all()]
    assert titles[0] == "Long [1/5]" and titles[-1] == "Long [5/5]"


def test_retrieve_finds_relevant_chunk(rag):
    hits = rag.retrieve("how do I bake pizza with cheese", k=3)
    assert hits[0][1].title == "Pizza recipe"
    assert all(dist <= 0.7 for dist, _ in hits)         # max-distance filter


def test_irrelevant_question_retrieves_nothing(rag):
    assert rag.retrieve("completely unrelated words", k=3) == []


def test_ask_builds_prompt_and_calls_llm(rag, fake_llm):
    result = rag.ask("which planet has an orbit around a star", k=2)
    assert result.answer == "fake answer" and result.model == "fake-llm"
    assert result.contexts[0][1].title == "Astronomy"
    prompt = fake_llm.prompts[-1]
    assert "[1] Astronomy:" in prompt
    assert prompt.endswith("Question: which planet has an orbit around a star\n\nAnswer:")


def test_build_prompt_without_context():
    prompt = build_prompt("Q?", [])
    assert "Context:\nQuestion: Q?" in prompt


def test_hnsw_path_used_for_larger_collections(fake_llm):
    """>= 10 chunks switches retrieval from brute force to HNSW."""
    db = DocumentDB()
    pipeline = RAGPipeline(db, fake_llm)
    for i in range(12):
        pipeline.ingest_document(f"doc{i}", "football team goal match" if i == 7 else "planet orbit star space")
    hits = pipeline.retrieve("football match", k=1)
    assert hits[0][1].title == "doc7"


def test_document_db_remove_and_dim_check():
    db = DocumentDB()
    doc_id = db.insert("a", "text", [1.0, 0.0])
    with pytest.raises(ValueError):
        db.insert("b", "text", [1.0, 0.0, 0.0])
    assert db.remove(doc_id) and not db.remove(doc_id)
    assert db.search([1.0, 0.0], 3) == []


def test_failed_embedding_inserts_nothing(fake_llm):
    class Flaky(type(fake_llm)):
        calls = 0

        def embed(self, text):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("ollama died")
            return super().embed(text)

    pipeline = RAGPipeline(DocumentDB(), Flaky(), chunk_words=5, overlap_words=1)
    with pytest.raises(RuntimeError):
        pipeline.ingest_document("x", "python " * 20)
    assert len(pipeline.doc_db) == 0
