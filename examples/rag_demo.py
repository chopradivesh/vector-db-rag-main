"""Full RAG pipeline against a real local Ollama.

Requires:  ollama serve
           ollama pull nomic-embed-text
           ollama pull llama3.2
Run:       python -m examples.rag_demo
"""

from src import config
from src.llm.ollama_client import OllamaClient
from src.rag.pipeline import RAGPipeline
from src.vector_db.document_db import DocumentDB

NOTES = {
    "Operating Systems": "A process is a program in execution with its own address space. "
    "Threads share the address space of their process. Deadlock needs four conditions: "
    "mutual exclusion, hold and wait, no preemption and circular wait.",
    "Databases": "An index speeds up lookups at the cost of extra storage and slower writes. "
    "B-trees keep keys sorted and balanced so search, insert and delete are O(log n).",
    "Cooking": "Risotto is made by slowly adding hot stock to arborio rice while stirring.",
}


def main() -> None:
    ollama = OllamaClient(config.OLLAMA_URL, config.EMBED_MODEL, config.GEN_MODEL)
    if not ollama.is_available():
        print(f"Ollama is not reachable at {config.OLLAMA_URL}. Start it with: ollama serve")
        return

    rag = RAGPipeline(DocumentDB(), ollama)
    for title, text in NOTES.items():
        rag.ingest_document(title, text)
    print(f"Stored {len(rag.doc_db)} chunks ({rag.doc_db.dims}-D embeddings)\n")

    question = "What conditions are needed for a deadlock?"
    result = rag.ask(question, k=3)
    print("Q:", question)
    for dist, doc in result.contexts:
        print(f"  retrieved: {doc.title}  (cosine distance {dist:.3f})")
    print(f"\nA ({result.model}): {result.answer}")


if __name__ == "__main__":
    main()
