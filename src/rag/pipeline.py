"""RAG (Retrieval-Augmented Generation) pipeline.

Ingest (once per document):
    text --chunk_text--> chunks --embed--> vectors --DocumentDB.insert--> HNSW index

Ask (per question):
    1. embed the question          (same embedding model as the chunks!)
    2. retrieve top-k chunks       (DocumentDB.search, cosine, max distance 0.7)
    3. build a prompt              (numbered context blocks + question)
    4. generate the answer         (LLM)

Why RAG? The LLM never saw your notes during training. Instead of
fine-tuning it, we look up the relevant passages at question time and paste
them into the prompt.

The `llm` argument is typed as OllamaClient, but Python does not enforce
type hints: tests pass a fake object that simply has the same embed(),
generate() and gen_model members ("duck typing").
"""

from dataclasses import dataclass

from src.llm.ollama_client import OllamaClient
from src.rag.chunker import chunk_text
from src.vector_db.document_db import DocumentDB
from src.vector_db.models import DocItem


@dataclass
class RAGAnswer:
    answer: str
    model: str
    contexts: list[tuple[float, DocItem]]  # (distance, chunk) used as context


def build_prompt(question: str, contexts: list[tuple[float, DocItem]]) -> str:
    """Same prompt text as the C++ version.

    NOTE: this prompt tells the model to fall back on its OWN knowledge when
    the context is not relevant. So answers are NOT guaranteed to come only
    from your documents (the original README claimed they were).
    """
    context_block = "".join(
        f"[{i}] {doc.title}:\n{doc.text}\n\n" for i, (_, doc) in enumerate(contexts, start=1)
    )
    return (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say "
        "things like 'the context doesn't mention'. "
        "Just answer the question naturally.\n\n"
        f"Context:\n{context_block}Question: {question}\n\nAnswer:"
    )


class RAGPipeline:
    def __init__(self, doc_db: DocumentDB, llm: OllamaClient,
                 chunk_words: int = 250, overlap_words: int = 30) -> None:
        self.doc_db = doc_db
        self.llm = llm
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words

    def ingest_document(self, title: str, text: str) -> list[int]:
        """Chunk, embed and store a document. Returns the new chunk ids.

        CHANGE vs. C++: all chunks are embedded BEFORE any is stored, so an
        Ollama failure half-way does not leave a partially inserted document.
        """
        chunks = chunk_text(text, self.chunk_words, self.overlap_words)
        embeddings = [self.llm.embed(chunk) for chunk in chunks]

        ids = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), start=1):
            chunk_title = f"{title} [{i}/{len(chunks)}]" if len(chunks) > 1 else title
            ids.append(self.doc_db.insert(chunk_title, chunk, embedding))
        return ids

    def retrieve(self, question: str, k: int = 3) -> list[tuple[float, DocItem]]:
        """Steps 1-2: embed the question and find the k closest chunks."""
        return self.doc_db.search(self.llm.embed(question), k)

    def ask(self, question: str, k: int = 3) -> RAGAnswer:
        """Full pipeline: retrieve -> build prompt -> generate."""
        contexts = self.retrieve(question, k)
        answer = self.llm.generate(build_prompt(question, contexts))
        return RAGAnswer(answer=answer, model=self.llm.gen_model, contexts=contexts)
