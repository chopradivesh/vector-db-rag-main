"""Search the 16-D demo database with all three algorithms. No Ollama needed.

    python -m examples.demo_search
"""

from src.embeddings.keyword_embedder import text_to_demo_embedding
from src.vector_db.demo_data import load_demo
from src.vector_db.vector_db import VectorDB


def main() -> None:
    db = VectorDB(16)
    load_demo(db)
    print(f"Loaded {len(db)} demo vectors\n")

    for text in ["binary tree", "sushi", "calculus", "basketball"]:
        query = text_to_demo_embedding(text, jitter=0)
        print(f'Query: "{text}"')
        for algo in ("bruteforce", "kdtree", "hnsw"):
            out = db.search(query, k=3, metric="cosine", algo=algo)
            top = ", ".join(f"{item.metadata.split(':')[0]} ({dist:.4f})" for dist, item in out.hits)
            print(f"  {algo:<10} {out.latency_us:>5} us  {top}")
        print()

    info = db.hnsw_info()
    print(f"HNSW: top layer = {info['topLayer']}, nodes per layer = {info['nodesPerLayer']}, "
          f"edges per layer = {info['edgesPerLayer']}")


if __name__ == "__main__":
    main()
