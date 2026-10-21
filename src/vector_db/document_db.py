"""DocumentDB — stores document chunks with their real Ollama embeddings.

Unlike the demo VectorDB, the dimension is not fixed in advance: it is taken
from the first embedding inserted (768 for nomic-embed-text).

Retrieval rules (same as the C++):
  * fewer than 10 chunks -> brute force (exact; HNSW gains nothing that small)
  * otherwise            -> HNSW with ef=50
  * always cosine distance (the right metric for text embeddings)
  * drop hits with distance > max_distance (0.7): too dissimilar to be useful
"""

import threading

from src.indexing.brute_force import BruteForce
from src.indexing.distances import Vector, cosine
from src.indexing.hnsw import HNSW
from src.vector_db.models import DocItem, VectorItem

SMALL_SET_THRESHOLD = 10
DEFAULT_MAX_DISTANCE = 0.7
HNSW_SEARCH_EF = 50


class DocumentDB:
    def __init__(self) -> None:
        self.store: dict[int, DocItem] = {}
        self.hnsw = HNSW(m=16, ef_construction=200)
        self.brute_force = BruteForce()
        self.dims = 0  # set by the first insert
        self._lock = threading.Lock()
        self._next_id = 1

    def __len__(self) -> int:
        with self._lock:
            return len(self.store)

    def _check_dims(self, embedding: Vector) -> None:
        # Added safety check: mixing dimensions (e.g. after switching embedding
        # model) would silently produce garbage distances.
        if self.dims and len(embedding) != self.dims:
            raise ValueError(f"embedding has {len(embedding)} dims, database uses {self.dims}")

    def insert(self, title: str, text: str, embedding: Vector) -> int:
        with self._lock:
            self._check_dims(embedding)
            if self.dims == 0:
                self.dims = len(embedding)
            doc = DocItem(self._next_id, title, text, list(embedding))
            self._next_id += 1
            self.store[doc.id] = doc
            # The indexes work on VectorItem, so wrap the chunk as one.
            as_vector = VectorItem(doc.id, title, "doc", doc.embedding)
            self.hnsw.insert(as_vector, cosine)
            self.brute_force.insert(as_vector)
            return doc.id

    def search(
        self, query: Vector, k: int, max_distance: float = DEFAULT_MAX_DISTANCE
    ) -> list[tuple[float, DocItem]]:
        """Top-k most similar chunks as (cosine distance, chunk), closest first."""
        with self._lock:
            if not self.store:
                return []
            self._check_dims(query)
            if len(self.store) < SMALL_SET_THRESHOLD:
                raw = self.brute_force.knn(query, k, cosine)
            else:
                raw = self.hnsw.knn(query, k, HNSW_SEARCH_EF, cosine)
            return [
                (dist, self.store[doc_id])
                for dist, doc_id in raw
                if doc_id in self.store and dist <= max_distance
            ]

    def remove(self, doc_id: int) -> bool:
        with self._lock:
            if doc_id not in self.store:
                return False
            del self.store[doc_id]
            self.hnsw.remove(doc_id)
            self.brute_force.remove(doc_id)
            return True

    def all(self) -> list[DocItem]:
        with self._lock:
            return list(self.store.values())
