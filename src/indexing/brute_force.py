"""Brute-force (flat) k-nearest-neighbour search.

What:  compare the query against EVERY stored vector, sort, take the top k.
Why:   it is always exact, so it is the ground truth we measure KD-Tree and
       HNSW against. For small collections it is also the fastest option.
Input: query vector, k, distance function.
Output: list of (distance, id), closest first.
Time:  O(N·d) distance computations + O(N log N) sort.
Space: O(N) for the temporary result list.
"""

from src.indexing.distances import DistanceFn, Vector
from src.vector_db.models import VectorItem


class BruteForce:
    def __init__(self) -> None:
        self.items: list[VectorItem] = []

    def insert(self, item: VectorItem) -> None:
        self.items.append(item)  # O(1)

    def remove(self, item_id: int) -> None:
        self.items = [v for v in self.items if v.id != item_id]  # O(N)

    def knn(self, query: Vector, k: int, dist_fn: DistanceFn) -> list[tuple[float, int]]:
        results = [(dist_fn(query, item.embedding), item.id) for item in self.items]
        # Tuples sort by distance first, then by id (deterministic tie-break).
        results.sort()
        return results[:k]
