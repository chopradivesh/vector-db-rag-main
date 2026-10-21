"""VectorDB — the 16-D demo database.

Keeps every vector in ALL THREE indexes at once (brute force, KD-Tree, HNSW)
so the same query can be run through each algorithm and compared.

Thread safety: the web server handles requests on several threads. An insert
touches four structures (store + 3 indexes); without a lock a concurrent
search could see a half-inserted vector. The GIL does not prevent that — it
only makes single bytecode operations atomic — so we use a threading.Lock
exactly like the C++ std::mutex.
"""

import threading
import time
from dataclasses import dataclass

from src.indexing.brute_force import BruteForce
from src.indexing.distances import DistanceFn, Vector, cosine, get_distance_fn
from src.indexing.hnsw import HNSW
from src.indexing.kd_tree import KDTree
from src.vector_db.models import VectorItem

HNSW_SEARCH_EF = 50  # beam width used for HNSW queries (same as C++)


@dataclass
class SearchOutput:
    hits: list[tuple[float, VectorItem]]  # (distance, item), closest first
    latency_us: int  # time spent inside the index, in microseconds


class VectorDB:
    def __init__(self, dims: int = 16) -> None:
        self.dims = dims
        self.store: dict[int, VectorItem] = {}  # id -> item (source of truth)
        self.brute_force = BruteForce()
        self.kd_tree = KDTree(dims)
        self.hnsw = HNSW(m=16, ef_construction=200)
        self._lock = threading.Lock()
        self._next_id = 1

    def __len__(self) -> int:
        with self._lock:
            return len(self.store)

    def insert(self, metadata: str, category: str, embedding: Vector, dist_fn: DistanceFn = cosine) -> int:
        """Add a vector to all indexes. `dist_fn` is the metric the HNSW graph
        is BUILT with (always cosine in the original project)."""
        if len(embedding) != self.dims:
            raise ValueError(f"expected {self.dims}D vector, got {len(embedding)}D")
        with self._lock:
            item = VectorItem(self._next_id, metadata, category, list(embedding))
            self._next_id += 1
            self.store[item.id] = item
            self.brute_force.insert(item)
            self.kd_tree.insert(item)
            self.hnsw.insert(item, dist_fn)
            return item.id

    def remove(self, item_id: int) -> bool:
        with self._lock:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.brute_force.remove(item_id)
            self.hnsw.remove(item_id)
            self.kd_tree.rebuild(list(self.store.values()))
            return True

    def _run_algorithm(self, algo: str, query: Vector, k: int, dist_fn: DistanceFn) -> list[tuple[float, int]]:
        if algo == "bruteforce":
            return self.brute_force.knn(query, k, dist_fn)
        if algo == "kdtree":
            return self.kd_tree.knn(query, k, dist_fn)
        return self.hnsw.knn(query, k, HNSW_SEARCH_EF, dist_fn)  # default: hnsw

    def search(self, query: Vector, k: int, metric: str = "cosine", algo: str = "hnsw") -> SearchOutput:
        dist_fn = get_distance_fn(metric)
        with self._lock:
            start = time.perf_counter_ns()
            raw = self._run_algorithm(algo, query, k, dist_fn)
            latency_us = (time.perf_counter_ns() - start) // 1000
            hits = [(dist, self.store[item_id]) for dist, item_id in raw if item_id in self.store]
        return SearchOutput(hits, latency_us)

    def benchmark(self, query: Vector, k: int, metric: str = "cosine") -> dict[str, int]:
        """Time the same query on all three algorithms (microseconds)."""
        dist_fn = get_distance_fn(metric)
        timings: dict[str, int] = {}
        with self._lock:
            for algo in ("bruteforce", "kdtree", "hnsw"):
                start = time.perf_counter_ns()
                self._run_algorithm(algo, query, k, dist_fn)
                timings[algo] = (time.perf_counter_ns() - start) // 1000
            # len(self.store), not len(self): we already hold self._lock, and
            # len(self) would try to take it again. Lock is not re-entrant -> deadlock.
            timings["item_count"] = len(self.store)
        return timings

    def all(self) -> list[VectorItem]:
        with self._lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self._lock:
            return self.hnsw.get_info()
