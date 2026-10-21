"""Compare the three algorithms on random vectors as N and d grow.
Shows why brute force is fine for small N, and why KD-Tree loses its
advantage in high dimensions while HNSW keeps it (at a small recall cost).

    python -m examples.benchmark
"""

import random
import time

from src.indexing.brute_force import BruteForce
from src.indexing.distances import euclidean
from src.indexing.hnsw import HNSW
from src.indexing.kd_tree import KDTree
from src.vector_db.models import VectorItem


def run(n: int, dims: int, k: int = 10, queries: int = 30) -> None:
    rng = random.Random(0)
    items = [VectorItem(i, "", "", [rng.random() for _ in range(dims)]) for i in range(n)]
    qs = [[rng.random() for _ in range(dims)] for _ in range(queries)]
    bf, kd, hnsw = BruteForce(), KDTree(dims), HNSW()
    for it in items:
        bf.insert(it)
        kd.insert(it)
        hnsw.insert(it, euclidean)

    def timed(fn):
        start = time.perf_counter()
        results = [fn(q) for q in qs]
        return (time.perf_counter() - start) / queries * 1e6, results

    bf_us, truth = timed(lambda q: bf.knn(q, k, euclidean))
    kd_us, _ = timed(lambda q: kd.knn(q, k, euclidean))
    hn_us, approx = timed(lambda q: hnsw.knn(q, k, 50, euclidean))
    recall = sum(len({i for _, i in a} & {i for _, i in t}) for a, t in zip(approx, truth)) / (k * queries)
    print(f"N={n:<5} d={dims:<4} brute={bf_us:8.0f}us  kdtree={kd_us:8.0f}us  "
          f"hnsw={hn_us:8.0f}us  hnsw recall@{k}={recall:.2f}")


if __name__ == "__main__":
    for n, d in [(1000, 4), (5000, 4), (5000, 64), (5000, 256)]:
        run(n, d)
