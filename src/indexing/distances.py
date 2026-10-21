"""Distance metrics.

Every search algorithm is metric-agnostic: it receives a `dist_fn(a, b)` and
only assumes "smaller = more similar". All three functions below are O(d)
time and O(1) extra space, where d = number of dimensions.

Implemented by hand (not numpy/scipy) on purpose, to mirror the original C++
and keep the math visible.
"""

import math
from typing import Callable

Vector = list[float]
DistanceFn = Callable[[Vector, Vector], float]


def euclidean(a: Vector, b: Vector) -> float:
    """Straight-line (L2) distance: sqrt(sum((a_i - b_i)^2))."""
    total = 0.0
    for x, y in zip(a, b):
        diff = x - y
        total += diff * diff
    return math.sqrt(total)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine *distance* = 1 - cosine similarity.

    Only the angle between the vectors matters, not their length, which is
    why it is the standard metric for text embeddings.
    Range: 0 (same direction) .. 1 (orthogonal) .. 2 (opposite).
    A zero vector has no direction, so we return 1.0 ("unrelated") instead
    of dividing by zero — same as the C++ version.
    """
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def manhattan(a: Vector, b: Vector) -> float:
    """City-block (L1) distance: sum(|a_i - b_i|)."""
    return sum(abs(x - y) for x, y in zip(a, b))


DISTANCE_FUNCTIONS: dict[str, DistanceFn] = {
    "euclidean": euclidean,
    "cosine": cosine,
    "manhattan": manhattan,
}


def get_distance_fn(metric: str) -> DistanceFn:
    """Look up a metric by name. Unknown names fall back to Euclidean (as in C++)."""
    return DISTANCE_FUNCTIONS.get(metric, euclidean)
