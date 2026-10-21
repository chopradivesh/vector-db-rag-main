import math

import pytest

from src.indexing.distances import cosine, euclidean, get_distance_fn, manhattan


def test_euclidean():
    assert euclidean([0, 0], [3, 4]) == pytest.approx(5.0)
    assert euclidean([1, 2, 3], [1, 2, 3]) == 0.0


def test_manhattan():
    assert manhattan([0, 0], [3, 4]) == pytest.approx(7.0)
    assert manhattan([1, -1], [-1, 1]) == pytest.approx(4.0)


def test_cosine_same_orthogonal_opposite():
    assert cosine([1, 2], [2, 4]) == pytest.approx(0.0)       # same direction, different length
    assert cosine([1, 0], [0, 1]) == pytest.approx(1.0)       # orthogonal
    assert cosine([1, 0], [-1, 0]) == pytest.approx(2.0)      # opposite
    assert cosine([1, 1], [1, 0]) == pytest.approx(1 - 1 / math.sqrt(2))


def test_cosine_zero_vector_is_unrelated():
    assert cosine([0, 0], [1, 1]) == 1.0


def test_get_distance_fn_and_fallback():
    assert get_distance_fn("cosine") is cosine
    assert get_distance_fn("manhattan") is manhattan
    assert get_distance_fn("euclidean") is euclidean
    assert get_distance_fn("unknown") is euclidean  # same fallback as C++
