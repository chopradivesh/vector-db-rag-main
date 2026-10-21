"""Brute force is the ground truth; KD-Tree must match it exactly and HNSW
must reach high recall."""

import pytest

from src.indexing.brute_force import BruteForce
from src.indexing.distances import cosine, euclidean, manhattan
from src.indexing.hnsw import HNSW
from src.indexing.kd_tree import KDTree
from tests.conftest import random_items

DIMS = 8
ITEMS = random_items(300, DIMS, seed=1)
QUERIES = [item.embedding for item in random_items(25, DIMS, seed=2)]


def ids(results):
    return [item_id for _, item_id in results]


def build_bf(items=ITEMS):
    bf = BruteForce()
    for it in items:
        bf.insert(it)
    return bf


def test_brute_force_insert_search_remove():
    bf = build_bf(ITEMS[:5])
    target = ITEMS[2]
    assert bf.knn(target.embedding, 1, euclidean) == [(0.0, target.id)]
    res = bf.knn(target.embedding, 3, euclidean)
    assert len(res) == 3 and res == sorted(res)
    bf.remove(target.id)
    assert target.id not in ids(bf.knn(target.embedding, 5, euclidean))


def test_brute_force_k_larger_than_n():
    assert len(build_bf(ITEMS[:3]).knn(QUERIES[0], 10, euclidean)) == 3


@pytest.mark.parametrize("dist_fn", [euclidean, manhattan, cosine])
def test_kdtree_matches_brute_force(dist_fn):
    bf, kd = build_bf(), KDTree(DIMS)
    for it in ITEMS:
        kd.insert(it)
    for q in QUERIES:
        assert ids(kd.knn(q, 5, dist_fn)) == ids(bf.knn(q, 5, dist_fn))


def test_kdtree_rebuild_after_removal():
    kd = KDTree(DIMS)
    kd.rebuild(ITEMS[:50])
    kd.rebuild([it for it in ITEMS[:50] if it.id != 7])
    assert 7 not in ids(kd.knn(ITEMS[6].embedding, 10, euclidean))


def test_kdtree_empty():
    assert KDTree(DIMS).knn(QUERIES[0], 3, euclidean) == []


@pytest.mark.parametrize("dist_fn", [euclidean, cosine])
def test_hnsw_recall(dist_fn):
    bf, hnsw = build_bf(), HNSW(m=16, ef_construction=200)
    for it in ITEMS:
        hnsw.insert(it, dist_fn)
    k, hits = 10, 0
    for q in QUERIES:
        hits += len(set(ids(hnsw.knn(q, k, 50, dist_fn))) & set(ids(bf.knn(q, k, dist_fn))))
    recall = hits / (k * len(QUERIES))
    assert recall >= 0.95, recall


def test_hnsw_finds_exact_match_and_sorted_output():
    hnsw = HNSW()
    for it in ITEMS[:100]:
        hnsw.insert(it, euclidean)
    res = hnsw.knn(ITEMS[40].embedding, 5, 50, euclidean)
    assert res[0] == (0.0, ITEMS[40].id)
    assert res == sorted(res)


def test_hnsw_empty_and_remove():
    hnsw = HNSW()
    assert hnsw.knn(QUERIES[0], 3, 50, euclidean) == []
    for it in ITEMS[:50]:
        hnsw.insert(it, euclidean)
    entry = hnsw.entry_point
    hnsw.remove(entry)
    hnsw.remove(ITEMS[10].id)
    assert len(hnsw) == 48
    found = ids(hnsw.knn(ITEMS[10].embedding, 48, 100, euclidean))
    assert entry not in found and ITEMS[10].id not in found
    for node in hnsw.nodes.values():  # no dangling edges
        for layer in node.neighbors:
            assert entry not in layer and ITEMS[10].id not in layer


def test_hnsw_graph_structure():
    hnsw = HNSW(m=4)
    for it in ITEMS:
        hnsw.insert(it, euclidean)
    info = hnsw.get_info()
    assert info["nodeCount"] == len(ITEMS)
    assert info["nodesPerLayer"][0] == len(ITEMS)          # layer 0 holds everyone
    assert info["nodesPerLayer"] == sorted(info["nodesPerLayer"], reverse=True)  # thinner going up
    for node in hnsw.nodes.values():                        # degree caps respected
        for layer, conns in enumerate(node.neighbors):
            assert len(conns) <= (hnsw.m0 if layer == 0 else hnsw.m)
