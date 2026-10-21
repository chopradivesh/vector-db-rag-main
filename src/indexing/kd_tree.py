"""KD-Tree (k-dimensional tree) for exact nearest-neighbour search.

What:  a binary tree that splits space along one axis per level. Level 0
       splits on dimension 0, level 1 on dimension 1, ... cycling through
       all dimensions (axis = depth % dims).
Why:   during search we can skip ("prune") whole subtrees that cannot
       possibly contain a closer point, which avoids comparing against
       every vector like brute force does.

Insert: O(depth) — O(log N) on a balanced tree, O(N) worst case. The tree is
        built by plain insertion (no median balancing), same as the C++.
Search: ~O(log N) in low dimensions, but degrades towards O(N) in high
        dimensions ("curse of dimensionality") because pruning rarely fires.
Space:  O(N) nodes.

CHANGE vs. the C++ version (bug fix):
    The pruning test uses |query[axis] - node[axis]| as a lower bound on the
    distance to anything on the far side of the split. That bound is only
    valid for Euclidean and Manhattan distance. The C++ code applied it to
    cosine distance too, which can skip the true nearest neighbours. Here we
    only prune for Euclidean/Manhattan; for cosine we visit both sides, which
    makes the result exact (at the cost of checking more nodes).
"""

import heapq
from dataclasses import dataclass

from src.indexing.distances import DistanceFn, Vector, cosine
from src.vector_db.models import VectorItem


@dataclass
class KDNode:
    item: VectorItem
    left: "KDNode | None" = None
    right: "KDNode | None" = None


class KDTree:
    def __init__(self, dims: int) -> None:
        self.dims = dims
        self.root: KDNode | None = None

    # ---------- insert ----------
    def insert(self, item: VectorItem) -> None:
        self.root = self._insert(self.root, item, depth=0)

    def _insert(self, node: KDNode | None, item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item)
        axis = depth % self.dims
        # Smaller value on this axis -> go left, otherwise right.
        if item.embedding[axis] < node.item.embedding[axis]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def rebuild(self, items: list[VectorItem]) -> None:
        """Deleting from a KD-Tree is awkward, so (like the C++) we rebuild
        the whole tree from the remaining items. O(N log N) on average."""
        self.root = None
        for item in items:
            self.insert(item)

    # ---------- search ----------
    def knn(self, query: Vector, k: int, dist_fn: DistanceFn) -> list[tuple[float, int]]:
        # Max-heap of the k best so far. Python's heapq is a MIN-heap, so we
        # store negative distances: heap[0] is then the WORST of the k best.
        heap: list[tuple[float, int]] = []
        can_prune = dist_fn is not cosine  # see module docstring
        self._knn(self.root, query, k, 0, dist_fn, heap, can_prune)
        return sorted((-neg_dist, item_id) for neg_dist, item_id in heap)

    def _knn(
        self,
        node: KDNode | None,
        query: Vector,
        k: int,
        depth: int,
        dist_fn: DistanceFn,
        heap: list[tuple[float, int]],
        can_prune: bool,
    ) -> None:
        if node is None:
            return

        # 1. Consider the current node itself.
        d = dist_fn(query, node.item.embedding)
        if len(heap) < k:
            heapq.heappush(heap, (-d, node.item.id))
        elif d < -heap[0][0]:
            heapq.heapreplace(heap, (-d, node.item.id))  # drop the worst, add this

        # 2. Descend first into the side of the split the query falls on.
        axis = depth % self.dims
        diff = query[axis] - node.item.embedding[axis]
        closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._knn(closer, query, k, depth + 1, dist_fn, heap, can_prune)

        # 3. Only visit the other side if it could hold something closer than
        #    our current k-th best. |diff| is the minimum distance from the
        #    query to ANY point across the splitting plane.
        heap_not_full = len(heap) < k
        if heap_not_full or not can_prune or abs(diff) < -heap[0][0]:
            self._knn(farther, query, k, depth + 1, dist_fn, heap, can_prune)
