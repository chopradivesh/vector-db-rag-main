"""HNSW — Hierarchical Navigable Small World graph (approximate k-NN).

The algorithm used by production vector databases (Pinecone, Weaviate,
Chroma, Milvus, pgvector...). Paper: Malkov & Yashunin, 2016.

Idea
----
Build a multi-layer graph, like a skip list made of graphs:
  * Layer 0 contains EVERY node, each linked to up to M0 = 2*M neighbours.
  * Each higher layer contains exponentially fewer nodes (long-range "highways").
  * A node's top layer is random: level = floor(-ln(U) * mL), mL = 1/ln(M).
    With M=16, ~94% of nodes live only on layer 0, ~6% reach layer 1, ...

Search: start at the entry point on the top layer, greedily walk towards the
query, drop one layer, repeat. On layer 0 run a wider "beam search" keeping
the `ef` best candidates, then return the top k.

Complexity (N nodes, d dims):
  insert: ~O(log N · ef_construction · M · d)
  search: ~O(log N · ef · M · d)   — approximate: may miss a true neighbour
  space:  O(N · M) edges (+ the vectors themselves)

Parameters (same defaults as the C++):
  M = 16                max neighbours per node on layers >= 1
  M0 = 2*M = 32         max neighbours per node on layer 0
  ef_construction=200   beam width while inserting (higher = better graph, slower build)
  ef (search) = 50      beam width while searching (higher = better recall, slower)

Faithful to the C++ version, including its simplifications:
  * Neighbour selection keeps the M closest candidates (the paper's "simple"
    strategy, not its diversity heuristic).
  * remove() deletes the node and every edge pointing to it but does NOT
    re-link the orphaned neighbours; top_layer is never lowered. Fine for a
    demo, but many deletes will slowly degrade recall.
"""

import heapq
import math
import random
from dataclasses import dataclass, field

from src.indexing.distances import DistanceFn, Vector
from src.vector_db.models import VectorItem


@dataclass
class HNSWNode:
    item: VectorItem
    max_layer: int
    # neighbors[layer] = list of neighbour ids on that layer
    neighbors: list[list[int]] = field(default_factory=list)


class HNSW:
    def __init__(self, m: int = 16, ef_construction: int = 200, seed: int = 42) -> None:
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = ef_construction
        self.level_multiplier = 1.0 / math.log(m)  # "mL" in the paper
        self.nodes: dict[int, HNSWNode] = {}  # id -> node (C++: unordered_map)
        self.top_layer = -1
        self.entry_point: int | None = None
        self.rng = random.Random(seed)  # fixed seed -> reproducible graph

    def __len__(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------
    def _random_level(self) -> int:
        """Exponentially decaying random level.
        We use 1 - random() (range (0, 1]) so log() never sees 0."""
        return math.floor(-math.log(1.0 - self.rng.random()) * self.level_multiplier)

    def _search_layer(
        self, query: Vector, entry: int, ef: int, layer: int, dist_fn: DistanceFn
    ) -> list[tuple[float, int]]:
        """Beam search on ONE layer. Core routine used by both insert and search.

        Input:  query, an entry node id, beam width ef, layer number.
        Output: up to ef (distance, id) pairs, closest first.

        Two heaps:
          candidates — min-heap: the next closest node to expand
          found      — max-heap (negated): the ef best nodes seen so far;
                       found[0] is the WORST of them
        Stop when the closest unexpanded candidate is farther than the worst
        result we already have: nothing reachable from it can improve things.
        """
        d0 = dist_fn(query, self.nodes[entry].item.embedding)
        visited = {entry}
        candidates = [(d0, entry)]
        found = [(-d0, entry)]

        while candidates:
            dist, node_id = heapq.heappop(candidates)
            # -found[0][0] is the distance of the WORST node in `found`.
            if len(found) >= ef and dist > -found[0][0]:
                break
            node = self.nodes[node_id]
            if layer >= len(node.neighbors):
                continue
            for nb in node.neighbors[layer]:
                if nb in visited or nb not in self.nodes:
                    continue
                visited.add(nb)
                nd = dist_fn(query, self.nodes[nb].item.embedding)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(candidates, (nd, nb))
                    heapq.heappush(found, (-nd, nb))
                    if len(found) > ef:
                        heapq.heappop(found)  # evict the current worst

        return sorted((-neg_d, node_id) for neg_d, node_id in found)

    def _greedy_descend(self, query: Vector, from_layer: int, to_layer: int, dist_fn: DistanceFn) -> int:
        """Walk down from `from_layer` to just above `to_layer` with ef=1
        (pure greedy), returning the best entry point for the next phase."""
        ep = self.entry_point
        for layer in range(from_layer, to_layer, -1):
            # After deletions the entry point may not reach this layer; skip it.
            if layer < len(self.nodes[ep].neighbors):
                nearest = self._search_layer(query, ep, 1, layer, dist_fn)
                if nearest:
                    ep = nearest[0][1]
        return ep

    # ------------------------------------------------------------------
    def insert(self, item: VectorItem, dist_fn: DistanceFn) -> None:
        level = self._random_level()
        self.nodes[item.id] = HNSWNode(item, level, [[] for _ in range(level + 1)])

        # First node ever: it simply becomes the entry point.
        if self.entry_point is None:
            self.entry_point = item.id
            self.top_layer = level
            return

        # Phase 1: greedy descent through layers ABOVE the new node's level.
        ep = self._greedy_descend(item.embedding, self.top_layer, level, dist_fn)

        # Phase 2: on each layer the node belongs to, find ef_construction
        # candidates and connect to the closest ones, in both directions.
        for layer in range(min(self.top_layer, level), -1, -1):
            candidates = self._search_layer(item.embedding, ep, self.ef_construction, layer, dist_fn)
            max_conn = self.m0 if layer == 0 else self.m
            selected = [node_id for _, node_id in candidates[:max_conn]]
            self.nodes[item.id].neighbors[layer] = selected

            for nb_id in selected:
                nb = self.nodes.get(nb_id)
                if nb is None:
                    continue
                # Normally nb already has this layer. After a delete, the new
                # entry point can sit on a lower layer than top_layer, so make
                # sure the neighbour's list-of-layers is long enough.
                while len(nb.neighbors) <= layer:
                    nb.neighbors.append([])
                conns = nb.neighbors[layer]
                conns.append(item.id)
                # Neighbour now has too many links: keep only its closest max_conn.
                if len(conns) > max_conn:
                    by_dist = sorted(
                        (dist_fn(nb.item.embedding, self.nodes[c].item.embedding), c)
                        for c in conns
                        if c in self.nodes
                    )
                    nb.neighbors[layer] = [c for _, c in by_dist[:max_conn]]

            if candidates:
                ep = candidates[0][1]  # closest node becomes entry for next layer down

        # The new node reaches higher than anyone before: it becomes the entry point.
        if level > self.top_layer:
            self.top_layer = level
            self.entry_point = item.id

    def knn(self, query: Vector, k: int, ef: int, dist_fn: DistanceFn) -> list[tuple[float, int]]:
        """Approximate k nearest neighbours. Returns (distance, id), closest first."""
        if self.entry_point is None:
            return []
        ep = self._greedy_descend(query, self.top_layer, 0, dist_fn)
        return self._search_layer(query, ep, max(ef, k), 0, dist_fn)[:k]

    def remove(self, item_id: int) -> None:
        """O(N·M): scans every node to drop edges pointing to item_id."""
        if item_id not in self.nodes:
            return
        for node in self.nodes.values():
            for layer, conns in enumerate(node.neighbors):
                if item_id in conns:
                    node.neighbors[layer] = [c for c in conns if c != item_id]
        del self.nodes[item_id]
        if self.entry_point == item_id:
            # Like the C++: pick any remaining node (or none).
            self.entry_point = next(iter(self.nodes), None)

    # ------------------------------------------------------------------
    def get_info(self) -> dict:
        """Graph statistics for the /hnsw-info endpoint (UI layer view).
        An edge a<->b is reported once, from the side with the smaller id."""
        num_layers = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * num_layers
        edges_per_layer = [0] * num_layers
        nodes, edges = [], []
        for node_id, node in self.nodes.items():
            nodes.append(
                {
                    "id": node_id,
                    "metadata": node.item.metadata,
                    "category": node.item.category,
                    "maxLyr": node.max_layer,
                }
            )
            for layer in range(min(node.max_layer + 1, num_layers)):
                nodes_per_layer[layer] += 1
                if layer < len(node.neighbors):
                    for nb in node.neighbors[layer]:
                        if node_id < nb:
                            edges_per_layer[layer] += 1
                            edges.append({"src": node_id, "dst": nb, "lyr": layer})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.nodes),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }
