"""Shared helpers for topology package. Array-based graph representation (no NetworkX)."""

from typing import List, Tuple, Optional, Set

from genice_core.graph_arrays import (
    node_to_idx,
    idx_to_node,
    connected_components,
)

__all__ = [
    "node_to_idx",
    "idx_to_node",
    "connected_components",
    "_remove_dummy_nodes",
    "_choose_free_edge",
    "_get_perimeters",
    "_copy_directed",
]


def _remove_dummy_nodes(
    n_orig: int, out_adj: List[List[int]], in_adj: List[List[int]]
) -> None:
    """Remove dummy nodes -1..-4 from the directed graph (in place)."""
    for i in range(n_orig, n_orig + 4):
        out_adj[i].clear()
        in_adj[i].clear()
    for v in range(n_orig):
        out_adj[v] = [w for w in out_adj[v] if 0 <= w < n_orig]
        in_adj[v] = [u for u in in_adj[v] if 0 <= u < n_orig]


def _choose_free_edge(
    n_orig: int,
    adj: List[List[int]],
    out_adj: List[List[int]],
    in_adj: List[List[int]],
    node: int,
    preferred: Optional[Set[int]] = None,
) -> Optional[int]:
    """Find an unfixed edge of the node. If preferred is set, try those neighbors first (e.g. sinks)."""
    import numpy as np

    neis = (list(adj[node]) + [-1, -2, -3, -4])[:4]
    if preferred:
        pref = [n for n in neis if n >= 0 and n in preferred]
        rest = [n for n in neis if n not in pref]
        neis = pref + rest
    np.random.shuffle(neis)
    for nei in neis:
        if nei is None:
            continue
        i_node = node_to_idx(node, n_orig)
        i_nei = node_to_idx(nei, n_orig) if nei >= 0 else n_orig + (-1 - nei)
        has_out = nei in out_adj[i_node]
        has_in = node in out_adj[i_nei]
        if not (has_out or has_in):
            return nei
    return None


def _get_perimeters(
    n_orig: int,
    adj: List[List[int]],
    out_adj: List[List[int]],
    in_adj: List[List[int]],
) -> Tuple[Set[int], Set[int]]:
    """Identify nodes with unbalanced in/out degrees."""
    in_peri: Set[int] = set()
    out_peri: Set[int] = set()
    size = n_orig + 4
    for i in range(size):
        node = idx_to_node(i, n_orig)
        deg_g = len(adj[node]) if 0 <= node < n_orig else 0
        in_d = len(in_adj[i])
        out_d = len(out_adj[i])
        if node < 0 or in_d + out_d >= deg_g:
            continue
        if in_d > out_d:
            out_peri.add(node)
        elif in_d < out_d:
            in_peri.add(node)
    return in_peri, out_peri


def _copy_directed(
    n_orig: int, out_adj: List[List[int]], in_adj: List[List[int]]
) -> Tuple[List[List[int]], List[List[int]]]:
    size = n_orig + 4
    return [list(row) for row in out_adj], [list(row) for row in in_adj]
