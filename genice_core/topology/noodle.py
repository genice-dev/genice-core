"""Noodle graph: divide vertices and split into simple paths/cycles."""

from logging import getLogger
from typing import List, Optional, Set, Tuple

from genice_core.topology._shared import (
    node_to_idx,
    connected_components,
)
import numpy as np


def _trace_path(
    n_nodes: int,
    adj: List[List[int]],
    path: List[int],
    vertex_set: Optional[Set[int]] = None,
) -> List[int]:
    """Trace the path in a linear or cyclic graph."""
    vs = vertex_set or set(range(n_nodes))
    while True:
        last, head = path[-2], path[-1]
        next_node = None
        for w in adj[head]:
            if w in vs and w != last:
                next_node = w
                break
        if next_node is None:
            return path
        path.append(next_node)
        if next_node == path[0]:
            return path


def _find_path(
    n_nodes: int,
    adj: List[List[int]],
    vertex_set: List[int],
) -> List[int]:
    """Find a path in a linear or cyclic graph. vertex_set is the connected component."""
    vs = set(vertex_set)
    if not vs:
        return []
    head = vertex_set[0]
    neighbors = [w for w in adj[head] if w in vs]
    if len(neighbors) == 0:
        return []
    if len(neighbors) == 1:
        return _trace_path(n_nodes, adj, [head, neighbors[0]], vs)
    c0 = _trace_path(n_nodes, adj, [head, neighbors[0]], vs)
    if c0[-1] == head:
        return c0
    c1 = _trace_path(n_nodes, adj, [head, neighbors[1]], vs)
    return c0[::-1] + c1[1:]


def _divide(
    n_nodes: int,
    adj: List[List[int]],
    vertex: int,
    offset: int,
) -> None:
    """Divide a vertex into two vertices and redistribute edges. Modifies adj in place."""
    nei = (list(adj[vertex]) + [None, None, None, None])[:4]
    valid = [x for x in nei if x is not None]
    migrants = set(np.random.choice(valid, 2, replace=False))
    new_vertex = vertex + offset
    for migrant in migrants:
        adj[migrant].remove(vertex)
        adj[vertex].remove(migrant)
        adj[new_vertex].append(migrant)
        adj[migrant].append(new_vertex)


def noodlize(
    n_orig: int,
    adj: List[List[int]],
    fixed_out: List[List[int]],
    fixed_in: List[List[int]],
) -> Tuple[int, List[List[int]]]:
    """Divide each vertex and make a set of paths. Returns (n_nodes, adj)."""
    n_nodes = 2 * n_orig
    adj_noodles = [list(adj[v]) for v in range(n_orig)] + [[] for _ in range(n_orig)]

    for u in range(n_orig):
        iu = node_to_idx(u, n_orig)
        for v in fixed_out[iu]:
            if 0 <= v < n_orig and v in adj_noodles[u]:
                adj_noodles[u].remove(v)
                adj_noodles[v].remove(u)

    for v in range(n_orig):
        nfixed = len(fixed_out[node_to_idx(v, n_orig)]) + len(
            fixed_in[node_to_idx(v, n_orig)]
        )
        if nfixed == 0:
            _divide(n_nodes, adj_noodles, v, n_orig)

    return n_nodes, adj_noodles


def _decompose_complex_path(path: List[int]):
    """Divide a complex path with self-crossings into simple cycles and paths."""
    logger = getLogger()
    if len(path) == 0:
        return
    logger.debug(f"decomposing {path}...")
    order: dict = {}
    order[path[0]] = 0
    store = [path[0]]
    headp = 1
    while headp < len(path):
        node = path[headp]
        if node in order:
            size = len(order) - order[node]
            cycle = store[-size:] + [node]
            yield cycle
            for v in cycle[1:]:
                del order[v]
            store = store[:-size]
        order[node] = len(order)
        store.append(node)
        headp += 1
    if len(store) > 1:
        yield store
    logger.debug("Done decomposition.")


def split_into_simple_paths(
    n_orig: int,
    n_nodes: int,
    adj: List[List[int]],
) -> List[List[int]]:
    """Yield simple paths and cycles from the noodle graph."""
    components = connected_components(n_nodes, adj)
    result: List[List[int]] = []
    for vertice_set in components:
        path = _find_path(n_nodes, adj, vertice_set)
        flatten = [v % n_orig for v in path]
        result.extend(_decompose_complex_path(flatten))
    return result
