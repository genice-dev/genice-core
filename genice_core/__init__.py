"""
.. include:: ../README.md
"""

import numpy as np
import networkx as nx
from genice_core.graph_arrays import (
    graph_to_adj,
    digraph_to_arrays,
    arrays_to_directed_edges,
    edges_to_digraph,
)
from genice_core.topology import (
    noodlize,
    split_into_simple_paths,
    connect_matching_paths,
    connect_matching_paths_bfs,
)
from genice_core.dipole import optimize, vector_sum
from genice_core.compat import accept_aliases
from typing import Union, List, Optional, Tuple, Sequence, Literal
from logging import getLogger, DEBUG


def _graph_to_adj(
    g: Union[nx.Graph, Sequence[Sequence[int]]],
    g_format: Optional[Literal["edges", "adjacency"]] = None,
) -> Tuple[int, List[List[int]]]:
    """Convert g to (n_orig, adj). g is nx.Graph, or edge list (default), or adjacency list when g_format='adjacency'."""
    if isinstance(g, nx.Graph):
        return graph_to_adj(g)
    g_list = list(g)
    if not g_list:
        return 0, []
    # List input: default is edge list (standard from user side)
    treat_as_edges = g_format == "edges" or (
        g_format is None and all(len(e) == 2 for e in g_list)
    )
    if treat_as_edges:
        try:
            n_orig = max(max(e) for e in g_list) + 1
            adj = [[] for _ in range(n_orig)]
            for u, v in g_list:
                adj[u].append(v)
                adj[v].append(u)
            return n_orig, adj
        except (TypeError, ValueError):
            if g_format == "edges":
                raise
    # Adjacency list: g[v] = list of neighbor indices（コピーせずそのまま返してメモリ節約）
    n_orig = len(g_list)
    if g_format == "adjacency":
        return n_orig, g_list
    adj = [list(neighbors) for neighbors in g_list]
    return n_orig, adj


def _finally_fixed_edges(
    n_orig: int,
    fixed_out: List[List[int]],
    fixed_in: List[List[int]],
    derived_cycles: List[List[int]],
) -> List[Tuple[int, int]]:
    """Directed edges that are fixed, excluding those that lie in derived_cycles."""
    edges = arrays_to_directed_edges(n_orig, fixed_out, fixed_in)
    cycle_edges = set()
    for cycle in derived_cycles:
        for i in range(len(cycle) - 1):
            cycle_edges.add((cycle[i], cycle[i + 1]))
    return [(u, v) for u, v in edges if (u, v) not in cycle_edges]


def _vertex_positions_array(vertex_positions, n_orig: int) -> np.ndarray:
    """Convert vertex_positions to (n_orig, dim) array. Handles dict (e.g. from nx.spring_layout)."""
    if isinstance(vertex_positions, dict):
        return np.array([vertex_positions[i] for i in range(n_orig)])
    arr = np.asarray(vertex_positions)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _path_edges(paths: List[List[int]]) -> List[Tuple[int, int]]:
    """Directed edges from a list of paths (each path gives consecutive (u,v))."""
    out: List[Tuple[int, int]] = []
    for path in paths:
        for i in range(len(path) - 1):
            out.append((path[i], path[i + 1]))
    return out


def _verify_ice_rules(n: int, edges: List[Tuple[int, int]], fixed_edges: nx.DiGraph) -> None:
    """Assert each node has in_degree and out_degree <= 2 (skip fixed nodes with degree > 2)."""
    in_d = [0] * n
    out_d = [0] * n
    for u, v in edges:
        if 0 <= u < n:
            out_d[u] += 1
        if 0 <= v < n:
            in_d[v] += 1
    for node in range(n):
        if fixed_edges.has_node(node):
            if fixed_edges.in_degree(node) > 2 or fixed_edges.out_degree(node) > 2:
                continue
        assert in_d[node] <= 2, f"{node} in_degree {in_d[node]}"
        assert out_d[node] <= 2, f"{node} out_degree {out_d[node]}"


@accept_aliases(
    vertexPositions="vertex_positions",
    isPeriodicBoundary="is_periodic_boundary",
    dipoleOptimizationCycles="dipole_optimization_cycles",
    fixedEdges="fixed_edges",
    pairingAttempts="pairing_attempts",
    returnEdges="return_edges",
    gFormat="g_format",
)
def ice_graph(
    g: Union[nx.Graph, Sequence[Sequence[int]]],
    vertex_positions: Union[np.ndarray, None] = None,
    is_periodic_boundary: bool = False,
    dipole_optimization_cycles: int = 0,
    fixed_edges: nx.DiGraph = nx.DiGraph(),
    pairing_attempts: int = 100,
    target_pol: Optional[np.ndarray] = np.zeros(3),
    return_edges: bool = False,
    g_format: Optional[Literal["edges", "adjacency"]] = None,
) -> Union[Optional[nx.DiGraph], Optional[List[Tuple[int, int]]]]:
    """Make a digraph that obeys the ice rules.

    Args:
        g: An ice-like undirected graph: nx.Graph, or edge list (default for list input), or
           adjacency list. Edge list = list of [u,v] or (u,v); nodes must be 0..N-1. For adjacency
           list (g[v] = list of neighbors) pass g_format="adjacency".
        vertex_positions (Union[np.ndarray, None], optional): Positions of the vertices, N x 3 array.
        fixed_edges (nx.DiGraph, optional): A digraph of edges whose directions are fixed.
        is_periodic_boundary (bool, optional): If True, positions are in fractional coordinates.
        dipole_optimization_cycles (int, optional): Number of iterations to reduce the net dipole moment.
        target_pol (Optional[np.ndarray], optional): Target polarization (dipole optimization).
        pairing_attempts (int, optional): Maximum attempts to pair up the fixed edges.
        return_edges (bool, optional): If True, return directed edge list instead of nx.DiGraph.
        g_format (str, optional): "edges" or "adjacency" when g is a list. None = auto (treat list of
           pairs as edge list; otherwise adjacency list).

    Returns:
        nx.DiGraph, or list of (tail, head) directed edges if return_edges=True. None if no solution found.
    """
    logger = getLogger()

    if logger.isEnabledFor(DEBUG):
        if vertex_positions is None and (
            is_periodic_boundary or dipole_optimization_cycles != 0 or np.any(target_pol != 0)
        ):
            logger.debug(
                "vertex_positions is None; is_periodic_boundary, "
                "dipole_optimization_cycles, and target_pol are ignored."
            )
        if fixed_edges.size() == 0 and pairing_attempts != 100:
            logger.debug("fixed_edges is empty; pairing_attempts is ignored.")

    n_orig, adj = _graph_to_adj(g, g_format)
    fixed_out, fixed_in = digraph_to_arrays(fixed_edges, n_orig)
    derived_cycles: List[List[int]] = []

    if fixed_edges.number_of_edges() > 0:
        if logger.isEnabledFor(DEBUG):
            for u, v in fixed_edges.edges():
                logger.debug(f"FIXED EDGE {u} {v}")
        for attempt in range(pairing_attempts):
            result = connect_matching_paths_bfs(n_orig, adj, fixed_out, fixed_in)
            if result[0] is not None:
                (fixed_out, fixed_in), derived_cycles = result
                break
            logger.info(f"Attempt {attempt + 1}/{pairing_attempts} failed to connect paths")
        else:
            logger.error(f"Failed to find a solution after {pairing_attempts} attempts")
            return None

    finally_fixed = _finally_fixed_edges(n_orig, fixed_out, fixed_in, derived_cycles)
    n_nodes, adj_noodles = noodlize(n_orig, adj, fixed_out, fixed_in)
    paths = split_into_simple_paths(n_orig, n_nodes, adj_noodles) + derived_cycles

    if vertex_positions is not None:
        pos_arr = _vertex_positions_array(vertex_positions, n_orig)
        dim = pos_arr.shape[1]
        target_pol = np.resize(np.asarray(target_pol, dtype=float).flatten(), dim).copy()
        target_pol -= vector_sum(finally_fixed, pos_arr, is_periodic_boundary)
        paths = optimize(
            paths,
            vertex_positions=pos_arr,
            is_periodic_boundary=is_periodic_boundary,
            dipole_optimization_cycles=dipole_optimization_cycles,
            target_pol=target_pol,
        )

    all_edges = finally_fixed + _path_edges(paths)
    _verify_ice_rules(n_orig, all_edges, fixed_edges)

    if return_edges:
        return all_edges
    return edges_to_digraph(n_orig, all_edges)
