"""
.. include:: ../README.md
"""

import numpy as np
import networkx as nx
from genice_core.topology.graph_arrays import (
    graph_to_adj,
    arrays_to_directed_edges,
)
from genice_core.topology_nx import (
    noodlize,
    split_into_simple_paths,
    connect_matching_paths,
    force_polarize,
)
from genice_core.dipole import optimize, vector_sum
from genice_core.compat import accept_aliases
from typing import Callable, Union, List, Optional, Tuple, Sequence, Literal, Set
# from collections import defaultdict
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


def _verify_ice_rules(
    n: int, edges: List[Tuple[int, int]], fixed_edges: nx.DiGraph
) -> None:
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


# === NetworkX-based core topology helpers (restored from nx tag) ===

def _trace_path_nx(g: nx.Graph, path: List[int]) -> List[int]:
    """Trace the path in a linear or cyclic graph."""
    while True:
        # look at the head of the path
        last, head = path[-2:]
        for next_node in g[head]:
            if next_node != last:
                # go ahead
                break
        else:
            # no next node
            return path
        path.append(next_node)
        if next_node == path[0]:
            # is cyclic
            return path


def _find_path_nx(g: nx.Graph) -> List[int]:
    """Find a path in a linear or cyclic graph."""
    nodes = list(g.nodes())
    if not nodes:
        return []
    # choose one node
    head = nodes[0]
    # look neighbors
    neighbors = list(g[head])
    if len(neighbors) == 0:
        # isolated node
        return []
    if len(neighbors) == 1:
        # head is an end node, fortunately.
        return _trace_path_nx(g, [head, neighbors[0]])
    # look forward
    c0 = _trace_path_nx(g, [head, neighbors[0]])

    if c0[-1] == head:
        # cyclic graph
        return c0

    # look backward
    c1 = _trace_path_nx(g, [head, neighbors[1]])
    return c0[::-1] + c1[1:]


def _divide_nx(g: nx.Graph, vertex: int, offset: int) -> None:
    """Divide a vertex into two vertices and redistribute edges."""
    # fill by Nones if number of neighbors is less than 4
    nei = (list(g[vertex]) + [None, None, None, None])[:4]

    # two neighbor nodes that are passed away to the new node
    migrants = set(np.random.choice(nei, 2, replace=False)) - {None}

    # new node label
    new_vertex = vertex + offset

    # assemble edges
    for migrant in migrants:
        g.remove_edge(migrant, vertex)
        g.add_edge(new_vertex, migrant)


def noodlize_nx(g: nx.Graph, fixed: Optional[nx.DiGraph] = None) -> nx.Graph:
    """NetworkX version: divide each vertex and make a set of paths."""
    logger = getLogger()

    if fixed is None:
        fixed = nx.DiGraph()

    g_fix = nx.Graph(fixed)  # undirected copy

    offset = len(g)

    # divided graph
    g_noodles = nx.Graph(g)
    for edge in fixed.edges():
        g_noodles.remove_edge(*edge)

    for v in g:
        if g_fix.has_node(v):
            nfixed = g_fix.degree[v]
        else:
            nfixed = 0
        if nfixed == 0:
            _divide_nx(g_noodles, v, offset)

    logger.debug("noodlize_nx finished.")
    return g_noodles


def _decompose_complex_path_nx(path: List[int]) -> List[List[int]]:
    """Divide a complex path with self-crossings into simple cycles and paths."""
    logger = getLogger()
    if len(path) == 0:
        return []
    logger.debug(f"decomposing {path}...")
    order: dict = {}
    order[path[0]] = 0
    store = [path[0]]
    headp = 1
    out: List[List[int]] = []
    while headp < len(path):
        node = path[headp]

        if node in order:
            # it is a cycle!
            size = len(order) - order[node]
            cycle = store[-size:] + [node]
            out.append(cycle)

            # remove them from the order[]
            for v in cycle[1:]:
                del order[v]

            # truncate the store
            store = store[:-size]

        order[node] = len(order)
        store.append(node)
        headp += 1
    if len(store) > 1:
        out.append(store)
    logger.debug("Done decomposition.")
    return out


def split_into_simple_paths_nx(nnode: int, g_noodles: nx.Graph) -> List[List[int]]:
    """Simplify noodle graph into simple paths/cycles (NetworkX version)."""
    paths: List[List[int]] = []
    for vertice_set in nx.connected_components(g_noodles):
        # a component of c is either a chain or a cycle.
        g_noodle = g_noodles.subgraph(vertice_set)

        # Find a simple path in the doubled graph
        # It must be a simple path or a simple cycle.
        path = _find_path_nx(g_noodle)

        # Flatten then path. It may make the path self-crossing.
        flatten = [v % nnode for v in path]

        # Divide a long path into simple paths and cycles.
        paths.extend(_decompose_complex_path_nx(flatten))
    return paths


def _remove_dummy_nodes_nx(g: Union[nx.Graph, nx.DiGraph]) -> None:
    """Remove dummy nodes -1..-4 from the graph (in place)."""
    for i in range(-1, -5, -1):
        if g.has_node(i):
            g.remove_node(i)


def _choose_free_edge_nx(g: nx.Graph, dg: nx.DiGraph, node: int) -> Optional[int]:
    """Find an unfixed edge of the node in NetworkX representation."""
    # add dummy nodes to make number of edges be four.
    neis = (list(g[node]) + [-1, -2, -3, -4])[:4]
    # and select one randomly
    np.random.shuffle(neis)
    for nei in neis:
        if not (dg.has_edge(node, nei) or dg.has_edge(nei, node)):
            return nei
    return None


def _get_perimeters_nx(
    fixed: nx.DiGraph, g: nx.Graph
) -> Tuple[Set[int], Set[int]]:
    """Identify nodes with unbalanced in/out degrees in NetworkX representation."""
    in_peri: Set[int] = set()
    out_peri: Set[int] = set()
    for node in fixed:
        # If the node has unfixed edges,
        if fixed.in_degree(node) + fixed.out_degree(node) < g.degree(node):
            # if it is not balanced,
            if fixed.in_degree(node) > fixed.out_degree(node):
                out_peri.add(node)
            elif fixed.in_degree(node) < fixed.out_degree(node):
                in_peri.add(node)
    return in_peri, out_peri


def connect_matching_paths_nx(
    fixed: nx.DiGraph, g: nx.Graph
) -> Tuple[Optional[nx.DiGraph], List[List[int]]]:
    """NetworkX version of connect_matching_paths (restored from nx tag)."""
    logger = getLogger()

    # Make a copy to keep the original graph untouched
    _fixed = nx.DiGraph(fixed)

    in_peri, out_peri = _get_perimeters_nx(_fixed, g)

    logger.debug(f"out_peri {out_peri}")
    logger.debug(f"in_peri {in_peri}")

    derived_cycles: List[List[int]] = []

    # Process out_peri nodes
    while out_peri:
        node = np.random.choice(list(out_peri))
        out_peri.remove(node)

        path = [node]
        while True:
            if node < 0:
                # Path search completed.
                logger.debug(f"Dead end at {node}. Path is {path}.")
                break
            if node in in_peri:
                # Path search completed.
                logger.debug(f"Reach at a perimeter node {node}. Path is {path}.")
                # in_peri and out_peri are now pair-annihilated.
                in_peri.remove(node)
                break
            if node in out_peri:
                logger.debug(f"node {node} is on the out_peri...")

            # if the node can no longer be balanced,
            if max(_fixed.in_degree(node), _fixed.out_degree(node)) * 2 > 4:
                # Start over.
                logger.info("Failed to balance. Starting over ...")
                return None, []

            if g.degree(node) == _fixed.degree(node):
                # Start over.
                logger.info(f"node {node} has no free edge. Starting over ...")
                return None, []

            # Find the next node. That may be a decorated one.
            next_node = _choose_free_edge_nx(g, _fixed, node)
            if next_node is None:
                logger.info(
                    f"node {node} has no free edge (unexpected). Starting over ..."
                )
                return None, []

            # fix the edge
            _fixed.add_edge(node, next_node)

            # record to the path
            if next_node >= 0:
                path.append(next_node)
                # if still incoming edges are more than outgoing ones,
                if _fixed.in_degree(node) > _fixed.out_degree(node):
                    # It is still a perimeter.
                    out_peri.add(node)

            # go ahead
            node = next_node

            # if it is circular
            if node in path[:-1]:
                try:
                    loc = path.index(node)
                    # Separate the cycle from the path and store in derivedCycles.
                    derived_cycles.append(path[loc:])
                    # and shorten the path
                    path = path[: loc + 1]
                except ValueError:
                    pass

    # Process in_peri nodes
    while in_peri:
        node = np.random.choice(list(in_peri))
        in_peri.remove(node)
        logger.debug(
            f"first node {node}, its neighbors {list(g[node])} "
            f"{list(_fixed.successors(node))} {list(_fixed.predecessors(node))}"
        )

        path = [node]
        while True:
            if node < 0:
                # Path search completed.
                logger.debug(f"Dead end at {node}. Path is {path} {in_peri}.")
                break
            if node in out_peri:
                # Path search completed.
                logger.debug(f"Reach at a perimeter node {node}. Path is {path}.")
                # in_peri and out_peri are now pair-annihilated.
                out_peri.remove(node)
                break
            if node in in_peri:
                logger.debug(f"node {node} is on the in_peri...")

            if max(_fixed.in_degree(node), _fixed.out_degree(node)) * 2 > 4:
                logger.info("Failed to balance. Starting over ...")
                return None, []

            if g.degree(node) == _fixed.degree(node):
                # Start over.
                logger.info(f"node {node} has no free edge. Starting over ...")
                return None, []

            next_node = _choose_free_edge_nx(g, _fixed, node)
            if next_node is None:
                logger.info(
                    f"node {node} has no free edge (unexpected). Starting over ..."
                )
                return None, []

            # record to the path
            if next_node >= 0:
                path.append(next_node)

            # fix the edge
            _fixed.add_edge(next_node, node)

            # if still incoming edges are more than outgoing ones,
            if next_node >= 0:
                if _fixed.in_degree(node) < _fixed.out_degree(node):
                    in_peri.add(node)
                    logger.debug(
                        f"{node} is added to in_peri "
                        f"{_fixed.in_degree(node)} . {_fixed.out_degree(node)}"
                    )

            # go ahead
            node = next_node

            # if it is circular
            if node in path[:-1]:
                try:
                    loc = path.index(node)
                    derived_cycles.append(path[loc:])
                    path = path[: loc + 1]
                except ValueError:
                    pass

    if logger.isEnabledFor(DEBUG):
        logger.debug(f"size of g {g.number_of_edges()}")
        logger.debug(f"size of fixed {_fixed.number_of_edges()}")
        assert len(in_peri) == 0, f"In-peri remains. {in_peri}"
        assert len(out_peri) == 0, f"Out-peri remains. {out_peri}"
        logger.debug("re-check perimeters")

        in_peri_check, out_peri_check = _get_perimeters_nx(_fixed, g)

        assert len(in_peri_check) == 0, f"In-peri remains. {in_peri_check}"
        assert len(out_peri_check) == 0, f"Out-peri remains. {out_peri_check}"

        # Check if extended graph contains all original fixed edges
        for edge in fixed.edges():
            assert _fixed.has_edge(*edge)

    _remove_dummy_nodes_nx(_fixed)

    if logger.isEnabledFor(DEBUG):
        logger.debug(
            f"Number of fixed edges is {_fixed.number_of_edges()} / {g.number_of_edges()}"
        )
        logger.debug(f"Number of free cycles: {len(derived_cycles)}")
        ne = sum(len(cycle) - 1 for cycle in derived_cycles)
        logger.debug(f"Number of edges in free cycles: {ne}")

    return _fixed, derived_cycles


@accept_aliases(
    vertexPositions="vertex_positions",
    isPeriodicBoundary="is_periodic_boundary",
    dipoleOptimizationCycles="dipole_optimization_cycles",
    dipoleOptimizationCycles2="dipole_optimization_cycles2",
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
    dipole_optimization_cycles2: int = 0,
    connect_engine: Callable[
        [nx.DiGraph, nx.Graph], Tuple[Optional[nx.DiGraph], List[List[int]]]
    ] = connect_matching_paths,
    g_format: Optional[Literal["edges", "adjacency"]] = None,
    seed: Optional[int] = None,
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
        dipole_optimization_cycles2 (int, optional): Number of cycles for force_polarize (target_pol-directed flip).
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
            is_periodic_boundary
            or dipole_optimization_cycles != 0
            or np.any(target_pol != 0)
        ):
            logger.debug(
                "vertex_positions is None; is_periodic_boundary, "
                "dipole_optimization_cycles, and target_pol are ignored."
            )
        if fixed_edges.size() == 0 and pairing_attempts != 100:
            logger.debug("fixed_edges is empty; pairing_attempts is ignored.")

    # Save the global random state to decouple ice_graph execution from the global sequence
    global_random_state = np.random.get_state()
    try:
        if seed is not None:
            np.random.seed(seed)
        else:
            # Seed with OS entropy to be unpredictable while not advancing the global numpy generator linearly
            np.random.seed(None)
            
        # Convert input g to a NetworkX Graph G.
        if isinstance(g, nx.Graph):
            G = g.copy()
        else:
            n_orig_tmp, adj = _graph_to_adj(g, g_format)
            G = nx.Graph()
            G.add_nodes_from(range(n_orig_tmp))
            for u in range(n_orig_tmp):
                for v in adj[u]:
                    if u < v:
                        G.add_edge(u, v)
        n_orig = G.number_of_nodes()

        # derived cycles in extending the fixed edges.
        derived_cycles: List[List[int]] = []

        if fixed_edges.size() > 0:
            if logger.isEnabledFor(DEBUG):
                for u, v in fixed_edges.edges():
                    logger.debug(f"FIXED EDGE {u} {v}")

            processed_edges: Optional[nx.DiGraph] = None
            for attempt in range(pairing_attempts):
                # NetworkX-based engine: (fixed_edges, G) -> (processed_edges, derived_cycles)
                processed_edges, derived_cycles = connect_engine(fixed_edges, G)
                if processed_edges is not None:
                    break
                logger.info(
                    f"Attempt {attempt + 1}/{pairing_attempts} failed to connect paths"
                )
            else:
                logger.error(f"Failed to find a solution after {pairing_attempts} attempts")
                return None
        else:
            processed_edges = nx.DiGraph()

        # really fixed in connect_matching_paths_nx()
        finally_fixed_edges = nx.DiGraph(processed_edges)
        for cycle in derived_cycles:
            for u, v in zip(cycle, cycle[1:]):
                if finally_fixed_edges.has_edge(u, v):
                    finally_fixed_edges.remove_edge(u, v)

        # Divide the remaining (unfixed) part of the graph into a noodle graph
        divided_graph = noodlize(G, processed_edges)

        # Simplify paths ( paths with least crossings )
        paths = split_into_simple_paths(n_orig, divided_graph) + derived_cycles

        # arrange the orientations here if you want to balance the polarization
        if vertex_positions is not None:
            pos_arr = _vertex_positions_array(vertex_positions, n_orig)
            dim = pos_arr.shape[1]
            target_pol = np.resize(
                np.asarray(target_pol, dtype=float).flatten(), dim
            ).copy()

            # Set the target_pol in order to cancel the polarization in the fixed part.
            fixed_edge_list = list(finally_fixed_edges.edges())
            target_pol -= vector_sum(fixed_edge_list, pos_arr, is_periodic_boundary)

            paths = optimize(
                paths,
                vertex_positions=pos_arr,
                is_periodic_boundary=is_periodic_boundary,
                dipole_optimization_cycles=dipole_optimization_cycles,
                target_pol=target_pol,
            )

        # Combine everything together
        dg = nx.DiGraph(finally_fixed_edges)
        for path in paths:
            nx.add_path(dg, path)


        dg = force_polarize(dg, fixed_edges, vertex_positions, target_pol, dipole_optimization_cycles2)


        all_edges = list(dg.edges())
        _verify_ice_rules(n_orig, all_edges, fixed_edges)

        if return_edges:
            return all_edges
        return dg
    finally:
        np.random.set_state(global_random_state)
