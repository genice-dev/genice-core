"""NetworkX-based topology operations (noodlize, path splitting, matching)."""

import logging
from logging import getLogger
from typing import List, Optional, Set, Tuple

import networkx as nx
import numpy as np


def _trace_path_nx(g: nx.Graph, path: List[int]) -> List[int]:
    """Trace the path in a linear or cyclic graph."""
    while True:
        last, head = path[-2:]
        for next_node in g[head]:
            if next_node != last:
                break
        else:
            return path
        path.append(next_node)
        if next_node == path[0]:
            return path


def _find_path_nx(g: nx.Graph) -> List[int]:
    """Find a path in a linear or cyclic graph."""
    nodes = list(g.nodes())
    if not nodes:
        return []
    head = nodes[0]
    neighbors = list(g[head])
    if len(neighbors) == 0:
        return []
    if len(neighbors) == 1:
        return _trace_path_nx(g, [head, neighbors[0]])
    c0 = _trace_path_nx(g, [head, neighbors[0]])
    if c0[-1] == head:
        return c0
    c1 = _trace_path_nx(g, [head, neighbors[1]])
    return c0[::-1] + c1[1:]


def _divide_nx(g: nx.Graph, vertex: int, offset: int) -> None:
    """Divide a vertex into two vertices and redistribute edges."""
    nei = (list(g[vertex]) + [None, None, None, None])[:4]
    migrants = set(np.random.choice(nei, 2, replace=False)) - {None}
    new_vertex = vertex + offset
    for migrant in migrants:
        g.remove_edge(migrant, vertex)
        g.add_edge(new_vertex, migrant)


def noodlize_nx(g: nx.Graph, fixed: Optional[nx.DiGraph] = None) -> nx.Graph:
    """Divide each vertex and make a set of paths (NetworkX version)."""
    logger = getLogger()
    if fixed is None:
        fixed = nx.DiGraph()
    g_fix = nx.Graph(fixed)  # undirected copy
    offset = len(g)
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
            size = len(order) - order[node]
            cycle = store[-size:] + [node]
            out.append(cycle)
            for v in cycle[1:]:
                del order[v]
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
        g_noodle = g_noodles.subgraph(vertice_set)
        path = _find_path_nx(g_noodle)
        flatten = [v % nnode for v in path]
        paths.extend(_decompose_complex_path_nx(flatten))
    return paths


def _remove_dummy_nodes_nx(g: nx.DiGraph) -> None:
    """Remove dummy nodes -1..-4 from the graph (in place)."""
    for i in range(-1, -5, -1):
        if g.has_node(i):
            g.remove_node(i)


def _choose_free_edge_nx(g: nx.Graph, dg: nx.DiGraph, node: int) -> Optional[int]:
    """Find an unfixed edge of the node in NetworkX representation."""
    neis = (list(g[node]) + [-1, -2, -3, -4])[:4]
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
        if fixed.in_degree(node) + fixed.out_degree(node) < g.degree(node):
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
                logger.debug(f"Dead end at {node}. Path is {path}.")
                break
            if node in in_peri:
                logger.debug(f"Reach at a perimeter node {node}. Path is {path}.")
                in_peri.remove(node)
                break
            if node in out_peri:
                logger.debug(f"node {node} is on the out_peri...")
            if max(_fixed.in_degree(node), _fixed.out_degree(node)) * 2 > 4:
                logger.info("Failed to balance. Starting over ...")
                return None, []
            if g.degree(node) == _fixed.degree(node):
                logger.info(f"node {node} has no free edge. Starting over ...")
                return None, []
            next_node = _choose_free_edge_nx(g, _fixed, node)
            if next_node is None:
                logger.info(
                    f"node {node} has no free edge (unexpected). Starting over ..."
                )
                return None, []
            _fixed.add_edge(node, next_node)
            if next_node >= 0:
                path.append(next_node)
                if _fixed.in_degree(node) > _fixed.out_degree(node):
                    out_peri.add(node)
            node = next_node
            if node in path[:-1]:
                try:
                    loc = path.index(node)
                    derived_cycles.append(path[loc:])
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
                logger.debug(f"Dead end at {node}. Path is {path} {in_peri}.")
                break
            if node in out_peri:
                logger.debug(f"Reach at a perimeter node {node}. Path is {path}.")
                out_peri.remove(node)
                break
            if node in in_peri:
                logger.debug(f"node {node} is on the in_peri...")
            if max(_fixed.in_degree(node), _fixed.out_degree(node)) * 2 > 4:
                logger.info("Failed to balance. Starting over ...")
                return None, []
            if g.degree(node) == _fixed.degree(node):
                logger.info(f"node {node} has no free edge. Starting over ...")
                return None, []
            next_node = _choose_free_edge_nx(g, _fixed, node)
            if next_node is None:
                logger.info(
                    f"node {node} has no free edge (unexpected). Starting over ..."
                )
                return None, []
            if next_node >= 0:
                path.append(next_node)
            _fixed.add_edge(next_node, node)
            if next_node >= 0:
                if _fixed.in_degree(node) < _fixed.out_degree(node):
                    in_peri.add(node)
                    logger.debug(
                        f"{node} is added to in_peri "
                        f"{_fixed.in_degree(node)} . {_fixed.out_degree(node)}"
                    )
            node = next_node
            if node in path[:-1]:
                try:
                    loc = path.index(node)
                    derived_cycles.append(path[loc:])
                    path = path[: loc + 1]
                except ValueError:
                    pass

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"size of g {g.number_of_edges()}")
        logger.debug(f"size of fixed {_fixed.number_of_edges()}")
        assert len(in_peri) == 0, f"In-peri remains. {in_peri}"
        assert len(out_peri) == 0, f"Out-peri remains. {out_peri}"
        logger.debug("re-check perimeters")
        in_peri_check, out_peri_check = _get_perimeters_nx(_fixed, g)
        assert len(in_peri_check) == 0, f"In-peri remains. {in_peri_check}"
        assert len(out_peri_check) == 0, f"Out-peri remains. {out_peri_check}"

    _remove_dummy_nodes_nx(_fixed)

    if logger.isEnabledFor(DEBUG):
        logger.debug(
            f"Number of fixed edges is {_fixed.number_of_edges()} / {g.number_of_edges()}"
        )
        logger.debug(f"Number of free cycles: {len(derived_cycles)}")
        ne = sum(len(cycle) - 1 for cycle in derived_cycles)
        logger.debug(f"Number of edges in free cycles: {ne}")

    return _fixed, derived_cycles

