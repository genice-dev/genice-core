"""Connect matching paths by random path extension (one path at a time)."""

from logging import getLogger, DEBUG
from typing import List, Tuple, Optional

import numpy as np

from genice_core.topology._shared import (
    node_to_idx,
    _copy_directed,
    _get_perimeters,
    _choose_free_edge,
    _remove_dummy_nodes,
)


def connect_matching_paths(
    n_orig: int,
    adj: List[List[int]],
    fixed_out: List[List[int]],
    fixed_in: List[List[int]],
) -> Tuple[Optional[Tuple[List[List[int]], List[List[int]]]], List[List[int]]]:
    """Connect matching paths. Returns ((out_adj, in_adj), derived_cycles) or (None, [])."""
    logger = getLogger()
    _fixed_out, _fixed_in = _copy_directed(n_orig, fixed_out, fixed_in)
    in_peri, out_peri = _get_perimeters(n_orig, adj, _fixed_out, _fixed_in)
    logger.debug(f"out_peri {out_peri}")
    logger.debug(f"in_peri {in_peri}")
    derived_cycles: List[List[int]] = []

    def add_edge(u: int, v: int) -> None:
        iu, iv = node_to_idx(u, n_orig), node_to_idx(v, n_orig)
        _fixed_out[iu].append(v)
        _fixed_in[iv].append(u)

    def in_degree(node: int) -> int:
        return len(_fixed_in[node_to_idx(node, n_orig)])

    def out_degree(node: int) -> int:
        return len(_fixed_out[node_to_idx(node, n_orig)])

    # Process out_peri
    while out_peri:
        node = int(np.random.choice(list(out_peri)))
        out_peri.discard(node)
        path = [node]
        while True:
            if node < 0:
                logger.debug(f"Dead end at {node}. Path is {path}.")
                break
            if node in in_peri:
                in_peri.discard(node)
                break
            if out_degree(node) * 2 > 4:
                logger.info("Failed to balance. Starting over ...")
                return None, []
            if len(adj[node]) == out_degree(node) + in_degree(node):
                logger.info(f"node {node} has no free edge. Starting over ...")
                return None, []
            next_node = _choose_free_edge(n_orig, adj, _fixed_out, _fixed_in, node)
            if next_node is None:
                logger.info(
                    f"node {node} has no free edge (unexpected). Starting over ..."
                )
                return None, []
            add_edge(node, next_node)
            if next_node >= 0:
                path.append(next_node)
                if in_degree(node) > out_degree(node):
                    out_peri.add(node)
            node = next_node
            if node in path[:-1]:
                loc = path.index(node)
                derived_cycles.append(path[loc:])
                path = path[: loc + 1]

    # Process in_peri
    while in_peri:
        node = int(np.random.choice(list(in_peri)))
        in_peri.discard(node)
        logger.debug(f"first node {node}")
        path = [node]
        while True:
            if node < 0:
                break
            if node in out_peri:
                out_peri.discard(node)
                break
            if out_degree(node) * 2 > 4:
                logger.info("Failed to balance. Starting over ...")
                return None, []
            if len(adj[node]) == out_degree(node) + in_degree(node):
                logger.info(f"node {node} has no free edge. Starting over ...")
                return None, []
            next_node = _choose_free_edge(n_orig, adj, _fixed_out, _fixed_in, node)
            if next_node is None:
                logger.info(
                    f"node {node} has no free edge (unexpected). Starting over ..."
                )
                return None, []
            if next_node >= 0:
                path.append(next_node)
            add_edge(next_node, node)
            if next_node >= 0 and in_degree(node) < out_degree(node):
                in_peri.add(node)
            node = next_node
            if node in path[:-1]:
                loc = path.index(node)
                derived_cycles.append(path[loc:])
                path = path[: loc + 1]

    if logger.isEnabledFor(DEBUG):
        assert len(in_peri) == 0, f"In-peri remains. {in_peri}"
        assert len(out_peri) == 0, f"Out-peri remains. {out_peri}"
        in_peri_check, out_peri_check = _get_perimeters(
            n_orig, adj, _fixed_out, _fixed_in
        )
        assert len(in_peri_check) == 0
        assert len(out_peri_check) == 0

    _remove_dummy_nodes(n_orig, _fixed_out, _fixed_in)
    return (_fixed_out, _fixed_in), derived_cycles
