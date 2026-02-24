"""Connect matching paths by BFS rounds (all sources/sinks advance one step per round)."""

from logging import getLogger, DEBUG
from typing import List, Tuple, Optional, Set

import numpy as np

from genice_core.topology._shared import (
    node_to_idx,
    idx_to_node,
    _copy_directed,
    _get_perimeters,
    _choose_free_edge,
    _remove_dummy_nodes,
)


def connect_matching_paths_bfs(
    n_orig: int,
    adj: List[List[int]],
    fixed_out: List[List[int]],
    fixed_in: List[List[int]],
) -> Tuple[Optional[Tuple[List[List[int]], List[List[int]]]], List[List[int]]]:
    """Connect matching paths by advancing from all sources/sinks in BFS rounds.
    Reduces dead ends by not letting a single path monopolize edges.
    Returns ((out_adj, in_adj), derived_cycles) or (None, []). derived_cycles may be [].
    """
    logger = getLogger()
    _fixed_out, _fixed_in = _copy_directed(n_orig, fixed_out, fixed_in)
    in_peri: Set[int] = set()
    out_peri: Set[int] = set()
    size = n_orig + 4
    for i in range(size):
        node = idx_to_node(i, n_orig)
        deg_g = len(adj[node]) if 0 <= node < n_orig else 0
        in_d = len(_fixed_in[i])
        out_d = len(_fixed_out[i])
        if node < 0 or in_d + out_d >= deg_g:
            continue
        if in_d > out_d:
            out_peri.add(node)
        elif in_d < out_d:
            in_peri.add(node)
    derived_cycles: List[List[int]] = []

    # 次数 1 or 3 でまだ固定辺が 0 のノードを未決定端点とする（シンク候補）
    undet_peri: Set[int] = set()
    for v in range(n_orig):
        if (
            len(adj[v]) in {1, 3}
            and len(_fixed_in[node_to_idx(v, n_orig)]) == 0
            and len(_fixed_out[node_to_idx(v, n_orig)]) == 0
        ):
            undet_peri.add(v)

    def add_edge(u: int, v: int) -> None:
        iu, iv = node_to_idx(u, n_orig), node_to_idx(v, n_orig)
        _fixed_out[iu].append(v)
        _fixed_in[iv].append(u)

    def in_degree(node: int) -> int:
        return len(_fixed_in[node_to_idx(node, n_orig)])

    def out_degree(node: int) -> int:
        return len(_fixed_out[node_to_idx(node, n_orig)])

    def push_round() -> bool:
        if not out_peri:
            return True
        preferred = in_peri | undet_peri
        nodes_list = list(out_peri)
        np.random.shuffle(nodes_list)
        next_out = set()
        for node in nodes_list:
            if node not in out_peri:
                continue
            if node < 0:
                continue
            if out_degree(node) * 2 > 4:
                logger.info("Failed to balance (push). Starting over ...")
                return False
            if len(adj[node]) == out_degree(node) + in_degree(node):
                logger.info(f"node {node} has no free edge (push). Starting over ...")
                return False
            next_node = _choose_free_edge(
                n_orig, adj, _fixed_out, _fixed_in, node, preferred=preferred
            )
            if next_node is None:
                logger.info(
                    f"node {node} has no free edge (push, unexpected). Starting over ..."
                )
                return False
            add_edge(node, next_node)
            out_peri.discard(node)
            if in_degree(node) > out_degree(node):
                next_out.add(node)
            if next_node >= 0:
                if next_node in in_peri:
                    in_peri.discard(next_node)
                if next_node in undet_peri:
                    undet_peri.discard(next_node)
                if in_degree(next_node) > out_degree(next_node):
                    next_out.add(next_node)
        out_peri.clear()
        out_peri.update(next_out)
        return True

    def pull_round() -> bool:
        if not in_peri:
            return True
        preferred = out_peri | undet_peri
        nodes_list = list(in_peri)
        np.random.shuffle(nodes_list)
        next_in = set()
        for node in nodes_list:
            if node not in in_peri:
                continue
            if node < 0:
                continue
            if out_degree(node) * 2 > 4:
                logger.info("Failed to balance (pull). Starting over ...")
                return False
            if len(adj[node]) == out_degree(node) + in_degree(node):
                logger.info(f"node {node} has no free edge (pull). Starting over ...")
                return False
            next_node = _choose_free_edge(
                n_orig, adj, _fixed_out, _fixed_in, node, preferred=preferred
            )
            if next_node is None:
                logger.info(
                    f"node {node} has no free edge (pull, unexpected). Starting over ..."
                )
                return False
            add_edge(next_node, node)
            in_peri.discard(node)
            if in_degree(node) < out_degree(node):
                next_in.add(node)
            if next_node >= 0:
                if next_node in out_peri:
                    out_peri.discard(next_node)
                if next_node in undet_peri:
                    undet_peri.discard(next_node)
                if in_degree(next_node) < out_degree(next_node):
                    next_in.add(next_node)
        in_peri.clear()
        in_peri.update(next_in)
        return True

    max_rounds = (n_orig + 4) * 4
    for _ in range(max_rounds):
        if not out_peri and not in_peri:
            break
        if not push_round():
            return None, []
        if not pull_round():
            return None, []
    if out_peri or in_peri:
        logger.info("BFS rounds did not converge. Starting over ...")
        return None, []

    if logger.isEnabledFor(DEBUG):
        in_peri_check, out_peri_check = _get_perimeters(
            n_orig, adj, _fixed_out, _fixed_in
        )
        assert len(in_peri_check) == 0
        assert len(out_peri_check) == 0

    _remove_dummy_nodes(n_orig, _fixed_out, _fixed_in)
    return (_fixed_out, _fixed_in), derived_cycles
