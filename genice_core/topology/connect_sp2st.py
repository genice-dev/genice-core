"""Connect matching paths by shortest path to a set of targets (SP2ST). Good for high doping."""

from collections import defaultdict, deque
from logging import getLogger, DEBUG
from typing import List, Tuple, Optional, Set

import numpy as np

from genice_core.topology._shared import (
    node_to_idx,
    idx_to_node,
    _copy_directed,
    _get_perimeters,
    _remove_dummy_nodes,
)


def _get_perimeters_bfs1(
    n_orig: int,
    adj: List[List[int]],
    out_adj: List[List[int]],
    in_adj: List[List[int]],
) -> Tuple[dict, dict, Set[int]]:
    """Perimeters with multiplicity (for SP2ST). Returns (in_peri, out_peri, undet_peri)."""
    in_peri: dict = defaultdict(int)
    out_peri: dict = defaultdict(int)
    undet_peri: Set[int] = set()
    size = n_orig + 4
    for i in range(size):
        node = idx_to_node(i, n_orig)
        deg_g = len(adj[node]) if 0 <= node < n_orig else 0
        in_d = len(in_adj[i])
        out_d = len(out_adj[i])
        if node < 0 or in_d + out_d >= deg_g:
            continue
        diff = in_d - out_d
        if diff > 0:
            out_peri[node] += diff
        elif diff < 0:
            in_peri[node] += -diff
    for v in range(n_orig):
        i = node_to_idx(v, n_orig)
        if len(adj[v]) in {1, 3} and len(in_adj[i]) == 0 and len(out_adj[i]) == 0:
            undet_peri.add(v)
    return in_peri, out_peri, undet_peri


def _build_free_adj(
    n_orig: int,
    adj: List[List[int]],
    out_adj: List[List[int]],
) -> List[List[int]]:
    """Build mutable adjacency of free edges only (real nodes 0..n_orig-1)."""
    free_adj: List[List[int]] = [[] for _ in range(n_orig)]
    for u in range(n_orig):
        iu = node_to_idx(u, n_orig)
        for v in adj[u]:
            if (
                0 <= v < n_orig
                and v not in out_adj[iu]
                and u not in out_adj[node_to_idx(v, n_orig)]
            ):
                free_adj[u].append(v)
    return free_adj


def _shortest_path_to_any_target_bfs(
    free_adj: List[List[int]],
    start: int,
    targets: Set[int],
) -> Tuple[List[int], Optional[int]]:
    """BFS from start to any node in targets. Returns (path, end) or ([], None)."""
    if start in targets:
        return [start], start
    pred: dict = {start: None}
    queue: deque = deque([start])
    visited: Set[int] = {start}
    found: Optional[int] = None
    while queue:
        cur = queue.popleft()
        for w in free_adj[cur]:
            if w not in visited:
                visited.add(w)
                pred[w] = cur
                queue.append(w)
                if w in targets:
                    found = w
                    break
        if found is not None:
            break
    if found is None:
        return [], None
    path: List[int] = []
    cur = found
    while cur is not None:
        path.append(cur)
        cur = pred.get(cur)
    path.reverse()
    return path, found


def connect_matching_paths_SP2ST(
    n_orig: int,
    adj: List[List[int]],
    fixed_out: List[List[int]],
    fixed_in: List[List[int]],
) -> Tuple[Optional[Tuple[List[List[int]], List[List[int]]]], List[List[int]]]:
    """Connect matching paths by shortest path to a set of targets (SP2ST).
    Same interface as connect_matching_paths. Good for high doping.
    Returns ((out_adj, in_adj), derived_cycles) or (None, [])."""
    logger = getLogger()
    _fixed_out, _fixed_in = _copy_directed(n_orig, fixed_out, fixed_in)
    in_peri, out_peri, undet_peri = _get_perimeters_bfs1(
        n_orig, adj, _fixed_out, _fixed_in
    )
    free_adj = _build_free_adj(n_orig, adj, _fixed_out)
    derived_cycles: List[List[int]] = []

    def add_edge(u: int, v: int) -> None:
        iu, iv = node_to_idx(u, n_orig), node_to_idx(v, n_orig)
        _fixed_out[iu].append(v)
        _fixed_in[iv].append(u)
        if 0 <= u < n_orig and 0 <= v < n_orig:
            if v in free_adj[u]:
                free_adj[u].remove(v)
            if u in free_adj[v]:
                free_adj[v].remove(u)

    # out_peri から 1 本ずつ最短経路で in_peri または undet_peri へ
    while out_peri:
        node = int(np.random.choice(list(out_peri)))
        out_peri[node] -= 1
        if out_peri[node] == 0:
            del out_peri[node]
        targets = set(in_peri) | undet_peri
        path, end = _shortest_path_to_any_target_bfs(free_adj, node, targets)
        if not path:
            logger.debug(f"No path from {node} to {targets}")
            return None, []
        for i, j in zip(path, path[1:]):
            add_edge(i, j)
        if end in in_peri:
            in_peri[end] -= 1
            if in_peri[end] == 0:
                del in_peri[end]
        else:
            undet_peri.discard(end)

    in_peri, out_peri, undet_peri = _get_perimeters_bfs1(
        n_orig, adj, _fixed_out, _fixed_in
    )

    # in_peri から 1 本ずつ最短経路で out_peri または undet_peri へ（逆向きに追加）
    while in_peri:
        node = int(np.random.choice(list(in_peri)))
        in_peri[node] -= 1
        if in_peri[node] == 0:
            del in_peri[node]
        targets = set(out_peri) | undet_peri
        path, end = _shortest_path_to_any_target_bfs(free_adj, node, targets)
        if not path:
            logger.debug(f"No path from {node} to {targets}")
            return None, []
        for i, j in zip(path, path[1:]):
            add_edge(j, i)
        if end in out_peri:
            out_peri[end] -= 1
            if out_peri[end] == 0:
                del out_peri[end]
        else:
            undet_peri.discard(end)

    if logger.isEnabledFor(DEBUG):
        in_chk, out_chk = _get_perimeters(n_orig, adj, _fixed_out, _fixed_in)
        assert len(in_chk) == 0 and len(out_chk) == 0

    _remove_dummy_nodes(n_orig, _fixed_out, _fixed_in)
    return (_fixed_out, _fixed_in), derived_cycles
