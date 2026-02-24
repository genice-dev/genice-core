"""
Arrange edges appropriately. Internal representation uses plain arrays (no NetworkX).
"""

from collections import defaultdict, deque
from logging import getLogger, DEBUG
import numpy as np
from typing import List, Tuple, Optional, Set

from genice_core.graph_arrays import (
    node_to_idx,
    idx_to_node,
    connected_components,
)


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
    # copy adjacency: original n_orig nodes + space for n_orig new nodes
    adj_noodles = [list(adj[v]) for v in range(n_orig)] + [[] for _ in range(n_orig)]

    # remove fixed edges from copy (fixed edges are (u,v) for v in fixed_out[u])
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


def _decompose_complex_path(path: List[int]) -> List[List[int]]:
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
        """One BFS round: each node in out_peri tries to push one step. Returns False on failure."""
        if not out_peri:
            return True
        preferred = in_peri | undet_peri  # 一歩でシンクに着地する辺を優先
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
        """One BFS round: each node in in_peri tries to pull one step. Returns False on failure."""
        if not in_peri:
            return True
        preferred = out_peri | undet_peri  # 一歩でソースに着地する辺を優先
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

    max_rounds = (
        n_orig + 4
    ) * 4  # at most one new edge per node per round, finite edges
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


def _get_perimeters_bfs1(
    n_orig: int,
    adj: List[List[int]],
    out_adj: List[List[int]],
    in_adj: List[List[int]],
) -> Tuple[dict, dict, Set[int]]:
    """Perimeters with multiplicity (for BFS1). Returns (in_peri, out_peri, undet_peri)."""
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
    """Connect matching paths by "Shortest path to a set of targets" (SP2ST). Same interface as connect_matching_paths.
    Good for high doping. Returns ((out_adj, in_adj), derived_cycles) or (None, [])."""
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
