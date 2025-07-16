"""
Arrange edges appropriately.
"""

import numpy as np
import networkx as nx
from logging import getLogger, DEBUG
from typing import Union, List, Tuple, Optional, Set
from collections import defaultdict, deque


def _trace_path(g: nx.Graph, path: List[int]) -> List[int]:
    """Trace the path in a linear or cyclic graph.

    Args:
        g (nx.Graph): A linear or a simple cyclic graph.
        path (List[int]): A given path to be extended.

    Returns:
        List[int]: The extended path or cycle.
    """
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


def _find_path(g: nx.Graph) -> List[int]:
    """Find a path in a linear or cyclic graph.

    Args:
        g (nx.Graph): A linear or a simple cyclic graph.

    Returns:
        List[int]: The path or cycle.
    """
    nodes = list(g.nodes())
    # choose one node
    head = nodes[0]
    # look neighbors
    neighbors = list(g[head])
    if len(neighbors) == 0:
        # isolated node
        return []
    elif len(neighbors) == 1:
        # head is an end node, fortunately.
        return _trace_path(g, [head, neighbors[0]])
    # look forward
    c0 = _trace_path(g, [head, neighbors[0]])

    if c0[-1] == head:
        # cyclic graph
        return c0

    # look backward
    c1 = _trace_path(g, [head, neighbors[1]])
    return c0[::-1] + c1[1:]


def _divide(g: nx.Graph, vertex: int, offset: int) -> None:
    """Divide a vertex into two vertices and redistribute edges.

    Args:
        g (nx.Graph): The graph to modify.
        vertex (int): The vertex to divide.
        offset (int): The offset for the new vertex label.
    """
    # fill by Nones if number of neighbors is less than 4
    nei = (list(g[vertex]) + [None, None, None, None])[:4]

    # two neighbor nodes that are passed away to the new node
    migrants = set(np.random.choice(nei, 2, replace=False)) - {None}

    # new node label
    newVertex = vertex + offset

    # assemble edges
    for migrant in migrants:
        g.remove_edge(migrant, vertex)
        g.add_edge(newVertex, migrant)


def noodlize(g: nx.Graph, fixed: nx.DiGraph = nx.DiGraph()) -> nx.Graph:
    """Divide each vertex of the graph and make a set of paths.

    A new algorithm suggested by Prof. Sakuma, Yamagata University.

    Args:
        g (nx.Graph): An ice-like undirected graph. All vertices must not be >4-degree.
        fixed (nx.DiGraph, optional): Specifies the edges whose direction is fixed. Defaults to an empty graph.

    Returns:
        nx.Graph: A graph made of chains and cycles.
    """
    logger = getLogger()

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
            _divide(g_noodles, v, offset)

    return g_noodles


def _decompose_complex_path(path: List[int]) -> List[List[int]]:
    """Divide a complex path with self-crossings into simple cycles and paths.

    Args:
        path (List[int]): A complex path.

    Yields:
        List[int]: A short and simple path/cycle.
    """
    logger = getLogger()
    if len(path) == 0:
        return
    logger.debug(f"decomposing {path}...")
    order = dict()
    order[path[0]] = 0
    store = [path[0]]
    headp = 1
    while headp < len(path):
        node = path[headp]

        if node in order:
            # it is a cycle!
            size = len(order) - order[node]
            cycle = store[-size:] + [node]
            yield cycle

            # remove them from the order[]
            for v in cycle[1:]:
                del order[v]

            # truncate the store
            store = store[:-size]

        order[node] = len(order)
        store.append(node)
        headp += 1
    if len(store) > 1:
        yield store
    logger.debug(f"Done decomposition.")


def split_into_simple_paths(
    nnode: int,
    g_noodles: nx.Graph,
) -> List[List[int]]:
    """Set the orientations to the components.

    Args:
        nnode (int): Number of nodes in the original graph.
        g_noodles (nx.Graph): The divided graph.

    Yields:
        List[int]: A short and simple path/cycle.
    """
    for verticeSet in nx.connected_components(g_noodles):
        # a component of c is either a chain or a cycle.
        g_noodle = g_noodles.subgraph(verticeSet)

        # Find a simple path in the doubled graph
        # It must be a simple path or a simple cycle.
        path = _find_path(g_noodle)

        # Flatten then path. It may make the path self-crossing.
        flatten = [v % nnode for v in path]

        # Divide a long path into simple paths and cycles.
        yield from _decompose_complex_path(flatten)


def _remove_dummy_nodes(g: Union[nx.Graph, nx.DiGraph]) -> None:
    """Remove dummy nodes from the graph.

    Args:
        g (Union[nx.Graph, nx.DiGraph]): The graph to clean.
    """
    for i in range(-1, -5, -1):
        if g.has_node(i):
            g.remove_node(i)


def _find_perimeters(fixed: nx.DiGraph, g: nx.Graph):
    """Find the perimeters of the graph.

    Args:
        fixed (nx.DiGraph): Fixed edges.
        g (nx.Graph): Skeletal graph.
    """
    logger = getLogger()

    logger.debug(f"fixed {fixed.number_of_edges()}")
    logger.debug(f"g {g.number_of_edges()}")

    in_peri = defaultdict(int)
    out_peri = defaultdict(int)
    undet_peri = set()

    for node in fixed:
        # If the node has unfixed edges,
        if fixed.in_degree[node] + fixed.out_degree[node] < g.degree[node]:
            # if it is not balanced,
            diff = fixed.in_degree[node] - fixed.out_degree[node]
            if diff > 0:
                out_peri[node] += diff
            elif diff < 0:
                in_peri[node] += -diff

    u = set(g.nodes()) - set(fixed.nodes())
    for node in u:
        if g.degree[node] in (1, 3):
            undet_peri.add(node)

    logger.debug(f"{out_peri=}")
    logger.debug(f"{in_peri=}")
    logger.debug(f"{undet_peri=}")

    return in_peri, out_peri, undet_peri


def find_shortest_path_to_any_target_bfs(
    graph: nx.DiGraph, start_node, target_nodes: set
):
    """
    始点から、終点の候補のうちいずれか一つへの重み1の最短経路を見つける（BFSベース）。

    Args:
        graph (nx.DiGraph): networkxの有向グラフオブジェクト。
        start_node: 始点となるノード。
        target_nodes (set): 終点の候補となるノードのセット。

    Returns:
        tuple: (shortest_path_list, end_of_shortest_path)
               - shortest_path_list (list): 始点から見つかったターゲットへの最短経路のノードリスト。
                                            到達できない場合は空リスト。
               - end_of_shortest_path: 最短経路の終点となったターゲットノード。
                                       到達できない場合は None。
    """

    # 初期化
    # BFSでは距離ではなく、経路自体を記録する辞書を使うことが多い
    # {ノード: そのノードに到達する直前のノード}
    predecessors = {node: None for node in graph.nodes()}

    # 探索キュー (dequeは高速な両端キュー)
    # BFSなので、(ノード) を格納。距離は不要（重み1なので距離はパスの長さに等しい）
    queue = deque([start_node])

    # 訪問済みノード (重複探索を防ぐ)
    visited = {start_node}

    found_target_node = None

    while queue:
        current_node = queue.popleft()  # キューの先頭からノードを取り出す (BFSの特性)

        # **** ここが重要な変更点 ****
        # 現在のノードが終点の候補のいずれかであれば、そこで探索を終了
        if current_node in target_nodes:
            found_target_node = current_node
            break  # ターゲットが見つかったのでループを抜ける

        # 隣接ノードの探索
        # networkxでは graph.neighbors(node) で隣接ノードをイテレートできる
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                predecessors[neighbor] = current_node  # 経路を記録
                # print(f"{current_node} -> {neighbor}")
                queue.append(neighbor)  # キューの末尾に追加

    # 最短経路の再構築
    path = []
    if found_target_node is not None:
        current = found_target_node
        while current is not None:
            path.append(current)
            current = predecessors[current]
        path.reverse()  # 逆順になっているので反転

    return path, found_target_node


def connect_matching_paths_BFS(
    fixed: nx.DiGraph, g: nx.Graph
) -> Tuple[Optional[nx.DiGraph], List[List[int]]]:

    def _choose_free_edge(g: nx.Graph, dg: nx.DiGraph, node: int) -> Optional[int]:
        """Find an unfixed edge of the node.

        Args:
            g (nx.Graph): The original graph.
            dg (nx.DiGraph): The directed graph.
            node (int): The node to find edges for.

        Returns:
            Optional[int]: A free edge if found, None otherwise.
        """
        # add dummy nodes to make number of edges be four.
        neis = (list(g[node]) + [-1, -2, -3, -4])[:4]
        # and select one randomly
        np.random.shuffle(neis)
        for nei in neis:
            if not (dg.has_edge(node, nei) or dg.has_edge(nei, node)):
                return nei
        return None

    # 個々のノードについて、幅優先探索を行い、最短matchingを見つける。
    # in_periとout_periは同数とは限らない。本物のperiに到達した場合にも終了する。
    # 1つパスを見付けたらそれを除外し、残りで同じ作業をくりかえす。

    logger = getLogger()

    # Make a copy to keep the original graph untouched
    _fixed = nx.DiGraph(fixed)

    in_peri, out_peri, undet_peri = _find_perimeters(_fixed, g)
    # 使える辺だけ残す。
    _undirected = nx.Graph()
    for i, j in g.edges():
        if not (_fixed.has_edge(i, j) or _fixed.has_edge(j, i)):
            _undirected.add_edge(i, j)
    logger.debug(f"size of _g {_undirected.number_of_edges()}")

    # out_periの1つから出発し、in_periまたはundet_periに到達する最短経路をさがす。

    # 問題点
    # - 二重out_peri2+0の可能性があるので、選んだnodeを即out_periから消してはいけない。
    # - 経路が二重out_peri2+0をかすめると3+1になり破綻する。破綻を検知して中断させるか、もしくはかすめないようにするか。
    while len(out_peri) > 0:
        node = np.random.choice(list(out_peri))
        out_peri[node] -= 1
        if out_peri[node] == 0:
            del out_peri[node]

        targets = set(in_peri) | undet_peri
        path, end = find_shortest_path_to_any_target_bfs(_undirected, node, targets)

        # 経路の途中でin_periやout_periやundet_periに到達した場合も、それらがperiでなくなることはない。
        # print(path, list(_fixed.predecessors(704)), list(_fixed.successors(704)))
        if len(path) == 0:
            logger.debug(
                f"No relevant path found from {node} to {targets}"
                f" {list(_fixed.predecessors(node))} {list(_fixed.successors(node))} {_undirected[node]}"
            )
            with open("failed.txt", "w") as f:
                print(node, file=f)
                print(*list(targets), file=f)
                for edge in _undirected.edges():
                    f.write(f"{edge[0]} {edge[1]}\n")

            # assert False
            return None, None

        for i, j in zip(path, path[1:]):
            _fixed.add_edge(i, j)
            _undirected.remove_edge(i, j)
        # もし、経路を追加することでendがin_periでなくなるなら、抹消する。
        if end in in_peri:
            in_peri[end] -= 1
            if in_peri[end] == 0:
                del in_peri[end]
        else:
            undet_peri -= {end}

    logger.debug(f"{out_peri=}")
    logger.debug(f"{in_peri=}")
    logger.debug(f"{undet_peri=}")

    in_peri_result, undet_peri_result = in_peri, undet_peri
    in_peri, out_peri, undet_peri = _find_perimeters(_fixed, g)

    while len(in_peri) > 0:
        # print(len(list(_g.edges())) + len(list(_fixed.edges())))
        node = np.random.choice(list(in_peri))
        in_peri[node] -= 1
        if in_peri[node] == 0:
            del in_peri[node]
        targets = set(out_peri) | undet_peri
        path, end = find_shortest_path_to_any_target_bfs(_undirected, node, targets)

        # 経路の途中でin_periやout_periやundet_periに到達した場合も、それらがperiでなくなることはない。
        # print(path)
        if len(path) == 0:
            logger.debug(f"No relevant path found from {node} to {targets}")
            return None, None

        for i, j in zip(path, path[1:]):
            _fixed.add_edge(j, i)
            _undirected.remove_edge(j, i)
        if end in out_peri:
            out_peri[end] -= 1
            if out_peri[end] == 0:
                del out_peri[end]
        else:
            undet_peri -= {end}

    # logger.debug(f"{out_peri=}")
    # logger.debug(f"{in_peri=}")
    # logger.debug(f"{undet_peri=}")

    in_peri, out_peri, undet_peri = _find_perimeters(_fixed, g)

    if logger.isEnabledFor(DEBUG):
        _g_nodes = set(_undirected.nodes())
        _fixed_nodes = set(_fixed.nodes())

        g_nodes = set(g.nodes())
        fixed_nodes = set(fixed.nodes())

        assert _g_nodes | _fixed_nodes == g_nodes | fixed_nodes

        # すべてのノードの、無向辺の数が偶数であることを確認する。
        for node in g_nodes:
            if node in _fixed_nodes:
                in_deg = _fixed.in_degree[node]
                out_deg = _fixed.out_degree[node]
            else:
                in_deg = 0
                out_deg = 0
            if node in _g_nodes:
                deg = _undirected.degree[node]
            else:
                deg = 0

            # 固定辺と無向辺の合計は、もとのグラフの辺の数と等しい。
            assert in_deg + out_deg + deg == g.degree[node]

            # もし、固定辺の先端が中和されずに残っているなら、それは、表面である。
            if in_deg != out_deg:
                assert deg % 2 == 0, (node, in_deg, out_deg, deg)
            # 固定辺の先端が中和されているなら、無向辺の個数はいくつでもよい。

    derivedCycles = []

    return _fixed, derivedCycles


if __name__ == "__main__":
    g = nx.Graph()
    with open("failed.txt", "r") as f:
        source = int(f.readline())
        targets = list(map(int, f.readline().split()))
        for line in f:
            i, j = map(int, line.split())
            g.add_edge(i, j)
    print(source, targets)
    print(find_shortest_path_to_any_target_bfs(g, source, targets))
