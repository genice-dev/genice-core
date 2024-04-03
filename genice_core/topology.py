"""
Arrange edges appropriately.
"""

from logging import getLogger, DEBUG
import networkx as nx
import numpy as np
from typing import Union


def _trace_path(g: nx.Graph, path: list) -> list:
    """trace the path

    Args:
        g (nx.Graph): a linear or a simple cyclic graph.
        path (list): A given path tho be extended

    Returns:
        list: the extended path or cycle
    """
    while True:
        # look at the head of the path
        last, head = path[-2:]
        for next in g[head]:
            if next != last:
                # go ahead
                break
        else:
            # no next node
            return path
        path.append(next)
        if next == path[0]:
            # is cyclic
            return path


def _find_path(g: nx.Graph) -> list:
    """Find the path in g

    Args:
        g (nx.Graph): a linear or a simple cyclic graph.

    Returns:
        list: the path or cycle
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


def _divide(g: nx.Graph, vertex: int, offset: int):
    # fill by Nones if number of neighbors is less than 4
    nei = (list(g[vertex]) + [None, None, None, None])[:4]

    # two neighbor nodes that are passed away to the new node
    migrants = set(np.random.choice(nei, 2, replace=False)) - set([None])

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
        fixed (Union[nx.DiGraph,None], optional): Specifies the edges whose direction is fixed.. Defaults to None.

    Returns:
        nx.Graph: A graph mode of chains and cycles.
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

    # divg is made of chains and cycles.
    # divg does not contain the edges in fixed.
    return g_noodles


def _decompose_complex_path(path: list):
    """A generator that divides a complex path with self-crossings to set of simple cycles and paths.

    Args:
        path (list): A complex path

    Yields:
        list: a short and simple path/cycle
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
):
    """Set the orientations to the components.

    Args:
        nnode (int): number of nodes in the original graph.
        divg (nx.Graph): the divided graph.

    Yields:
        list: a short and simple path/cycle
    """

    for verticeSet in nx.connected_components(g_noodles):
        # a component of c is either a chain or a cycle.
        g_noodle = g_noodles.subgraph(verticeSet)
        # nn = len(g_noodle)
        # ne = len([e for e in g_noodle.edges()])
        # assert nn == ne or nn == ne + 1

        # Find a simple path in the doubled graph
        # It must be a simple path or a simple cycle.
        path = _find_path(g_noodle)

        # Flatten then path. It may make the path self-crossing.
        flatten = [v % nnode for v in path]

        # Divide a long path into simple paths and cycles.
        yield from _decompose_complex_path(flatten)


def _remove_dummy_nodes(g: Union[nx.Graph, nx.DiGraph]):
    for i in range(-1, -5, -1):
        if g.has_node(i):
            g.remove_node(i)


def balance(fixed: nx.DiGraph, g: nx.Graph):
    """Extend the prefixed digraph to make the remaining graph balanced.

    Args:
        fixed (nx.DiGraph): fixed edges
        g (nx.Graph): skeletal graph

    Returns:
        nx.DiGraph: extended fixed graph (derived cycles are included)
        list: a list of derived cycles.
    """

    def _choose_free_edge(g: nx.Graph, dg: nx.DiGraph, node: int):
        """Find an unfixed edge of the node.

        Args:
            g (nx.Graph): _description_
            dg (nx.DiGraph): _description_
            node (int): _description_

        Returns:
            _type_: _description_
        """
        # add dummy nodes to make number of edges be four.
        neis = (list(g[node]) + [-1, -2, -3, -4])[:4]
        # and select one randomly
        np.random.shuffle(neis)
        for nei in neis:
            if not (dg.has_edge(node, nei) or dg.has_edge(nei, node)):
                return nei
        return None

    logger = getLogger()

    # Make a copy to keep the original graph untouched
    _fixed = nx.DiGraph(fixed)

    in_peri = set()
    out_peri = set()
    for node in _fixed:
        # If the node has unfixed edges,
        if _fixed.in_degree[node] + _fixed.out_degree[node] < g.degree[node]:
            # if it is not balanced,
            if _fixed.in_degree[node] > _fixed.out_degree[node]:
                out_peri.add(node)
            elif _fixed.in_degree[node] < _fixed.out_degree[node]:
                in_peri.add(node)

    logger.debug(f"out_peri {out_peri}")
    logger.debug(f"in_peri {in_peri}")

    derivedCycles = []

    while len(out_peri) > 0:
        node = np.random.choice(list(out_peri))
        out_peri -= {node}

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
                in_peri -= {node}
                break
            if node in out_peri:
                logger.debug(f"node {node} is on the out_peri...")
            # if the node can no longer be balanced,
            if max(_fixed.in_degree(node), _fixed.out_degree(node)) * 2 > 4:
                # Start over.
                logger.info(f"Failed to balance. Starting over ...")
                return None, None
            # Find the next node. That may be a decorated one.
            next = _choose_free_edge(g, _fixed, node)
            # fix the edge
            _fixed.add_edge(node, next)
            # record to the path
            if next >= 0:
                path.append(next)
                # if still incoming edges are more than outgoing ones,
                if _fixed.in_degree[node] > _fixed.out_degree[node]:
                    # It is still a perimeter.
                    out_peri.add(node)
            # go ahead
            node = next

            # if it is circular, i.e. if the last node of the path has already included in the path,
            try:
                loc = path[:-1].index(node)
                # Separate the cycle from the path and store in derivedCycles.
                derivedCycles.append(path[loc:])
                # and shorten the path
                path = path[: loc + 1]
            except ValueError:
                pass

    # starting from in_peri
    # Almost the same process, again.
    while len(in_peri) > 0:
        node = np.random.choice(list(in_peri))
        in_peri -= {node}
        logger.debug(
            f"first node {node}, its neighbors {g[node]} {list(_fixed.successors(node))} {list(_fixed.predecessors(node))}"
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
                out_peri -= {node}
                break
            if node in in_peri:
                logger.debug(f"node {node} is on the in_peri...")
                # out_periのノードを何度も通ると、欠陥になってしまう。
            if max(_fixed.in_degree(node), _fixed.out_degree(node)) * 2 > 4:
                logger.info(f"Failed to balance. Starting over ...")
                return None, None
            next = _choose_free_edge(g, _fixed, node)
            # record to the path
            if next >= 0:
                path.append(next)
            # fix the edge  #####
            _fixed.add_edge(next, node)
            # if still incoming edges are more than outgoing ones,
            if next >= 0:
                #####
                if _fixed.in_degree[node] < _fixed.out_degree[node]:
                    in_peri.add(node)
                    logger.debug(
                        f"{node} is added to in_peri {_fixed.in_degree[node]} . {_fixed.out_degree[node]}"
                    )
            # go ahead
            node = next
            # if it is circular
            try:
                loc = path[:-1].index(node)
                derivedCycles.append(path[loc:])
                path = path[: loc + 1]
            except ValueError:
                pass

    if logger.isEnabledFor(DEBUG):
        logger.debug(f"size of g {g.number_of_edges()}")
        logger.debug(f"size of fixed {_fixed.number_of_edges()}")
        assert len(in_peri) == 0, f"In-peri remains. {in_peri}"
        assert len(out_peri) == 0, f"Out-peri remains. {out_peri}"
        logger.debug("re-check perimeters")

        in_peri = set()
        out_peri = set()
        for node in _fixed:
            if node >= 0:
                if _fixed.in_degree[node] + _fixed.out_degree[node] < g.degree[node]:
                    if _fixed.in_degree[node] > _fixed.out_degree[node]:
                        out_peri.add(node)
                    elif _fixed.in_degree[node] < _fixed.out_degree[node]:
                        in_peri.add(node)

        assert len(in_peri) == 0, f"In-peri remains. {in_peri}"
        assert len(out_peri) == 0, f"Out-peri remains. {out_peri}"

        # 拡大したグラフが指定された固定辺をすべて含んでいることを確認。
        for edge in fixed.edges():
            assert _fixed.has_edge(*edge)

    # # remove edges in derivedCycles from _fixed
    # for cycle in derivedCycles:
    #     for edge in zip(cycle, cycle[1:]):
    #         _fixed.remove_edge(*edge)

    _remove_dummy_nodes(_fixed)

    if logger.isEnabledFor(DEBUG):
        logger.debug(f"Number of fixed edges is {_fixed.size()} / {g.size()}")
        logger.debug(f"Number of free cycles: {len(derivedCycles)}")
        ne = sum([len(cycle) - 1 for cycle in derivedCycles])
        logger.debug(f"Number of edges in free cycles: {ne}")

    return _fixed, derivedCycles
