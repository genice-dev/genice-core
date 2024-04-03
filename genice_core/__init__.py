"""
.. include:: ../README.md
"""

import numpy as np
import networkx as nx
from genice_core.topology import noodlize, split_into_simple_paths, balance
from genice_core.dipole import optimize, vector_sum, _dipole_moment_pbc
from typing import Union
from logging import getLogger, DEBUG

__version__ = "0.9.1"


def ice_graph(
    g: nx.Graph,
    vertexPositions: Union[np.ndarray, None] = None,
    isPeriodicBoundary: bool = False,
    dipoleOptimizationCycles: int = 0,
    fixedEdges: nx.DiGraph = nx.DiGraph(),
) -> nx.DiGraph:
    """Make a digraph that obeys the ice rules.

    Args:
        g (nx.Graph): A ice-like undirected graph.
        vertexPositions (Union[nx.ndarray, None], optional): Positions of the vertices. Defaults to None.
        isPeriodicBoundary (bool, optional): If True, the positions are considered to be in the fractional coordinate system. Defaults to False.
        dipoleOptimizationCycles (int, optional): Number of iterations to reduce the net dipole moment. Defaults to 0 (no iteration).
        fixed (nx.DiGraph, optional): A digraph made of edges whose directions are fixed. All edges in fixed must also be included in g. Defaults to an empty graph.

    Returns:
        nx.DiGraph: An ice graph.
    """
    logger = getLogger()

    # derived cycles in extending the fixed edges.
    derivedCycles = []

    if fixedEdges.size() > 0:
        # コメントが日本語の部分はまだデバッグ中と思え。
        if logger.isEnabledFor(DEBUG):
            for edge in fixedEdges.edges():
                logger.debug(f"EDGE {edge}")

        # balance fixed edges
        processedEdges = None
        while processedEdges is None:
            # It returns Nones when it fails to balance.
            # The processedEdges also include derivedCycles.
            processedEdges, derivedCycles = balance(fixedEdges, g)
    else:
        processedEdges = nx.DiGraph()

    # really fixed in balance()
    finallyFixedEdges = nx.DiGraph(processedEdges)
    for cycle in derivedCycles:
        for edge in zip(cycle, cycle[1:]):
            finallyFixedEdges.remove_edge(*edge)

    # Divide the remaining (unfixed) part of the graph into a noodle graph
    dividedGraph = noodlize(g, processedEdges)

    # Simplify paths ( paths with least crossings )
    paths = list(split_into_simple_paths(len(g), dividedGraph)) + derivedCycles

    # 欠陥がない氷なら、すべてcycleになっているはず。
    # for path in paths:
    #     assert path[0] == path[-1]

    # arrange the orientations here if you want to balance the polarization
    if vertexPositions is not None:
        # Set the targetPol in order to cancel the polarization in the fixed part.
        targetPol = -vector_sum(finallyFixedEdges, vertexPositions, isPeriodicBoundary)

        paths = optimize(
            paths,
            vertexPositions,
            isPeriodicBoundary=isPeriodicBoundary,
            dipoleOptimizationCycles=dipoleOptimizationCycles,
            targetPol=targetPol,
        )

    # 欠陥がない氷なら、すべてcycleになっているはず。
    # for path in paths:
    #     assert path[0] == path[-1]

    # Combine everything together
    dg = nx.DiGraph(finallyFixedEdges)

    # 30-->97-->31という辺はextendedFixedEdgesに含まれるべきではない
    # for edge in extendedFixedEdges.edges():
    #     logger.debug(f"EDGE eFE {edge}")
    # logger.debug(f"{list(dg.predecessors(97))} --> 97 --> {list(dg.successors(97))}")
    for path in paths:
        nx.add_path(dg, path)

    # Does the graph really obey the ice rules?
    # if logger.isEnabledFor(DEBUG):
    for node in dg:
        if fixedEdges.has_node(node):
            if fixedEdges.in_degree(node) > 2 or fixedEdges.out_degree(node) > 2:
                continue
        assert (
            dg.in_degree(node) <= 2
        ), f"{node} {list(dg.successors(node))} {list(dg.predecessors(node))}"
        assert dg.out_degree(node) <= 2

    # bug? まれにこのチェックでひっかかる場合があるようだ。

    # # この時点で、pathsを検査しておく。
    # if logger.isEnabledFor(DEBUG):
    #     gg = nx.Graph(extendedFixedEdges)
    #     for path in paths:
    #         nx.add_path(gg, path)
    #     logger.debug(f"Size g {g.number_of_nodes()} {g.number_of_edges()}")
    #     logger.debug(f"Size gg {gg.number_of_nodes()} {gg.number_of_edges()}")
    #     assert g.number_of_edges() == gg.number_of_edges()
    #     e1 = set([(min(i, j), max(i, j)) for i, j in g.edges()])
    #     e2 = set([(min(i, j), max(i, j)) for i, j in gg.edges()])
    #     logger.debug(
    #         f"{sorted(list(e1 - e2))} edges only in original undirected graph."
    #     )
    #     logger.debug(f"{sorted(list(e2 - e1))} edges only in derived directed graph.")

    return dg
