"""
.. include:: ../README.md
"""

import numpy as np
import networkx as nx
from genice_core.topology import (
    noodlize,
    split_into_simple_paths,
    connect_matching_paths,
)
from genice_core.dipole import optimize, vector_sum, _dipole_moment_pbc
from typing import Union, List, Optional
from logging import getLogger, DEBUG


def ice_graph(
    g: nx.Graph,
    vertexPositions: Union[np.ndarray, None] = None,
    isPeriodicBoundary: bool = False,
    dipoleOptimizationCycles: int = 0,
    fixedEdges: nx.DiGraph = nx.DiGraph(),
    max_attempts: int = 100,
) -> Optional[nx.DiGraph]:
    """Make a digraph that obeys the ice rules.

    Args:
        g (nx.Graph): An ice-like undirected graph. Node labels in the graph g must be consecutive integers from 0 to N-1, where N is the number of nodes, and the labels correspond to the order in vertexPositions.
        vertexPositions (Union[nx.ndarray, None], optional): Positions of the vertices in N x 3 numpy array. Defaults to None.
        isPeriodicBoundary (bool, optional): If True, the positions are considered to be in the fractional coordinate system. Defaults to False.
        dipoleOptimizationCycles (int, optional): Number of iterations to reduce the net dipole moment. Defaults to 0 (no iteration).
        fixedEdges (nx.DiGraph, optional): A digraph made of edges whose directions are fixed. All edges in fixed must also be included in g. Defaults to an empty graph.
        max_attempts (int, optional): Maximum number of attempts to connect matching paths. Defaults to 100.

    Returns:
        Optional[nx.DiGraph]: An ice graph that obeys the ice rules, or None if no solution is found within max_attempts.
    """
    logger = getLogger()

    # derived cycles in extending the fixed edges.
    derivedCycles: List[List[int]] = []

    if fixedEdges.size() > 0:
        if logger.isEnabledFor(DEBUG):
            for edge in fixedEdges.edges():
                logger.debug(f"FIXED EDGE {edge}")

        # connect matching paths
        processedEdges = None
        for attempt in range(max_attempts):
            # It returns Nones when it fails to connect paths.
            # The processedEdges also include derivedCycles.
            processedEdges, derivedCycles = connect_matching_paths(fixedEdges, g)
            if processedEdges:
                break
            logger.info(f"Attempt {attempt + 1}/{max_attempts} failed to connect paths")
        else:
            logger.error(f"Failed to find a solution after {max_attempts} attempts")
            return None
    else:
        processedEdges = nx.DiGraph()

    # really fixed in connect_matching_paths()
    finallyFixedEdges = nx.DiGraph(processedEdges)
    for cycle in derivedCycles:
        for edge in zip(cycle, cycle[1:]):
            finallyFixedEdges.remove_edge(*edge)

    # Divide the remaining (unfixed) part of the graph into a noodle graph
    dividedGraph = noodlize(g, processedEdges)

    # Simplify paths ( paths with least crossings )
    paths = list(split_into_simple_paths(len(g), dividedGraph)) + derivedCycles

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

    # Combine everything together
    dg = nx.DiGraph(finallyFixedEdges)

    for path in paths:
        nx.add_path(dg, path)

    # Verify that the graph obeys the ice rules
    for node in dg:
        if fixedEdges.has_node(node):
            if fixedEdges.in_degree(node) > 2 or fixedEdges.out_degree(node) > 2:
                continue
        assert (
            dg.in_degree(node) <= 2
        ), f"{node} {list(dg.successors(node))} {list(dg.predecessors(node))}"
        assert dg.out_degree(node) <= 2

    return dg
