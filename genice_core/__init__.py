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
from genice_core.dipole import optimize, vector_sum
from genice_core.compat import accept_aliases
from typing import Union, List, Optional
from logging import getLogger, DEBUG


@accept_aliases(
    vertexPositions="vertex_positions",
    isPeriodicBoundary="is_periodic_boundary",
    dipoleOptimizationCycles="dipole_optimization_cycles",
    fixedEdges="fixed_edges",
    pairingAttempts="pairing_attempts",
)
def ice_graph(
    g: nx.Graph,
    vertex_positions: Union[np.ndarray, None] = None,
    is_periodic_boundary: bool = False,
    dipole_optimization_cycles: int = 0,
    fixed_edges: nx.DiGraph = nx.DiGraph(),
    pairing_attempts: int = 100,
    target_pol: Optional[np.ndarray] = np.zeros(3),
) -> Optional[nx.DiGraph]:
    """Make a digraph that obeys the ice rules.

    Args:
        g (nx.Graph): An ice-like undirected graph. Node labels must be consecutive integers 0 to N-1, corresponding to the order in vertex_positions when given.
        vertex_positions (Union[np.ndarray, None], optional): Positions of the vertices, N x 3 array. Defaults to None. If None, dipole-related options are ignored.
        fixed_edges (nx.DiGraph, optional): A digraph of edges whose directions are fixed; all must be in g. Defaults to an empty graph.

        Only when vertex_positions is not None (dipole optimization):
        is_periodic_boundary (bool, optional): If True, positions are in fractional coordinates. Defaults to False.
        dipole_optimization_cycles (int, optional): Number of iterations to reduce the net dipole moment. Defaults to 0 (no optimization).
        target_pol (Optional[np.ndarray], optional): Target polarization; subtracted from the net polarization. Used only when dipole_optimization_cycles > 0. Defaults to zeros(3).

        Only when fixed_edges is not empty:
        pairing_attempts (int, optional): Maximum attempts to pair up the fixed edges. Defaults to 100.

    Returns:
        Optional[nx.DiGraph]: An ice graph that obeys the ice rules, or None if no solution is found within pairing_attempts.
    """
    logger = getLogger()

    if logger.isEnabledFor(DEBUG):
        if vertex_positions is None and (
            is_periodic_boundary or dipole_optimization_cycles != 0 or np.any(target_pol != 0)
        ):
            logger.debug(
                "vertex_positions is None; is_periodic_boundary, "
                "dipole_optimization_cycles, and target_pol are ignored."
            )
        if fixed_edges.size() == 0 and pairing_attempts != 100:
            logger.debug("fixed_edges is empty; pairing_attempts is ignored.")

    # derived cycles in extending the fixed edges.
    derived_cycles: List[List[int]] = []

    if fixed_edges.size() > 0:
        if logger.isEnabledFor(DEBUG):
            for edge in fixed_edges.edges():
                logger.debug(f"FIXED EDGE {edge}")

        # connect matching paths
        processed_edges = None
        for attempt in range(pairing_attempts):
            # It returns Nones when it fails to connect paths.
            # The processed_edges also include derived_cycles.
            processed_edges, derived_cycles = connect_matching_paths(fixed_edges, g)
            if processed_edges:
                break
            logger.info(
                f"Attempt {attempt + 1}/{pairing_attempts} failed to connect paths"
            )
        else:
            logger.error(f"Failed to find a solution after {pairing_attempts} attempts")
            return None
    else:
        processed_edges = nx.DiGraph()

    # really fixed in connect_matching_paths()
    finally_fixed_edges = nx.DiGraph(processed_edges)
    for cycle in derived_cycles:
        for edge in zip(cycle, cycle[1:]):
            finally_fixed_edges.remove_edge(*edge)

    # Divide the remaining (unfixed) part of the graph into a noodle graph
    divided_graph = noodlize(g, processed_edges)

    # Simplify paths ( paths with least crossings )
    paths = list(split_into_simple_paths(len(g), divided_graph)) + derived_cycles

    # arrange the orientations here if you want to balance the polarization
    if vertex_positions is not None:
        # Set the target_pol in order to cancel the polarization in the fixed part.
        target_pol -= vector_sum(
            finally_fixed_edges, vertex_positions, is_periodic_boundary
        )

        paths = optimize(
            paths,
            vertex_positions=vertex_positions,
            is_periodic_boundary=is_periodic_boundary,
            dipole_optimization_cycles=dipole_optimization_cycles,
            target_pol=target_pol,
        )

    # Combine everything together
    dg = nx.DiGraph(finally_fixed_edges)

    for path in paths:
        nx.add_path(dg, path)

    # Verify that the graph obeys the ice rules
    for node in dg:
        if fixed_edges.has_node(node):
            if fixed_edges.in_degree(node) > 2 or fixed_edges.out_degree(node) > 2:
                continue
        assert (
            dg.in_degree(node) <= 2
        ), f"{node} {list(dg.successors(node))} {list(dg.predecessors(node))}"
        assert dg.out_degree(node) <= 2

    return dg
