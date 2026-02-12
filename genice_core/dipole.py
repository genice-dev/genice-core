"""
Optimizes the orientations of directed paths to reduce the net dipole moment.
"""

from logging import getLogger, DEBUG
from typing import List, Optional

import numpy as np
import networkx as nx

from genice_core.compat import accept_aliases


def vector_sum(
    dg: nx.DiGraph, vertex_positions: np.ndarray, is_periodic_boundary: bool = False
) -> np.ndarray:
    """Calculate the net polarization (vector sum) of a digraph.

    Args:
        dg (nx.DiGraph): The digraph.
        vertex_positions (np.ndarray): Positions of the vertices.
        is_periodic_boundary (bool, optional): If true, the vertex positions must be in fractional coordinate. Defaults to False.

    Returns:
        np.ndarray: Net polarization vector.
    """
    pol = np.zeros_like(vertex_positions[0])
    for i, j in dg.edges():
        d = vertex_positions[j] - vertex_positions[i]
        if is_periodic_boundary:
            d -= np.floor(d + 0.5)
        pol += d
    return pol


def _dipole_moment_pbc(path: List[int], vertex_positions: np.ndarray) -> np.ndarray:
    """Calculate the dipole moment of a path with periodic boundary conditions.

    Args:
        path (List[int]): The path to calculate dipole moment for.
        vertex_positions (np.ndarray): Positions of the vertices.

    Returns:
        np.ndarray: The dipole moment vector.
    """
    # vectors between adjacent vertices.
    relative_vector = vertex_positions[path[1:]] - vertex_positions[path[:-1]]
    # PBC wrap
    relative_vector -= np.floor(relative_vector + 0.5)
    # total dipole along the chain (or a cycle)
    return np.sum(relative_vector, axis=0)


@accept_aliases(
    vertexPositions="vertex_positions",
    dipoleOptimizationCycles="dipole_optimization_cycles",
    isPeriodicBoundary="is_periodic_boundary",
    targetPol="target_pol",
)
def optimize(
    paths: List[List[int]],
    vertex_positions: np.ndarray,
    dipole_optimization_cycles: int = 2000,
    is_periodic_boundary: bool = False,
    target_pol: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """Minimize the net polarization by flipping several paths.

    It is assumed that every vector has an identical dipole moment.

    Args:
        paths (List[List[int]]): List of directed paths. A path is a list of integers. A path with identical labels at first and last items are considered to be cyclic.
        vertex_positions (np.ndarray): Positions of the nodes.
        dipole_optimization_cycles (int, optional): Number of random orientations for the paths. Defaults to 2000.
        is_periodic_boundary (bool, optional): If `True`, the positions of the nodes must be in the fractional coordinate system. Defaults to False.
        target_pol (Optional[np.ndarray], optional): Target value for the dipole-moment optimization. Defaults to None.

    Returns:
        List[List[int]]: Optimized paths with minimized net polarization.
    """
    logger = getLogger()

    if target_pol is None:
        target_pol = np.zeros_like(vertex_positions[0])

    # polarized chains and cycles. Small cycle of dipoles are eliminated.
    polarized_edges: List[int] = []

    dipoles: List[np.ndarray] = []
    for i, path in enumerate(paths):
        if is_periodic_boundary:
            chain_pol = _dipole_moment_pbc(path, vertex_positions)
            # if it is large enough, i.e. if it is a spanning cycle or a chain
            if chain_pol @ chain_pol > 1e-6:
                dipoles.append(chain_pol)
                polarized_edges.append(i)
        else:
            # dipole moment of a path; NOTE: No PBC.
            if path[0] != path[-1]:
                # If no PBC, a chain pol is simply an end-to-end pol.
                chain_pol = vertex_positions[path[-1]] - vertex_positions[path[0]]
                dipoles.append(chain_pol)
                polarized_edges.append(i)
    
    if not dipoles:
        return paths

    dipoles_arr = np.array(dipoles)

    optimal_parities = np.random.randint(2, size=len(dipoles_arr)) * 2 - 1
    minimal_residual_pol = optimal_parities @ dipoles_arr - target_pol

    if logger.isEnabledFor(DEBUG):
        logger.debug(f"initial {optimalParities @ dipoles_arr} target {target_pol}")
        logger.debug(f"dipoles {dipoles_arr}")
        for i, parity in zip(polarized_edges, optimal_parities):
            logger.debug(f"{parity}: {paths[i]}")

    loop = 0
    for loop in range(dipole_optimization_cycles):
        # random sequence of +1/-1
        parities = np.random.randint(2, size=len(dipoles_arr)) * 2 - 1

        # Set directions to chains by parity.
        residual_pol = parities @ dipoles_arr - target_pol

        # If the new directions give better (smaller) net dipole moment,
        if residual_pol @ residual_pol < minimal_residual_pol @ minimal_residual_pol:
            # that is the optimal
            minimal_residual_pol = residual_pol
            optimal_parities = parities
            logger.debug(f"Depol. loop {loop}: {minimal_residual_pol}")

            # if well-converged,
            if minimal_residual_pol @ minimal_residual_pol < 1e-4:
                logger.debug("Optimized.")
                break

    logger.info(f"Depol. loop {loop}: {minimal_residual_pol}")

    # invert some chains according to parity_optimal
    for i, parity in zip(polarized_edges, optimal_parities):
        if parity < 0:
            # invert the chain
            paths[i] = paths[i][::-1]

    return paths
