"""
Optimizes the orientations of directed paths to reduce the net dipole moment.
"""

from logging import getLogger, DEBUG
from typing import Union, List, Optional

import numpy as np
import networkx as nx


def vector_sum(
    dg: nx.DiGraph, vertexPositions: np.ndarray, isPeriodicBoundary: bool = False
) -> np.ndarray:
    """Calculate the net polarization (vector sum) of a digraph.

    Args:
        dg (nx.DiGraph): The digraph.
        vertexPositions (np.ndarray): Positions of the vertices.
        isPeriodicBoundary (bool, optional): If true, the vertex positions must be in fractional coordinate. Defaults to False.

    Returns:
        np.ndarray: Net polarization vector.
    """
    pol = np.zeros_like(vertexPositions[0])
    for i, j in dg.edges():
        d = vertexPositions[j] - vertexPositions[i]
        if isPeriodicBoundary:
            d -= np.floor(d + 0.5)
        pol += d
    return pol


def _dipole_moment_pbc(path: List[int], vertexPositions: np.ndarray) -> np.ndarray:
    """Calculate the dipole moment of a path with periodic boundary conditions.

    Args:
        path (List[int]): The path to calculate dipole moment for.
        vertexPositions (np.ndarray): Positions of the vertices.

    Returns:
        np.ndarray: The dipole moment vector.
    """
    # vectors between adjacent vertices.
    relativeVector = vertexPositions[path[1:]] - vertexPositions[path[:-1]]
    # PBC wrap
    relativeVector -= np.floor(relativeVector + 0.5)
    # total dipole along the chain (or a cycle)
    return np.sum(relativeVector, axis=0)


def optimize(
    paths: List[List[int]],
    vertexPositions: np.ndarray,
    dipoleOptimizationCycles: int = 2000,
    isPeriodicBoundary: bool = False,
    targetPol: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """Minimize the net polarization by flipping several paths.

    It is assumed that every vector has an identical dipole moment.

    Args:
        paths (List[List[int]]): List of directed paths. A path is a list of integers. A path with identical labels at first and last items are considered to be cyclic.
        vertexPositions (np.ndarray): Positions of the nodes.
        dipoleOptimizationCycles (int, optional): Number of random orientations for the paths. Defaults to 2000.
        isPeriodicBoundary (bool, optional): If `True`, the positions of the nodes must be in the fractional coordinate system. Defaults to False.
        targetPol (Optional[np.ndarray], optional): Target value for the dipole-moment optimization. Defaults to None.

    Returns:
        List[List[int]]: Optimized paths with minimized net polarization.
    """
    logger = getLogger()

    if targetPol is None:
        targetPol = np.zeros_like(vertexPositions[0])

    # polarized chains and cycles. Small cycle of dipoles are eliminated.
    polarizedEdges: List[int] = []

    dipoles: List[np.ndarray] = []
    for i, path in enumerate(paths):
        if isPeriodicBoundary:
            chainPol = _dipole_moment_pbc(path, vertexPositions)
            # if it is large enough, i.e. if it is a spanning cycle or a chain
            if chainPol @ chainPol > 1e-6:
                dipoles.append(chainPol)
                polarizedEdges.append(i)
        else:
            # dipole moment of a path; NOTE: No PBC.
            if path[0] != path[-1]:
                # If no PBC, a chain pol is simply an end-to-end pol.
                chainPol = vertexPositions[path[-1]] - vertexPositions[path[0]]
                dipoles.append(chainPol)
                polarizedEdges.append(i)
    dipoles = np.array(dipoles)

    optimalParities = np.random.randint(2, size=len(dipoles)) * 2 - 1
    optimalPol = optimalParities @ dipoles - targetPol

    if logger.isEnabledFor(DEBUG):
        logger.debug(f"initial {optimalParities @ dipoles} target {targetPol}")
        logger.debug(f"dipoles {dipoles}")
        for i, parity in zip(polarizedEdges, optimalParities):
            logger.debug(f"{parity}: {paths[i]}")

    loop = 0
    for loop in range(dipoleOptimizationCycles):
        # random sequence of +1/-1
        parities = np.random.randint(2, size=len(dipoles)) * 2 - 1

        # Set directions to chains by parity.
        pol = parities @ dipoles - targetPol

        # If the new directions give better (smaller) net dipole moment,
        if pol @ pol < optimalPol @ optimalPol:
            # that is the optimal
            optimalPol = pol
            optimalParities = parities
            logger.debug(f"Depol. loop {loop}: {optimalPol}")

            # if well-converged,
            if optimalPol @ optimalPol < 1e-4:
                logger.debug("Optimized.")
                break

    logger.info(f"Depol. loop {loop}: {optimalPol}")

    # invert some chains according to parity_optimal
    for i, parity in zip(polarizedEdges, optimalParities):
        if parity < 0:
            # invert the chain
            paths[i] = paths[i][::-1]

    return paths
