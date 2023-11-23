"""
Optimizes the orientations of directed paths to reduce the net dipole moment.
"""
from logging import getLogger, DEBUG
from typing import Union

import numpy as np
import networkx as nx


def vector_sum(
    dg: nx.DiGraph, vertexPositions: np.ndarray, isPeriodicBoundary: bool = False
) -> np.ndarray:
    """Net polarization (actually a vector sum) of a digraph

    Args:
        dg (nx.DiGraph): The digraph.
        vertexPositions (np.ndarray): Positions of the vertices.
        isPeriodicBoundary (bool, optional): If true, the vertex positions must be in fractional coordinate. Defaults to False.

    Returns:
        np.ndarray: net polarization
    """
    pol = np.zeros_like(vertexPositions[0])
    for i, j in dg.edges():
        d = vertexPositions[j] - vertexPositions[i]
        if isPeriodicBoundary:
            d -= np.floor(d + 0.5)
        pol += d
    return pol


def _dipole_moment_pbc(path, vertexPositions):
    # vectors between adjacent vertices.
    relativeVector = vertexPositions[path[1:]] - vertexPositions[path[:-1]]
    # PBC wrap
    relativeVector -= np.floor(relativeVector + 0.5)
    # total dipole along the chain (or a cycle)
    return np.sum(relativeVector, axis=0)


def optimize(
    paths: list[list],
    vertexPositions: np.ndarray,
    dipoleOptimizationCycles: int = 2000,
    isPeriodicBoundary: bool = False,
    targetPol: Union[np.ndarray, None] = None,
) -> list[list]:
    """Minimize the net polarization by flipping several paths.

    It is assumed that every vector has an identical dipole moment.

    Args:
        paths (list of list): List of directed paths. A path is a list of integer. A path with identical labels at first and last items are considered to be cyclic.
        pos (nx.ndarray[*,3]): Positions of the nodes.
        maxiter (int, optional): Number of random orientations for the paths. Defaults to 1000.
        pbc (bool, optional): If `True`, the positions of the nodes must be in the fractional coordinate system.
        target (np.ndarray, optional): Target value for the dipole-moment optimization.
    Returns:
        list of paths: Optimized paths.
    """
    logger = getLogger()

    if dipoleOptimizationCycles < 1:
        return paths

    if targetPol is None:
        targetPol = np.zeros_like(vertexPositions[0])

    # polarized chains and cycles. Small cycle of dipoles are eliminated.
    polarizedEdges = []

    dipoles = []
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

    optimalParities = np.ones(len(dipoles))
    optimalPol = optimalParities @ dipoles - targetPol
    logger.debug(f"initial {optimalParities @ dipoles} target {targetPol}")

    if len(dipoles) > 0 and logger.isEnabledFor(DEBUG):
        logger.debug(f"dipoles {dipoles}")
        order = np.argsort(np.linalg.norm(dipoles, axis=1))

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
