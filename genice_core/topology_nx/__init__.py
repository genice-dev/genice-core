"""
NetworkX-based topology helpers (no array backend).
"""

from .noodle import (
    noodlize,
    split_into_simple_paths,
)
from .connect_random import connect_matching_paths
from .connect_mcf import connect_matching_paths_mcf
from ..dipole import vector_sum, _path_dipole_moment_pbc
import numpy as np
import networkx as nx
from collections import defaultdict
from logging import getLogger

__all__ = [
    "noodlize",
    "split_into_simple_paths",
    "connect_matching_paths",  # deprecated
    "connect_matching_paths_mcf",
]


def reverse_path_edges(dg: nx.DiGraph, path: list[int]) -> None:
    for u, v in zip(path, path[1:]):
        dg.remove_edge(u, v)
        dg.add_edge(v, u)


def polarize(
    dg: nx.DiGraph,
    vertex_positions: np.ndarray,
    residents: defaultdict,
    grid_shape: tuple,
    target_pol: np.ndarray,
):
    while True:
        A = np.random.choice(list(dg.nodes()))
        if dg.in_degree(A) > 0 and dg.out_degree(A) > 0:
            break

    address_A = np.array(
        [
            int(vertex_positions[A][0] * grid_shape[0]),
            int(vertex_positions[A][1] * grid_shape[1]),
            int(vertex_positions[A][2] * grid_shape[2]),
        ]
    )
    antipode_x = 0 if abs(target_pol[0]) < 0.01 else grid_shape[0] // 2
    antipode_y = 0 if abs(target_pol[1]) < 0.01 else grid_shape[1] // 2
    antipode_z = 0 if abs(target_pol[2]) < 0.01 else grid_shape[2] // 2
    address_B = address_A + np.array([antipode_x, antipode_y, antipode_z])
    address_B %= grid_shape
    residents_B = residents[tuple(address_B)]

    while True:
        B = np.random.choice(list(residents_B))
        if dg.in_degree(B) > 0 and dg.out_degree(B) > 0:
            break

    A_to_B = nx.shortest_path(dg, A, B)
    B_to_A = nx.shortest_path(dg, B, A)

    AB_edges = [(u, v) for u, v in zip(A_to_B, A_to_B[1:])]
    BA_edges = [(u, v) for u, v in zip(B_to_A, B_to_A[1:])]
    if set(AB_edges) & set(BA_edges):
        return np.zeros(3)

    pol_AB = _path_dipole_moment_pbc(A_to_B, vertex_positions)
    pol_BA = _path_dipole_moment_pbc(B_to_A, vertex_positions)
    delta_pol = pol_AB + pol_BA

    if np.allclose(delta_pol, 0, atol=1e-3):
        return np.zeros(3)

    new_pol = target_pol + delta_pol * 2
    if new_pol @ new_pol <= target_pol @ target_pol:
        cycle = A_to_B + B_to_A[1:]
        reverse_path_edges(dg, cycle)
        return -delta_pol * 2

    return np.zeros(3)


def force_polarize(
    hbn: nx.DiGraph,
    user_fixed: nx.DiGraph,
    vertex_positions: np.ndarray,
    target_pol: np.ndarray,
    dipole_optimization_cycles2: int = 1000,
):
    # hbnには、genice_coreアルゴリズムで最適化された水素結合がはいっている。
    # そのうち、当初からユーザーにより固定されていた辺はuser_fixedにはいっている。
    # 残る部分を再配置することで、分極をできるだけtarget_polに近付ける。
    if dipole_optimization_cycles2 <= 0:
        return nx.DiGraph(hbn)

    logger = getLogger()
    dg = nx.DiGraph(hbn)
    for edge in hbn.edges():
        if user_fixed.has_edge(edge[0], edge[1]):
            dg.remove_edge(edge[0], edge[1])

    original_pol = vector_sum(
        list(hbn.edges()), vertex_positions, is_periodic_boundary=True
    )

    N = len(vertex_positions)
    Ng = int((N / 5) ** (1 / 3))

    residents = defaultdict(set)
    for idx, pos in enumerate(vertex_positions):
        residents[int(pos[0] * Ng), int(pos[1] * Ng), int(pos[2] * Ng)].add(idx)

    last_loop = -1
    for i in range(dipole_optimization_cycles2):
        delta_pol = polarize(
            dg,
            vertex_positions,
            residents,
            grid_shape=(Ng, Ng, Ng),
            target_pol=target_pol - original_pol,
        )
        original_pol += delta_pol
        remain = target_pol - original_pol
        last_loop = i
        logger.debug(
            f"Polarization 2: loop {i}: yet to be optimized {remain[0]:.2f}, {remain[1]:.2f}, {remain[2]:.2f}"
        )
        if np.allclose(target_pol - original_pol, 0, atol=1e-3):
            break
    remain = target_pol - original_pol
    logger.info(
        f"Polarization 2: loop {last_loop}: yet to be optimized {remain[0]:.2f}, {remain[1]:.2f}, {remain[2]:.2f}"
    )
    for edge in user_fixed.edges():
        dg.add_edge(edge[0], edge[1])

    return dg
