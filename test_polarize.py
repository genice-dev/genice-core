import pairlist as pl
import networkx as nx
import genice_core
from genice_core.dipole import vector_sum as dipole
import numpy as np
import time
import matplotlib.pyplot as plt
from logging import getLogger, basicConfig, INFO, DEBUG

from genice_core.topology_nx import force_polarize
def diamond(N: int) -> np.ndarray:
    """Diamond lattice. == ice 1c

    Args:
        N (int): Number of unit cells per an edge of the simulation cell.

    Returns:
        np.ndarray: atomic positions in the fractional coordinate.
    """
    # make an FCC
    xyz = np.array(
        [
            (x, y, z)
            for x in range(N)
            for y in range(N)
            for z in range(N)
            if (x + y + z) % 2 == 0
        ]
    )
    xyz = np.vstack([xyz, xyz + 0.5])
    return xyz / N


basicConfig(level=INFO)
logger = getLogger()

np.random.seed(998)

depol = 1000

N = 32
logger.info(f"Size {N}")

pos = diamond(N)
cell = np.diag([N, N, N])

# adjacency graph
g = nx.Graph(
    [
        (i, j)
        for i, j in pl.pairs_iter(pos, 1, cell, fractional=True, distance=False)
    ]
)

fixed = nx.DiGraph()

dg = genice_core.ice_graph(
    g,
    vertex_positions=pos,
    dipole_optimization_cycles=depol,
    is_periodic_boundary=True,
    fixed_edges=fixed,
    target_pol=(20 ,0, 0),
)

edges = [[i, j] for i, j in dg.edges()]
print(dipole(edges, pos, is_periodic_boundary=True))

dg2 = force_polarize(hbn=dg, user_fixed=fixed, vertex_positions=pos, target_pol=(20 ,0, 0), polarize_cycles=1000)

edges2 = [[i, j] for i, j in dg2.edges()]
print(dipole(edges2, pos, is_periodic_boundary=True))
