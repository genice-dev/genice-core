from logging import DEBUG, INFO, basicConfig, getLogger

import networkx as nx
import numpy as np

# import py3Dmol
from genice2.genice import GenIce
from genice2.plugin import Format, Lattice

from genice_core import ice_graph


def test_pbc():
    logger = getLogger()
    basicConfig(level=DEBUG)

    np.random.seed(999)

    lattice = Lattice("1h")
    formatter = Format("raw", stage=(1, 2))
    raw = GenIce(lattice, signature="Ice Ih", rep=(6, 6, 6)).generate_ice(formatter)

    # graph is the topology of the hydrogen-bond network
    g = nx.Graph(raw["graph"])
    # reppositions contains the positions of CoM of water in fractional coordinate
    layout = raw["reppositions"]
    # repcell is the cell matrix (transposed)
    cell = raw["repcell"]

    fixed = nx.DiGraph(
        [
            [0, 4],
        ]
    )

    # set orientations of the hydrogen bonds.
    dg = ice_graph(
        g,
        vertexPositions=layout,
        isPeriodicBoundary=True,
        dipoleOptimizationCycles=1000,
        fixedEdges=fixed,
    )
