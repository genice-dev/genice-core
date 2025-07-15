# uses GenIce API to generate a hydrogen-ordered ice 11

import networkx as nx
import genice_core
import numpy as np

from genice2.genice import GenIce
from genice2.plugin import Format, Lattice

from logging import getLogger, INFO, DEBUG, basicConfig

basicConfig(level=INFO)
logger = getLogger()


for i in range(1000):
    np.random.seed(i)

    lattice = Lattice("ice11")
    # exectute the stages 1 and 4 to generate molecular positions and the digraph of HBs, and get the raw data.
    formatter = Format("raw", stage=(1, 4))
    raw = GenIce(lattice, signature="Ice Ih", rep=(4, 4, 4)).generate_ice(formatter)
    print(raw["reppositions"].shape)
    # fix some edges randomly
    fixed = nx.DiGraph()

    choice = np.random.random(len(raw["digraph"].edges())) < 0.03
    for i, edge in enumerate(raw["digraph"].edges()):
        if choice[i]:
            fixed.add_edge(*edge)

    dg = genice_core.ice_graph(
        nx.Graph(raw["digraph"]),
        raw["reppositions"],
        isPeriodicBoundary=True,
        dipoleOptimizationCycles=1000,
        fixedEdges=fixed,
        max_attempts=10000,
    )
    print("#" * 120, i)
