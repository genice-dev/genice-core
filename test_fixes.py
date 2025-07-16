# uses GenIce API to generate a hydrogen-ordered ice 11

import networkx as nx
import genice_core
import numpy as np

from genice2.genice import GenIce
from genice2.plugin import Format, Lattice

from logging import getLogger, INFO, DEBUG, basicConfig

# basicConfig(level=DEBUG)
basicConfig(level=INFO)
logger = getLogger()

rpos = []
edges = []
with open("ice11.txt", "r") as f:
    N = int(f.readline())
    for i in range(N):
        rpos.append(list(map(float, f.readline().split())))
    for edge in f.readlines():
        edges.append(list(map(int, edge.split())))
# print(f"{rpos=}")
# print(f"{edges=}")

rpos = np.array(rpos)

# edgeをいくつか省くことでわざと周縁を作る。
# edges = edges[:-10]


for i in range(10000):
    np.random.seed(i)
    print(i)

    # fix some edges randomly
    fixed = nx.DiGraph()

    choice = np.random.random(len(edges)) < 0.01
    for i, edge in enumerate(edges):
        if choice[i]:
            fixed.add_edge(*edge)
    logger.debug(f"fixed {fixed.number_of_edges()}")

    dg = genice_core.ice_graph(
        nx.Graph(edges),
        rpos,
        isPeriodicBoundary=True,
        dipoleOptimizationCycles=100,
        fixedEdges=fixed,
        max_attempts=100,
    )
