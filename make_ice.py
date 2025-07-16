# uses GenIce API to generate a hydrogen-ordered ice 11

import networkx as nx
import genice_core
import numpy as np

from genice2.genice import GenIce
from genice2.plugin import Format, Lattice

from logging import getLogger, INFO, DEBUG, basicConfig

basicConfig(level=DEBUG)
logger = getLogger()


np.random.seed(999)

lattice = Lattice("ice11")
# exectute the stages 1 and 4 to generate molecular positions and the digraph of HBs, and get the raw data.
formatter = Format("raw", stage=(1, 4))
raw = GenIce(lattice, signature="Ice Ih", rep=(4, 4, 4)).generate_ice(formatter)
N = raw["reppositions"].shape[0]
print(N)
for i in range(N):
    print(*raw["reppositions"][i])
for edge in raw["digraph"].edges():
    print(*edge)
