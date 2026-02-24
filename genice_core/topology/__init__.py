"""
Arrange edges appropriately. Internal representation uses plain arrays (no NetworkX).
"""

from genice_core.topology.noodle import (
    noodlize,
    split_into_simple_paths,
)
from genice_core.topology.connect_random import connect_matching_paths
from genice_core.topology.connect_bfs import connect_matching_paths_bfs
from genice_core.topology.connect_sp2st import (
    connect_matching_paths_SP2ST,
)
# Backward compatibility alias
connect_matching_paths_BFS1 = connect_matching_paths_SP2ST

__all__ = [
    "noodlize",
    "split_into_simple_paths",
    "connect_matching_paths",
    "connect_matching_paths_bfs",
    "connect_matching_paths_SP2ST",
    "connect_matching_paths_BFS1",
]
