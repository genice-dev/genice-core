"""Legacy connect_engine implementations emit DeprecationWarning."""

import warnings

import networkx as nx
import pytest

from genice_core.topology_nx import connect_matching_paths
from genice_core.topology_nx.connect_mcf import connect_matching_paths_mcf


def test_legacy_connect_matching_paths_warns():
    g = nx.path_graph(5)
    fixed = nx.DiGraph([(0, 1)])
    with pytest.warns(DeprecationWarning, match="connect_matching_paths_mcf"):
        connect_matching_paths(fixed, g)


def test_mcf_connect_does_not_warn():
    g = nx.path_graph(5)
    fixed = nx.DiGraph([(0, 1), (3, 4)])
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result, _ = connect_matching_paths_mcf(fixed, g)
    assert result is not None
