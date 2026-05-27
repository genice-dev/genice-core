"""Tests for connect_matching_paths_mcf and MCF as ice_graph default connect_engine."""

import inspect
import unittest

import networkx as nx

from genice_core import ice_graph
from genice_core.topology_nx import connect_matching_paths_mcf
from genice_core.topology_nx.connect_random import _get_perimeters


def _perimeters_cleared(fixed: nx.DiGraph, g: nx.Graph) -> bool:
    in_peri, out_peri = _get_perimeters(fixed, g)
    return len(in_peri) == 0 and len(out_peri) == 0


def _max_directed_degree(dg: nx.DiGraph, node: int) -> int:
    return max(dg.in_degree(node), dg.out_degree(node))


class TestConnectMatchingPathsMcf(unittest.TestCase):
    def test_ice_graph_default_connect_engine_is_mcf(self):
        sig = inspect.signature(ice_graph)
        default = sig.parameters["connect_engine"].default
        self.assertIs(default, connect_matching_paths_mcf)

    def test_no_perimeter_returns_fixed_unchanged(self):
        g = nx.cycle_graph(4)
        fixed = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 0)])
        result, cycles = connect_matching_paths_mcf(fixed, g)
        self.assertIsNotNone(result)
        self.assertEqual(cycles, [])
        self.assertTrue(nx.is_isomorphic(result, fixed))

    def test_connects_path_perimeters(self):
        # 0-1-2-3-4 with fixed 0->1 and 3->4 leaves out_peri={1}, in_peri={3}
        g = nx.path_graph(5)
        fixed = nx.DiGraph([(0, 1), (3, 4)])
        result, cycles = connect_matching_paths_mcf(fixed, g)
        self.assertIsNotNone(result)
        self.assertEqual(cycles, [])
        self.assertTrue(result.has_edge(0, 1))
        self.assertTrue(result.has_edge(3, 4))
        self.assertTrue(_perimeters_cleared(result, g))
        for n in g.nodes():
            self.assertLessEqual(_max_directed_degree(result, n), 2)

    def test_preserves_existing_fixed_edges(self):
        g = nx.dodecahedral_graph()
        u, v = next(iter(g.edges()))
        fixed = nx.DiGraph()
        fixed.add_edge(u, v)
        result, _ = connect_matching_paths_mcf(fixed, g)
        self.assertIsNotNone(result)
        self.assertTrue(result.has_edge(u, v))
        self.assertFalse(result.has_edge(v, u))

    def test_returns_none_when_perimeter_counts_mismatch(self):
        # Triangle with two out_peri nodes, no in_peri -> |out| != |in|
        g = nx.cycle_graph(3)
        fixed = nx.DiGraph([(0, 1), (0, 2)])
        result, cycles = connect_matching_paths_mcf(fixed, g)
        self.assertIsNone(result)
        self.assertEqual(cycles, [])

    def test_free_edges_not_used_twice(self):
        g = nx.path_graph(5)
        fixed = nx.DiGraph([(0, 1), (3, 4)])
        result, _ = connect_matching_paths_mcf(fixed, g)
        self.assertIsNotNone(result)
        new_edges = set(result.edges()) - set(fixed.edges())
        used_undirected: set[tuple[int, int]] = set()
        for u, v in new_edges:
            key = (min(u, v), max(u, v))
            self.assertNotIn(key, used_undirected, f"undirected edge {{{u},{v}}} used twice")
            used_undirected.add(key)
            self.assertTrue(g.has_edge(u, v))

    def test_dodecahedral_single_fixed_edge(self):
        g = nx.dodecahedral_graph()
        u, v = next(iter(g.edges()))
        fixed = nx.DiGraph()
        fixed.add_edge(u, v)
        result, _ = connect_matching_paths_mcf(fixed, g)
        self.assertIsNotNone(result)
        self.assertTrue(_perimeters_cleared(result, g))

    def test_mcf_is_deterministic(self):
        g = nx.dodecahedral_graph()
        fixed = nx.DiGraph()
        fixed.add_edge(0, 1)
        a, _ = connect_matching_paths_mcf(fixed, g)
        b, _ = connect_matching_paths_mcf(fixed, g)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertEqual(set(a.edges()), set(b.edges()))

    def test_mcf_clears_perimeters_on_dodecahedral(self):
        g = nx.dodecahedral_graph()
        fixed = nx.DiGraph()
        fixed.add_edge(0, 1)
        result, _ = connect_matching_paths_mcf(fixed, g)
        self.assertIsNotNone(result)
        self.assertTrue(_perimeters_cleared(result, g))

    def test_ice_graph_with_fixed_edges_uses_mcf_by_default(self):
        g = nx.dodecahedral_graph()
        fixed = nx.DiGraph()
        fixed.add_edge(0, 1)
        dg = ice_graph(g, fixed_edges=fixed, pairing_attempts=1)
        self.assertIsNotNone(dg)
        self.assertTrue(dg.has_edge(0, 1))
        for node in dg.nodes():
            self.assertLessEqual(dg.in_degree(node), 2)
            self.assertLessEqual(dg.out_degree(node), 2)


if __name__ == "__main__":
    unittest.main()
