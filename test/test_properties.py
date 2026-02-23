
import unittest
import warnings
import networkx as nx
import numpy as np
from genice_core import ice_graph
from genice_core.dipole import optimize

class TestIceGraphProperties(unittest.TestCase):
    def test_ice_rules_random_graphs(self):
        """Verify that generated graphs obey ice rules (in/out degree <= 2)."""
        for i in range(10):
            # Generate a random 4-regular graph (ice-like)
            # A dodecahedral graph is 3-regular.
            # Grid graph is 4-regular (mostly).
            # Let's use something simple that resembles ice topology.
            # A 3D grid graph is 6-regular. Ice is 4-coordinated.
            # We can use a random regular graph with d=4.
            
            # Need even number of nodes for cubic, but we need degree 4.
            n_nodes = 20 # Must be small for speed
            try:
                # nx.random_regular_graph(d, n)
                # Ice rules apply to 4-coordinated networks.
                g = nx.random_regular_graph(4, n_nodes, seed=i)
            except nx.NetworkXError:
                continue

            # Assign random positions (needed for ice_graph but strictly not for topology if we don't optimize)
            # But ice_graph requires node labels to correspond to positions indices.
            # And it might use positions for optimization?
            # actually vertex_positions is optional in refactored code? 
            # Reviewing my code: vertex_positions=None is allowed, default is None.
            
            # However, if vertex_positions is None, it skips optimization.
            
            dg = ice_graph(g, pairing_attempts=500)
            
            if dg is None:
                # It might fail to find a valid configuration for random graphs
                continue
                
            for node in dg.nodes():
                in_deg = dg.in_degree(node)
                out_deg = dg.out_degree(node)
                self.assertLessEqual(in_deg, 2, f"Node {node} has in-degree {in_deg} > 2")
                self.assertLessEqual(out_deg, 2, f"Node {node} has out-degree {out_deg} > 2")
                # For a perfect ice crystal, in + out = 4 (for 4-coordinated nodes)
                # But our input graph is 4-regular.
                # So in_deg + out_deg should be 4.
                # Thus in_deg must be exactly 2 and out_deg must be exactly 2 for all nodes?
                # The docstring says "obeys the ice rules", which usually means in=2, out=2.
                # But it says <= 2 in the assertions in the code.
                # Let's verify what the code asserts:
                # assert dg.in_degree(node) <= 2
                # assert dg.out_degree(node) <= 2
                
                # If the underlying graph is 4-regular, then in=2, out=2 is the only way to satisfy sum=4 and <=2 constraints.
                if g.degree(node) == 4:
                     self.assertEqual(in_deg, 2)
                     self.assertEqual(out_deg, 2)

    def test_fixed_edges_respect(self):
        """Verify that fixed edges are respected in the output."""
        g = nx.dodecahedral_graph() # 3-regular
        # Dodecahedral nodes have degree 3.
        # Ice rules: in+out = degree.
        # If degree 3, can be (1,2) or (2,1).
        
        # Pick an edge to fix
        edges = list(g.edges())
        fixed_edge = edges[0]
        u, v = fixed_edge
        
        fixed = nx.DiGraph()
        fixed.add_edge(u, v)
        
        dg = ice_graph(g, fixed_edges=fixed)
        
        self.assertIsNotNone(dg, "Should find a solution for dodecahedral graph")
        
        if dg:
            self.assertTrue(dg.has_edge(u, v), "Fixed edge (u, v) must be present in output")
            self.assertFalse(dg.has_edge(v, u), "Reverse of fixed edge should not be present")


    def test_snake_case_compatibility_alias(self):
        """Verify that old camelCase arguments still work via alias decorator."""
        g = nx.dodecahedral_graph()
        
        # Calling with old argument names
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", DeprecationWarning)
            dg = ice_graph(
                g,
                pairingAttempts=100
            )
        self.assertIsNotNone(dg)

    def test_ice_graph_traditional_camel_case_args(self):
        """従来通りの camelCase 引数だけで ice_graph が動作することを検証する。"""
        g = nx.dodecahedral_graph()
        # 小さいグラフ用の適当な座標
        positions = np.random.rand(g.number_of_nodes(), 3)
        fixed = nx.DiGraph()
        fixed.add_edge(0, 1)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", DeprecationWarning)
            dg = ice_graph(
                g,
                vertexPositions=positions,
                isPeriodicBoundary=False,
                dipoleOptimizationCycles=10,
                fixedEdges=fixed,
                pairingAttempts=50,
            )
        self.assertIsNotNone(dg)
        self.assertTrue(dg.has_edge(0, 1))
        for node in dg.nodes():
            self.assertLessEqual(dg.in_degree(node), 2)
            self.assertLessEqual(dg.out_degree(node), 2)

    def test_ice_graph_accepts_adjacency_list(self):
        """ice_graph accepts g as adjacency list (list of lists of neighbor indices)."""
        g_nx = nx.dodecahedral_graph()
        # Build adjacency list: adj[v] = list of neighbors (symmetric)
        n = g_nx.number_of_nodes()
        adj_list = [[] for _ in range(n)]
        for u, v in g_nx.edges():
            adj_list[u].append(v)
            adj_list[v].append(u)
        dg = ice_graph(adj_list, pairing_attempts=100)
        self.assertIsNotNone(dg)
        for node in dg.nodes():
            self.assertLessEqual(dg.in_degree(node), 2)
            self.assertLessEqual(dg.out_degree(node), 2)

    def test_dipole_optimize_traditional_camel_case_args(self):
        """従来通りの camelCase 引数だけで dipole.optimize が動作することを検証する。"""
        paths = [[0, 1, 2], [2, 3, 0]]
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        target = np.zeros(3)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", DeprecationWarning)
            result = optimize(
                paths,
                vertexPositions=positions,
                dipoleOptimizationCycles=5,
                isPeriodicBoundary=False,
                targetPol=target,
            )
        self.assertEqual(len(result), len(paths))
        for r, p in zip(result, paths):
            self.assertEqual(len(r), len(p))
            self.assertEqual(set(r), set(p))

if __name__ == '__main__':
    unittest.main()
