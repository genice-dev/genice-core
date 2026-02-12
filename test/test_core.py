from genice_core import ice_graph
import networkx as nx
from clustice.graph import great_icosahedron



def test_ice_graph():
    g = great_icosahedron(10)

    vertex_positions = [g.nodes[i]["pos"] for i in g]
    dg = ice_graph(
        g,
        vertex_positions=vertex_positions,
        is_periodic_boundary=False,
        dipole_optimization_cycles=1000,
        # fixed_edges=nx.DiGraph([(0, 8096)]),
    )


if __name__ == "__main__":
    test_ice_graph()
