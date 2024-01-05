from genice_core import ice_graph
import networkx as nx
from clustice.graph import great_icosahedron


def test_ice_graph():
    g = great_icosahedron(10)

    vertexPositions = [g.nodes[i]["pos"] for i in g]
    dg = ice_graph(
        g,
        vertexPositions=vertexPositions,
        isPeriodicBoundary=False,
        dipoleOptimizationCycles=1000,
        # fixedEdges=nx.DiGraph([(0, 8096)]),
    )


if __name__ == "__main__":
    test_ice_graph()
