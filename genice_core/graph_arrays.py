"""
Internal graph representation using plain arrays (no NetworkX).

- Undirected: n_nodes (int), adj (list of list of int); adj[v] = neighbors of v.
- Directed: n_orig (int), out_adj and in_adj (list of list of int), length n_orig+4
  to allow dummy nodes -1..-4. Node v maps to index v if v>=0 else n_orig + (-1-v).
"""

from typing import List, Tuple

# Optional import for conversion; topology/dipole do not import nx
try:
    import networkx as nx
except ImportError:
    nx = None


def node_to_idx(v: int, n: int) -> int:
    """Map node id to array index. Real nodes 0..n-1; dummies -1..-4 map to n..n+3."""
    return v if v >= 0 else n + (-1 - v)


def idx_to_node(i: int, n: int) -> int:
    """Map array index back to node id."""
    return i if i < n else -1 - (i - n)


def graph_to_adj(g: "nx.Graph") -> Tuple[int, List[List[int]]]:
    """Convert nx.Graph to (n, adj). Nodes must be 0..n-1."""
    n = g.order()
    adj: List[List[int]] = [[] for _ in range(n)]
    for u, v in g.edges():
        adj[u].append(v)
        adj[v].append(u)
    return n, adj


def adj_to_graph(n: int, adj: List[List[int]]) -> "nx.Graph":
    """Build nx.Graph from (n, adj)."""
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for u in range(n):
        for v in adj[u]:
            if u < v:
                g.add_edge(u, v)
    return g


def digraph_to_arrays(dg: "nx.DiGraph", n_orig: int) -> Tuple[List[List[int]], List[List[int]]]:
    """Convert nx.DiGraph to (out_adj, in_adj). Uses n_orig+4 slots for dummies -1..-4."""
    size = n_orig + 4
    out_adj: List[List[int]] = [[] for _ in range(size)]
    in_adj: List[List[int]] = [[] for _ in range(size)]
    for u, v in dg.edges():
        i, j = node_to_idx(u, n_orig), node_to_idx(v, n_orig)
        out_adj[i].append(v)
        in_adj[j].append(u)
    return out_adj, in_adj


def arrays_to_directed_edges(
    n_orig: int,
    out_adj: List[List[int]],
    in_adj: List[List[int]],
    *,
    include_dummy: bool = False,
) -> List[Tuple[int, int]]:
    """List directed edges (u,v) from out_adj/in_adj. By default only real nodes 0..n_orig-1."""
    edges: List[Tuple[int, int]] = []
    size = len(out_adj)
    for i in range(size):
        u = idx_to_node(i, n_orig)
        if not include_dummy and u < 0:
            continue
        for v in out_adj[i]:
            if include_dummy or (v >= 0 and v < n_orig):
                edges.append((u, v))
    return edges


def edges_to_digraph(n: int, edges: List[Tuple[int, int]]) -> "nx.DiGraph":
    """Build nx.DiGraph with nodes 0..n-1 and given directed edges."""
    dg = nx.DiGraph()
    dg.add_nodes_from(range(n))
    for u, v in edges:
        dg.add_edge(u, v)
    return dg


def connected_components(n_nodes: int, adj: List[List[int]]) -> List[List[int]]:
    """Return list of components; each component is a list of node indices."""
    seen = [False] * n_nodes
    components: List[List[int]] = []

    def bfs(start: int) -> List[int]:
        comp: List[int] = []
        stack = [start]
        seen[start] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for w in adj[v]:
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        return comp

    for v in range(n_nodes):
        if not seen[v]:
            components.append(bfs(v))
    return components
