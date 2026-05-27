"""Connect matching paths by MCF (minimum-cost flow)."""

from logging import getLogger
from typing import List, Tuple, Any, Dict

import networkx as nx


def connect_matching_paths_mcf(
    fixed: nx.DiGraph, g: nx.Graph
) -> Tuple[nx.DiGraph | None, List[List[int]]]:
    """MCF-based connect_engine for genice_core.ice_graph.

    This is a best-effort replacement of genice_core's
    connect_matching_paths_nx. It constructs a min-cost flow that routes
    from out_peri to in_peri using edge-disjoint (undirected) free O-O bonds.
    """
    logger = getLogger(__name__)

    _fixed = nx.DiGraph(fixed)

    # Approximate perimeters like connect_matching_paths_nx.
    in_peri: set[int] = set()
    out_peri: set[int] = set()
    for node in _fixed.nodes():
        if _fixed.in_degree(node) + _fixed.out_degree(node) >= g.degree(node):
            continue
        if _fixed.in_degree(node) > _fixed.out_degree(node):
            out_peri.add(int(node))
        elif _fixed.in_degree(node) < _fixed.out_degree(node):
            in_peri.add(int(node))

    if not out_peri and not in_peri:
        return _fixed, []

    if len(out_peri) != len(in_peri):
        return None, []

    sources = sorted(out_peri)
    sinks = set(in_peri)

    # Free undirected edges: neither direction is fixed.
    free_edges: List[Tuple[int, int]] = []
    for u, v in g.edges():
        u = int(u)
        v = int(v)
        if _fixed.has_edge(u, v) or _fixed.has_edge(v, u):
            continue
        free_edges.append((u, v))

    if not free_edges:
        return None, []

    # Restrict to nodes in connected components of sources within free-edge subgraph.
    free_graph = nx.Graph()
    free_graph.add_nodes_from(g.nodes())
    free_graph.add_edges_from(free_edges)
    relevant: set[int] = set()
    for s in sources:
        if s in free_graph:
            try:
                relevant |= nx.node_connected_component(free_graph, s)
            except Exception:
                relevant.add(s)

    relevant_edges = [(u, v) for (u, v) in free_edges if u in relevant and v in relevant]
    if not relevant_edges:
        return None, []

    # Build min-cost flow network with an undirected-edge "resource" gadget.
    # Each free undirected edge {a,b} becomes:
    #   a -> mid_in -> mid_out -> b   (and symmetrically b -> mid_in -> mid_out -> a)
    # using capacity=1 on mid_in->mid_out to prevent overlap.
    INF_CAP = 10**9
    cost_per_resource = 1
    flow_net = nx.DiGraph()

    # demands: negative = send, positive = receive (min_cost_flow convention)
    demand: Dict[int, int] = {}
    for s in sources:
        demand[int(s)] = demand.get(int(s), 0) - 1
    for t in sinks:
        demand[int(t)] = demand.get(int(t), 0) + 1
    if abs(sum(demand.values())) > 1e-10:
        return None, []

    for u, v in relevant_edges:
        a, b = (u, v) if u <= v else (v, u)
        mid_in = ("e_in", a, b)
        mid_out = ("e_out", a, b)

        # resource edge (capacity 1)
        flow_net.add_edge(mid_in, mid_out, capacity=1, weight=cost_per_resource)

        # traverse from either endpoint into resource, and leave to either endpoint
        for x in (a, b):
            flow_net.add_edge(x, mid_in, capacity=INF_CAP, weight=0)
            flow_net.add_edge(mid_out, x, capacity=INF_CAP, weight=0)

    for n in flow_net.nodes():
        flow_net.nodes[n]["demand"] = int(demand.get(n, 0))  # type: ignore[arg-type]

    try:
        flow_dict = nx.min_cost_flow(flow_net)
    except Exception as e:
        logger.debug("MCF connect_engine failed: %s", e)
        return None, []

    # Decompose remaining flow into directed paths. We ignore derived_cycles.
    rem: Dict[Any, Dict[Any, int]] = {}
    for u, nbrs in flow_dict.items():
        rem[u] = {v: int(f) for v, f in nbrs.items() if f}

    processed = nx.DiGraph(_fixed)

    def next_pos_out(cur: Any) -> Any | None:
        for nxt, f in rem.get(cur, {}).items():
            if f > 0:
                return nxt
        return None

    # Decompose integral flow into directed paths.
    # Each source may send multiple units in degenerate cases; we consume
    # all remaining outgoing flow from each source greedily.
    for s in sources:
        source = int(s)
        if source in sinks:
            continue
        while next_pos_out(source) is not None:
            cur: Any = source
            prev_original: int = source
            steps = 0
            while int(cur) not in sinks:
                steps += 1
                if steps > 20_000:
                    return None, []

                mid_in = next_pos_out(cur)
                if mid_in is None:
                    return None, []
                rem[cur][mid_in] -= 1

                mid_out = next_pos_out(mid_in)
                if mid_out is None:
                    return None, []
                rem[mid_in][mid_out] -= 1

                nxt_original = next_pos_out(mid_out)
                if nxt_original is None:
                    return None, []
                rem[mid_out][nxt_original] -= 1

                u = int(prev_original)
                v = int(nxt_original)
                processed.add_edge(u, v)
                prev_original = v
                cur = v

    return processed, []
