#!/usr/bin/env python3
"""Nim/Julia/C++ 用の入力データを書き出す。poetry run python test/data/write_data.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import networkx as nx

DATA_DIR = Path(__file__).resolve().parent


def _adj_to_edges(n_orig, adj):
    """隣接リストから辺リスト (u,v) を生成（u < v で1本ずつ）。"""
    edges = []
    for u in range(n_orig):
        for v in adj[u]:
            if u < v:
                edges.append((u, v))
    return edges




def write_graph(n_orig, adj, fixed_edges=None, positions=None, path=None):
    """辺リスト形式で出力（Python/Nim/Julia/C++ でそのまま読める）。
    adj[v] = list of neighbors (0-based). fixed_edges = list of (u,v) or None.
    positions = (n_orig, dim) array or list of (x,y) or (x,y,z); None なら座標ブロックなし。"""
    edges = _adj_to_edges(n_orig, adj)
    lines = [str(n_orig), str(len(edges))]
    for u, v in edges:
        lines.append(f"{u} {v}")
    fixed_edges = fixed_edges or []
    lines.append(str(len(fixed_edges)))
    for u, v in fixed_edges:
        lines.append(f"{u} {v}")
    if positions is not None:
        import numpy as np
        pos = np.asarray(positions)
        if pos.ndim == 1:
            pos = pos.reshape(-1, 1)
        dim = pos.shape[1]
        lines.append(str(dim))
        for v in range(n_orig):
            lines.append(" ".join(f"{pos[v, d]:.10g}" for d in range(dim)))
    out = path or DATA_DIR / "out.txt"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


def main():
    # simple5: 5 nodes, triangle 0-1-2 + edge 3-4
    g5 = nx.Graph([(0, 1), (1, 2), (2, 0), (3, 4)])
    adj5 = [[] for _ in range(5)]
    for u, v in g5.edges():
        adj5[u].append(v)
        adj5[v].append(u)
    pos5 = nx.spring_layout(g5, seed=0)
    positions5 = [pos5[i].tolist() for i in range(5)]  # 2D
    write_graph(5, adj5, positions=positions5, path=DATA_DIR / "simple5.txt")

    # dodecahedral
    g = nx.dodecahedral_graph()
    n = g.order()
    adj = [[] for _ in range(n)]
    for u, v in g.edges():
        adj[u].append(v)
        adj[v].append(u)
    pos = nx.spring_layout(g, seed=0)
    positions = [pos[i].tolist() for i in range(n)]  # 2D
    write_graph(n, adj, positions=positions, path=DATA_DIR / "dodecahedral.txt")

    # PBC (ice 1h): 相対座標（分数座標）付き
    ROOT = DATA_DIR.parent.parent
    ice1h_path = ROOT / "ice1h.txt"
    if not ice1h_path.exists():
        print(f"Skip PBC: {ice1h_path} not found")
    else:
        try:
            import numpy as np
            import pairlist
            pos = np.loadtxt(ice1h_path).reshape(-1, 3)
            cell, coords = pos[0], pos[1:]
            cellmat = np.diag(cell)
            frac_coords = coords / cell  # 相対座標（分数座標）
            g_pbc = nx.Graph(
                [(i, j) for i, j, _ in pairlist.pairs_iter(frac_coords, 0.3, cellmat)]
            )
            n_pbc = g_pbc.order()
            adj_pbc = [[] for _ in range(n_pbc)]
            for u, v in g_pbc.edges():
                adj_pbc[u].append(v)
                adj_pbc[v].append(u)
            positions_pbc = [frac_coords[i].tolist() for i in range(n_pbc)]  # 3D 分数座標
            write_graph(
                n_pbc,
                adj_pbc,
                positions=positions_pbc,
                path=DATA_DIR / "pbc_ice1h.txt",
            )
        except ImportError as e:
            print(f"Skip PBC (pairlist?): {e}")

    # 96x96x96 diamond 格子（example.ipynb と同じ: diamond(N) + pairlist）


if __name__ == "__main__":
    main()
