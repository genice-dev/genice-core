#!/usr/bin/env python3
"""標準入力から test/data 形式のグラフを読み、有向グラフを標準出力に出す。
出力: 1行目 = 有向辺の本数 M、続く M 行が "u v"（tail → head）。解なしは "0" のみ出力して exit 1。
"""

import sys

from genice_core import ice_graph
import networkx as nx


def main():
    n_orig = int(sys.stdin.readline())
    m_edges = int(sys.stdin.readline())
    # 辺リストを保持せず隣接リストだけ構築（大規模でメモリ削減）
    adj = [[] for _ in range(n_orig)]
    for _ in range(m_edges):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)

    n_fixed = int(sys.stdin.readline())
    fixed = nx.DiGraph()
    for _ in range(n_fixed):
        u, v = map(int, sys.stdin.readline().split())
        fixed.add_edge(u, v)

    # 座標ブロックがあれば読む（dipole 用）
    positions = None
    line = sys.stdin.readline()
    if line:
        dim = int(line)
        positions = []
        for _ in range(n_orig):
            positions.append([float(x) for x in sys.stdin.readline().split()])

    result = ice_graph(
        adj,
        fixed_edges=fixed,
        vertex_positions=positions,
        return_edges=True,
        g_format="adjacency",
    )
    if result is None:
        print("0")
        sys.exit(1)
    print(len(result))
    for u, v in result:
        print(u, v)


if __name__ == "__main__":
    main()
