# genice-core Julia 実装

Python の genice_core と同じアルゴリズムを Julia で実装。入出力は配列のみ。

## API

- `ice_graph_edges(n_orig, adj, fixed_out, fixed_in; pairing_attempts=100)` → `Vector{Tuple{Int,Int}}` または `nothing`
- 配列は 1-based（Julia 慣習）。adj[i] = ノード i-1 の隣接リスト（0-based ノード番号）

## 実行

```bash
julia genice_core.jl
```

## 依存

- 標準ライブラリのみ（Random 使用）
