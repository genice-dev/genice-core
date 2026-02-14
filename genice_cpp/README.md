# genice-core C++ 実装

Python の genice_core と同じアルゴリズムを C++ で実装。入出力は配列のみ。

## API

- `ice_graph_edges(n_orig, adj, fixed_out, fixed_in, pairing_attempts)` → `std::vector<std::pair<int,int>>`（解なしは空）
- adj: `vector<vector<int>>`、fixed_out/fixed_in: 長さ n_orig+4

## ビルド

```bash
c++ -std=c++17 -O2 -o genice_core genice_core.cpp
./genice_core
```

## 依存

- C++17、標準ライブラリのみ（乱数: std::random）
