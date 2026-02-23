# genice-core 3言語実装（Nim / Julia / C++）

Python の `genice_core` と同じアルゴリズムを、Nim・Julia・C++ で実装したものです。  
入出力は配列のみ（NetworkX 非依存）。後でベンチや好みで言語を選べるようにしています。

## 配置

| 言語 | ディレクトリ | エントリ |
|------|--------------|----------|
| **Nim**  | [../genice_nim/](../genice_nim/)  | `genice_core.nim`（`iceGraphEdges`） |
| **Julia**| [../genice_julia/](../genice_julia/) | `genice_core.jl`（`ice_graph_edges`） |
| **C++**  | [../genice_cpp/](../genice_cpp/)  | `genice_core.cpp`（`ice_graph_edges`） |

## 共通 API イメージ

- **入力**: `n_orig`（頂点数）、`adj`（隣接リスト）、`fixed_out` / `fixed_in`（固定有向辺、長さ n_orig+4）
- **出力**: 有向辺のリスト `(tail, head)` の配列。解なしのときは Nim は `none`、Julia は `nothing`、C++ は空 `vector`。

固定辺なしの場合は、`fixed_out` / `fixed_in` を空の配列（Nim/Julia/C++ とも長さ n_orig+4）で渡します。

## ビルド・実行

```bash
# Nim（要: nim インストール）
cd genice_nim && nim c -r genice_core.nim

# Julia
cd genice_julia && julia genice_core.jl

# C++
cd genice_cpp && c++ -std=c++17 -O2 -o genice_core genice_core.cpp && ./genice_core
```

## 注意

- **dipole 最適化**（分極最小化）は今回の 3 言語版には入れていません。必要なら Python の `dipole.optimize` を同じロジックで移植してください。
- **乱数**: 各言語でデフォルトの RNG を使用。再現性が欲しい場合はシードを固定してください。
- **検証**: Python の `ice_graph(..., return_edges=True)` と同じ入力で辺リストを比較すると、アルゴリズムの一致を確認できます。
