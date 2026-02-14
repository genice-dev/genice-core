# genice-core Nim 実装

Python の genice_core と同じアルゴリズムを Nim で実装。入出力は配列のみ（NetworkX 非依存）。

## API

- `iceGraphEdges(nOrig, adj, fixedOut, fixedIn, ...)` → `seq[(int,int)]`（有向辺リスト）
- 固定辺なしの場合は `fixedOut`/`fixedIn` は空の seq（サイズ n_orig+4）

## ビルド・実行

```bash
nim c -r genice_core.nim
```

## 依存

- 標準ライブラリのみ（random 使用）
