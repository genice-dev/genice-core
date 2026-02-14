# 入力データ（Nim / Julia / C++ 用）

3言語で同じように読めるプレーンテキスト形式。

## フォーマット（辺リスト形式）

Python の `ice_graph(edges, ...)` や Nim/Julia/C++ で同じファイルを読めるように、**辺リスト**で統一。

```
n_orig
m_edges
u1 v1
u2 v2
...
n_fixed
<固定有向辺 u v が n_fixed 行>
...
[オプション] 座標ブロック:
dim          （2 または 3）
<行1> x0 y0 [z0]
...
<行n_orig> x_{n_orig-1} y [z]
```

- 無向辺は `u v` の1行（u < v で1本ずつ）。コメント行はなし。
- 座標は `write_data.py` で付与（2D は spring_layout、PBC は分数座標 3D）。dim 以降がなければ「座標なし」。

## ファイル

| ファイル | 内容 |
|----------|------|
| simple5.txt | 5頂点（三角形+辺）。固定辺なし。座標は 2D。 |
| dodecahedral.txt | 12面体グラフ（20頂点、30辺）。固定辺なし。座標は 2D。 |
| pbc_ice1h.txt | PBC グラフ（ice1h.txt + pairlist、0.3 カットオフ）。相対座標（分数座標）3D 付き。ice1h.txt と pairlist があるときだけ生成。 |

## 読み方の例（疑似コード）

```
n_orig = 先頭行を整数で読む
m_edges = 次の行を整数で読む
edges = []
for k in 0..m_edges-1:
  (u, v) = 次の行を2整数で読む
  edges.append((u, v))
# 必要ならここで edges から隣接リスト adj を構築
n_fixed = 次の行を整数で読む
for k in 0..n_fixed-1:
  (u, v) = 次の行を2整数で読む  # 固定有向辺
# 座標が必要なら:
if まだ行がある:
  dim = 次の行を整数で読む
  for i in 0..n_orig-1:
    positions[i] = 次の行を dim 個の浮動小数で読む
```

## 再生成（任意）

同じデータを Python で再生成する場合（プロジェクトルートで）:

```bash
poetry run python test/data/write_data.py
```
