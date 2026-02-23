# connect_matching_paths_bfs への改良案（あなたの BFS 実装のアイデアを取り込む）

## 取り込めるアイデア

| アイデア                 | 内容                                                                       | ラウンド制 BFS への入れ方                                                                                      |
| ------------------------ | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **undet_peri**           | 次数 1 or 3 のノードを「未決定の端点」として扱う                           | 初期に undet_peri を計算し、push/pull でそのノードに着地したら undet_peri から除去する                         |
| **シンクを優先した一歩** | 空き辺を選ぶとき「シンク（in_peri / out_peri）に直接つながる辺」を優先する | `_choose_free_edge` に optional の `preferred` を渡し、候補の先頭に preferred な隣接を置いてからシャッフルする |
| 最短経路で 1 本ずつ      | BFS で最短経路を 1 本決めてから次へ                                        | ラウンド制とは別の戦略のため、今回は「一歩の選び方の優先」のみ取り込む                                         |

※ 多重度（defaultdict(int)）は、現状の「まだ不均衡なら next_frontier に再追加」で既に表現できているのでそのままでよい。

---

## 1. \_choose_free_edge の変更（preferred 対応）

配列版の `_choose_free_edge` に **optional** で「優先して試したい隣接ノードの集合」を渡す。

```python
def _choose_free_edge(
    n_orig: int,
    adj: List[List[int]],
    out_adj: List[List[int]],
    in_adj: List[List[int]],
    node: int,
    preferred: Optional[Set[int]] = None,
) -> Optional[int]:
    """Find an unfixed edge of the node. If preferred is set, try those neighbors first."""
    neis = (list(adj[node]) + [-1, -2, -3, -4])[:4]
    if preferred:
        # 優先ターゲット（シンクなど）に直接つながる辺を先に試す
        pref = [n for n in neis if n is not None and n >= 0 and n in preferred]
        rest = [n for n in neis if n not in pref]
        neis = pref + rest
    np.random.shuffle(neis)
    for nei in neis:
        if nei is None:
            continue
        i_node = node_to_idx(node, n_orig)
        i_nei = node_to_idx(nei, n_orig) if nei >= 0 else n_orig + (-1 - nei)
        has_out = nei in out_adj[i_node]
        has_in = node in out_adj[i_nei]
        if not (has_out or has_in):
            return nei
    return None
```

- **push_round** では `preferred=in_peri | undet_peri` を渡す（一歩で in_peri/undet に着地しやすくする）。
- **pull_round** では `preferred=out_peri | undet_peri` を渡す。
- 既存の `connect_matching_paths` からは `preferred=None` で呼べば従来どおり。

---

## 2. undet_peri の導入（connect_matching_paths_bfs 内）

- **定義**: 固定辺がまだ 0 本で、かつ元グラフの次数が 1 または 3 のノードを「未決定の端点」とする。
- **更新**: push で `next_node` が undet_peri に入っていれば `undet_peri.discard(next_node)`。pull でも同様に、着地したノードを undet_peri から除去する。

初期化（`_get_perimeters` の直後など）:

```python
# 次数 1 or 3 で、まだ固定辺が 0 のノードを未決定端点とする
undet_peri: Set[int] = set()
for v in range(n_orig):
    if len(adj[v]) in (1, 3):
        if in_degree(v) == 0 and out_degree(v) == 0:
            undet_peri.add(v)
```

push_round 内で `_choose_free_edge(..., preferred=in_peri | undet_peri)` を渡し、辺を追加したあと:

```python
if next_node >= 0:
    if next_node in in_peri:
        in_peri.discard(next_node)
    if next_node in undet_peri:
        undet_peri.discard(next_node)
    if in_degree(next_node) > out_degree(next_node):
        next_out.add(next_node)
```

pull_round 内も同様に、`preferred=out_peri | undet_peri` と、着地したノードを `in_peri` / `undet_peri` から除去する処理を追加。

---

## 3. まとめ

- **undet_peri**: 次数 1/3 の「端点候補」を明示し、そこに着地したら未決定集合から外す。あなたの実装と同じ考え方。
- **preferred**: 一歩でシンク（in_peri / out_peri）や undet_peri に直接つながる辺を優先する。最短でつなぐ効果は、ラウンド制のままでも一部得られる。
- 既存の `connect_matching_paths` や他呼び出しは、`_choose_free_edge` の `preferred=None` で従来どおり。

配列版のブランチに戻したら、上記の 1 と 2 を `topology.py` に適用すれば、あなたの BFS 実装のアイデアを取り込んだ改良版になります。
