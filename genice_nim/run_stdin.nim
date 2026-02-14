## 標準入力から test/data 形式のグラフを読み、有向グラフを標準出力に出す。
## 出力: 1行目 = 有向辺の本数 M、続く M 行が "u v"（tail → head）。解なしは "0" のみ出力して exit 1。

import std/options
import std/strutils
import genice_core

proc main =
  let nOrig = stdin.readLine.parseInt
  let mEdges = stdin.readLine.parseInt
  # 隣接リスト: ice 則は次数≤4 なので長さ4で確保してから詰め、最後に setLen で切り詰める（メモリ削減）
  var adj = newSeq[seq[int]](nOrig)
  for i in 0 ..< nOrig:
    adj[i] = newSeq[int](4)
  var degree = newSeq[int](nOrig)
  for _ in 0 ..< mEdges:
    let line = stdin.readLine.split
    let u = line[0].parseInt
    let v = line[1].parseInt
    adj[u][degree[u]] = v
    degree[u].inc
    adj[v][degree[v]] = u
    degree[v].inc
  for i in 0 ..< nOrig:
    adj[i].setLen(degree[i])

  let nFixed = stdin.readLine.parseInt
  var fixedOut: Adj
  var fixedIn: Adj
  for i in 0 ..< nOrig + 4:
    fixedOut.add newSeq[int]()
    fixedIn.add newSeq[int]()
  for _ in 0 ..< nFixed:
    let line = stdin.readLine.split
    let u = line[0].parseInt
    let v = line[1].parseInt
    fixedOut[u].add v
    fixedIn[v].add u

  # 座標ブロックがあれば読み飛ばす（dipole 未実装のため未使用）
  var line: string
  if readLine(stdin, line):
    discard line.parseInt  # dim
    for _ in 0 ..< nOrig:
      discard stdin.readLine

  let res = iceGraphEdges(nOrig, adj, fixedOut, fixedIn)
  if isNone(res):
    echo "0"
    quit(1)
  let edges = res.get
  echo edges.len
  for (u, v) in edges:
    echo u, " ", v

main()
