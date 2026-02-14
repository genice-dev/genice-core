## genice-core: ice rule を満たす有向グラフを生成（Nim 実装）
## 入出力は配列のみ。API: iceGraphEdges(nOrig, adj, fixedOut, fixedIn, ...) -> seq[(int,int)]

import std/algorithm
import std/options
import std/random
import std/sequtils
import std/sets
import std/tables

type
  Adj* = seq[seq[int]]
  DirectedEdges* = seq[(int, int)]

# ---------- graph_arrays 相当 ----------
func nodeToIdx(v, n: int): int = (if v >= 0: v else: n + (-1 - v))
func idxToNode(i, n: int): int = (if i < n: i else: -1 - (i - n))

func connectedComponents(nNodes: int; adj: Adj): seq[seq[int]] =
  var seen = newSeq[bool](nNodes)
  for v in 0 ..< nNodes:
    if not seen[v]:
      var comp: seq[int]
      var stack = @[v]
      seen[v] = true
      while stack.len > 0:
        let cur = stack.pop
        comp.add cur
        for w in adj[cur]:
          if not seen[w]:
            seen[w] = true
            stack.add w
      result.add comp

# ---------- topology: trace_path, find_path ----------
func tracePath(nNodes: int; adj: Adj; path: var seq[int]; vertexSet: HashSet[int]): void =
  let vs = vertexSet
  while true:
    let last = path[^2]
    let head = path[^1]
    var nextNode = -1
    for w in adj[head]:
      if w in vs and w != last:
        nextNode = w
        break
    if nextNode < 0:
      return
    path.add nextNode
    if nextNode == path[0]:
      return

func findPath(nNodes: int; adj: Adj; vertexSet: seq[int]): seq[int] =
  if vertexSet.len == 0:
    return @[]
  let vs = toHashSet(vertexSet)
  let head = vertexSet[0]
  var neighbors: seq[int]
  for w in adj[head]:
    if w in vs:
      neighbors.add w
  if neighbors.len == 0:
    return @[]
  if neighbors.len == 1:
    var path = @[head, neighbors[0]]
    tracePath(nNodes, adj, path, vs)
    return path
  var c0 = @[head, neighbors[0]]
  tracePath(nNodes, adj, c0, vs)
  if c0[^1] == head:
    return c0
  var c1 = @[head, neighbors[1]]
  tracePath(nNodes, adj, c1, vs)
  result = reversed(c0) & c1[1..^1]

# ---------- topology: _divide, noodlize ----------
proc divide(nNodes: int; adj: var Adj; vertex, offset: int) =
  var nei: seq[int]
  for x in adj[vertex]:
    nei.add x
  while nei.len < 4:
    nei.add -1
  nei.setLen(4)
  var valid: seq[int]
  for x in nei:
    if x >= 0:
      valid.add x
  if valid.len < 2:
    return
  shuffle(valid)
  let migrants = [valid[0], valid[1]]
  let newVertex = vertex + offset
  for m in migrants:
    var p = adj[m].find(vertex)
    if p >= 0:
      adj[m].delete(p)
    p = adj[vertex].find(m)
    if p >= 0:
      adj[vertex].delete(p)
    adj[newVertex].add m
    adj[m].add newVertex

## noodlize: グラフを 2*nOrig ノードに拡張するため、ピーク時は元の adj と adjNoodles の両方が生きる（8M 頂点で 2〜3 GB 程度になる要因）
proc noodlize(nOrig: int; adj: Adj; fixedOut, fixedIn: Adj): (int, Adj) =
  let nNodes = 2 * nOrig
  var adjNoodles: Adj
  for v in 0 ..< nOrig:
    adjNoodles.add adj[v].mapIt(it)
  for v in 0 ..< nOrig:
    adjNoodles.add newSeq[int]()
  for u in 0 ..< nOrig:
    let iu = nodeToIdx(u, nOrig)
    for v in fixedOut[iu]:
      if v >= 0 and v < nOrig and v in adjNoodles[u]:
        var p = adjNoodles[u].find(v)
        adjNoodles[u].delete(p)
        p = adjNoodles[v].find(u)
        adjNoodles[v].delete(p)
  for v in 0 ..< nOrig:
    let nfixed = fixedOut[nodeToIdx(v, nOrig)].len + fixedIn[nodeToIdx(v, nOrig)].len
    if nfixed == 0:
      divide(nNodes, adjNoodles, v, nOrig)
  return (nNodes, adjNoodles)

# ---------- topology: _decompose_complex_path ----------
func decomposeComplexPath(path: seq[int]): seq[seq[int]] =
  if path.len == 0:
    return @[]
  var order: Table[int, int]
  var store: seq[int]
  order[path[0]] = 0
  store.add path[0]
  var headp = 1
  while headp < path.len:
    let node = path[headp]
    if node in order:
      let size = order.len - order[node]
      var cycle: seq[int]
      for i in (store.len - size) ..< store.len:
        cycle.add store[i]
      cycle.add node
      result.add cycle
      for v in cycle[1..^1]:
        order.del(v)
      store.setLen(store.len - size)
    order[node] = order.len
    store.add node
    headp += 1
  if store.len > 1:
    result.add store

# ---------- topology: split_into_simple_paths ----------
proc splitIntoSimplePaths(nOrig, nNodes: int; adj: Adj): seq[seq[int]] =
  let components = connectedComponents(nNodes, adj)
  for vertexSet in components:
    let path = findPath(nNodes, adj, vertexSet)
    var flatten: seq[int]
    for v in path:
      flatten.add v mod nOrig
    for p in decomposeComplexPath(flatten):
      result.add p

# ---------- topology: connect_matching_paths 用ヘルパ ----------
func copyDirected(nOrig: int; outAdj, inAdj: Adj): (Adj, Adj) =
  let size = nOrig + 4
  var oa, ia: Adj
  for i in 0 ..< size:
    oa.add outAdj[i].mapIt(it)
    ia.add inAdj[i].mapIt(it)
  return (oa, ia)

func getPerimeters(nOrig: int; adj, outAdj, inAdj: Adj): (HashSet[int], HashSet[int]) =
  var inPeri, outPeri: HashSet[int]
  for i in 0 ..< nOrig:
    let degG = adj[i].len
    let inD = inAdj[i].len
    let outD = outAdj[i].len
    if inD + outD >= degG:
      continue
    if inD > outD:
      outPeri.incl i
    elif inD < outD:
      inPeri.incl i
  return (inPeri, outPeri)

proc chooseFreeEdge(nOrig: int; adj, outAdj, inAdj: Adj; node: int): int =
  var neis: seq[int]
  for x in adj[node]:
    neis.add x
  while neis.len < 4:
    neis.add -1 - neis.len
  neis.setLen(4)
  shuffle(neis)
  for nei in neis:
    if nei == -1:
      continue
    let iNode = nodeToIdx(node, nOrig)
    let iNei = if nei >= 0: nodeToIdx(nei, nOrig) else: nOrig + (-1 - nei)
    let hasOut = nei in outAdj[iNode]
    let hasIn = node in outAdj[iNei]
    if not (hasOut or hasIn):
      return nei
  return -999

proc removeDummyNodes(nOrig: int; outAdj, inAdj: var Adj) =
  let size = nOrig + 4
  for i in nOrig ..< size:
    outAdj[i].setLen(0)
    inAdj[i].setLen(0)
  for v in 0 ..< nOrig:
    outAdj[v] = outAdj[v].filterIt(it >= 0 and it < nOrig)
    inAdj[v] = inAdj[v].filterIt(it >= 0 and it < nOrig)

proc connectMatchingPaths(nOrig: int; adj, fixedOut, fixedIn: Adj): Option[(Adj, Adj, seq[seq[int]])] =
  var (fOut, fIn) = copyDirected(nOrig, fixedOut, fixedIn)
  var (inPeri, outPeri) = getPerimeters(nOrig, adj, fOut, fIn)
  var derivedCycles: seq[seq[int]]

  proc addEdge(u, v: int) =
    let iu = nodeToIdx(u, nOrig)
    let iv = nodeToIdx(v, nOrig)
    fOut[iu].add v
    fIn[iv].add u

  func inDegree(node: int): int = fIn[nodeToIdx(node, nOrig)].len
  func outDegree(node: int): int = fOut[nodeToIdx(node, nOrig)].len

  while outPeri.len > 0:
    var node = 0
    for n in outPeri:
      node = n
      break
    outPeri.excl node
    var path = @[node]
    var cur = node
    block innerOut:
      while true:
        if cur < 0:
          break innerOut
        if cur in inPeri:
          inPeri.excl cur
          break innerOut
        if outDegree(cur) * 2 > 4:
          return options.none[(Adj, Adj, seq[seq[int]])]()
        if adj[cur].len == outDegree(cur) + inDegree(cur):
          return options.none[(Adj, Adj, seq[seq[int]])]()
        let nextNode = chooseFreeEdge(nOrig, adj, fOut, fIn, cur)
        if nextNode == -999:
          return options.none[(Adj, Adj, seq[seq[int]])]()
        addEdge(cur, nextNode)
        if nextNode >= 0:
          path.add nextNode
          if inDegree(cur) > outDegree(cur):
            outPeri.incl cur
        cur = nextNode
        for i in 0 ..< path.len - 1:
          if path[i] == cur:
            derivedCycles.add path[i..^1]
            path.setLen(i + 1)
            break

  while inPeri.len > 0:
    var node = 0
    for n in inPeri:
      node = n
      break
    inPeri.excl node
    var path = @[node]
    var cur = node
    block innerIn:
      while true:
        if cur < 0:
          break innerIn
        if cur in outPeri:
          outPeri.excl cur
          break innerIn
        if outDegree(cur) * 2 > 4:
          return options.none[(Adj, Adj, seq[seq[int]])]()
        if adj[cur].len == outDegree(cur) + inDegree(cur):
          return options.none[(Adj, Adj, seq[seq[int]])]()
        let nextNode = chooseFreeEdge(nOrig, adj, fOut, fIn, cur)
        if nextNode == -999:
          return options.none[(Adj, Adj, seq[seq[int]])]()
        if nextNode >= 0:
          path.add nextNode
        addEdge(nextNode, cur)
        if nextNode >= 0 and inDegree(cur) < outDegree(cur):
          inPeri.incl cur
        cur = nextNode
        for i in 0 ..< path.len - 1:
          if path[i] == cur:
            derivedCycles.add path[i..^1]
            path.setLen(i + 1)
            break

  removeDummyNodes(nOrig, fOut, fIn)
  return some((fOut, fIn, derivedCycles))

# ---------- directed edges from arrays ----------
func arraysToDirectedEdges(nOrig: int; outAdj, inAdj: Adj): DirectedEdges =
  for i in 0 ..< outAdj.len:
    let u = idxToNode(i, nOrig)
    if u < 0:
      continue
    for v in outAdj[i]:
      if v >= 0 and v < nOrig:
        result.add (u, v)

func pathEdges(paths: seq[seq[int]]): DirectedEdges =
  for path in paths:
    for i in 0 ..< path.len - 1:
      result.add (path[i], path[i+1])

func finallyFixedEdges(nOrig: int; fixedOut, fixedIn: Adj; derivedCycles: seq[seq[int]]): DirectedEdges =
  let edges = arraysToDirectedEdges(nOrig, fixedOut, fixedIn)
  var cycleSet: HashSet[(int,int)]
  for cycle in derivedCycles:
    for i in 0 ..< cycle.len - 1:
      cycleSet.incl (cycle[i], cycle[i+1])
  for (u, v) in edges:
    if (u, v) notin cycleSet:
      result.add (u, v)

# ---------- dipole (省略版: positions なしなら optimize は identity) ----------
# ここでは dipole 最適化は未実装とする。paths をそのまま使う。

# ---------- main: iceGraphEdges ----------
proc iceGraphEdges*(
  nOrig: int,
  adj: Adj,
  fixedOut, fixedIn: Adj,
  pairingAttempts: int = 100,
): Option[DirectedEdges] =
  ## 固定辺なしの場合は fixedOut/fixedIn は size nOrig+4 の空 seq を渡す。
  var fOut = fixedOut
  var fIn = fixedIn
  var derivedCycles: seq[seq[int]] = @[]

  let nFixed = block:
    var c = 0
    for i in 0 ..< fOut.len:
      c += fOut[i].len
    c

  if nFixed > 0:
    block tryPairing:
      for attempt in 0 ..< pairingAttempts:
        let res = connectMatchingPaths(nOrig, adj, fOut, fIn)
        if res.isSome:
          let (o, i, dc) = res.get
          fOut = o
          fIn = i
          derivedCycles = dc
          break tryPairing
      return options.none[DirectedEdges]()

  let finFixed = finallyFixedEdges(nOrig, fOut, fIn, derivedCycles)
  let (nNodes, adjNoodles) = noodlize(nOrig, adj, fOut, fIn)
  var paths = splitIntoSimplePaths(nOrig, nNodes, adjNoodles)
  for c in derivedCycles:
    paths.add c

  var allEdges = finFixed
  allEdges.add pathEdges(paths)
  return some(allEdges)

# ---------- 使用例 ----------
when isMainModule:
  randomize()
  let adj: Adj = @[
    @[1, 2],
    @[0, 2],
    @[0, 1],
    @[4],
    @[3],
  ]
  let nOrig = 5
  var fixedOut: Adj
  for i in 0 ..< nOrig + 4:
    fixedOut.add newSeq[int]()
  var fixedIn: Adj
  for i in 0 ..< nOrig + 4:
    fixedIn.add newSeq[int]()

  let res = iceGraphEdges(nOrig, adj, fixedOut, fixedIn)
  if res.isSome:
    echo "edges: ", res.get.len
    echo "sample: ", res.get[0..min(4, res.get.len-1)]
  else:
    echo "no solution"
