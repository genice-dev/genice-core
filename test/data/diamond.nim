## diamond.txt を生成（write_data.py の diamond と同一形式）。
## PairList Nim API 使用。cell=[1,1,1]、pos は分数座標 [0,1)。rc = (√3/4)/N × 1.1。distance は使わない。
##
## ビルド（PairList をルートに。csource を解決するため）:
##   cd ../PairList && nim c -p:nim ../genice-core/test/data/diamond.nim -o:../genice-core/test/data/diamond
## 実行:
##   ./diamond [N=96] [out=diamond.txt]

import std/[math, os, strformat, strutils, syncio]

import pairlist  # pairsFine(pos, rc, cell, fractional=true)

type Vec3 = array[3, float64]

func diamond(N: int): seq[Vec3] =
  ## 単位胞あたり 8 原子。返り値は分数座標 [0,1)。
  const offsets: array[8, array[3, float64]] = [
    [0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.25, 0.25, 0.25],
    [0.25, 0.75, 0.75],
    [0.75, 0.25, 0.75],
    [0.75, 0.75, 0.25],
  ]
  result = newSeq[Vec3](8 * N * N * N)
  var idx = 0
  for i in 0 ..< N:
    for j in 0 ..< N:
      for k in 0 ..< N:
        for o in offsets:
          result[idx] = [
            (float64(i) + o[0]) / float64(N),
            (float64(j) + o[1]) / float64(N),
            (float64(k) + o[2]) / float64(N),
          ]
          idx += 1

proc main =
  let N = if paramCount() >= 1: paramStr(1).parseInt else: 96
  let outFile = if paramCount() >= 2: paramStr(2) else: "diamond.txt"

  let posFrac = diamond(N)
  let nOrig = posFrac.len
  # 分数座標 [0,1)、直交セル N×N×N。第一近接のみ: rc=√3/4×1.05（Cartesian）。
  let cell: pairlist.Mat3 = [
    [float(N), 0.0, 0.0],
    [0.0, float(N), 0.0],
    [0.0, 0.0, float(N)],
  ]
  let rc = sqrt(3.0) / 4.0 * 1.05
  let pairs = pairsFine(posFrac, rc, cell)
  let nEdges = pairs.len

  var lines: seq[string]
  lines.add $nOrig
  lines.add $nEdges
  for (i, j) in pairs:
    lines.add &"{i} {j}"
  lines.add "0"
  lines.add "3"
  for i in 0 ..< nOrig:
    lines.add &"{posFrac[i][0]:.10g} {posFrac[i][1]:.10g} {posFrac[i][2]:.10g}"

  let content = lines.join("\n") & "\n"
  writeFile(outFile, content)
  echo &"Wrote {outFile}: {nOrig} nodes, {nEdges} edges (N={N})"

main()
