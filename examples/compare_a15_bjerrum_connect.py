#!/usr/bin/env python3
"""
GenIce3 A15 2×2×2 超胞に Bjerrum D/L 欠陥対を置き、connect / ice_graph の成功率を比較する。

ループ構造:
  外側: connect_engine（MCF → 従来ランダム）
  内側: 欠陥濃度（水分子数に対する欠陥サイト割合 %%）

欠陥濃度: 水分子数 N に対し欠陥サイト数 ≈ N * (percent/100)、
D/L が同数になるよう D/L 対は約 N * percent / 200 組。

``ice_graph`` 失敗の例（いずれもベンチでは失敗として数え、中断しない）:
  - connect_in_ice_graph … connect_engine が None
  - ice_rules … 氷規則違反 (AssertionError)
  - no_path_polarize … force_polarize 内で D–L 間に有向経路なし
  - networkx … その他 NetworkX エラー

依存: genice3, genice-core

例:
  python examples/compare_a15_bjerrum_connect.py
  python examples/compare_a15_bjerrum_connect.py --percents 1 2 3 5 8 10 --trials 20
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from logging import WARNING, basicConfig, getLogger
from pathlib import Path
from typing import Callable, DefaultDict, List, NamedTuple, Optional, Sequence, Tuple

# Allow running as ``python examples/compare_a15_bjerrum_connect.py`` without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import networkx as nx
import numpy as np

import genice_core
from genice_core.topology_nx import (
    connect_matching_paths,
    connect_matching_paths_mcf,
)
from genice3.genice import GenIce3
from genice3.util import find_nearest_edges_pbc

ConnectFn = Callable[[nx.DiGraph, nx.Graph], Tuple[Optional[nx.DiGraph], list]]


class ConnectMethod(NamedTuple):
    label: str
    fn: ConnectFn
    pairing_attempts: int


@dataclass
class ConcentrationStats:
    percent: float
    n_pairs: int
    placed: int
    connect_ok: int
    ice_ok: int
    ice_fail_reasons: DefaultDict[str, int] = field(default_factory=defaultdict)


def _n_pairs(n_water: int, percent: float) -> int:
    """D/L 対の本数（欠陥サイト 2 個 = 1 対）。"""
    n_defect_sites = n_water * percent / 100.0
    return max(1, int(round(n_defect_sites / 2.0)))


def _min_pair_distance(graph: nx.Graph, sites: Sequence[int]) -> int:
    if len(sites) < 2:
        return 10**9
    dmin = 10**9
    for i, a in enumerate(sites):
        for b in sites[i + 1 :]:
            try:
                d = nx.shortest_path_length(graph, a, b)
            except nx.NetworkXNoPath:
                d = 10**9
            dmin = min(dmin, d)
    return dmin


def place_defects_by_sites(
    genice: GenIce3,
    graph: nx.Graph,
    n_pairs: int,
    rng: np.random.Generator,
    min_graph_distance: int,
    max_tries: int = 500,
) -> bool:
    n = graph.number_of_nodes()
    need = 2 * n_pairs
    if need > n:
        return False

    for _ in range(max_tries):
        candidates = rng.choice(n, size=need, replace=False)
        d_sites = list(candidates[:n_pairs])
        l_sites = list(candidates[n_pairs:])
        if _min_pair_distance(graph, d_sites + l_sites) < min_graph_distance:
            continue

        d_edges: List[Tuple[int, int]] = []
        l_edges: List[Tuple[int, int]] = []
        for i in d_sites:
            j = int(rng.choice(list(graph.neighbors(i))))
            d_edges.append((i, j))
        for i in l_sites:
            j = int(rng.choice(list(graph.neighbors(i))))
            l_edges.append((i, j))

        genice.bjerrum_D_edges = []
        genice.bjerrum_L_edges = []
        try:
            genice.bjerrum_D_edges = d_edges
            genice.bjerrum_L_edges = l_edges
            _ = genice.fixed_edges
        except Exception:
            continue
        return True
    return False


def place_defects_by_fractional_positions(
    genice: GenIce3,
    graph: nx.Graph,
    lattice_sites: np.ndarray,
    cell: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    min_graph_distance: int,
    max_tries: int = 500,
) -> bool:
    for _ in range(max_tries):
        d_edges: List[Tuple[int, int]] = []
        l_edges: List[Tuple[int, int]] = []
        used: set[int] = set()
        ok = True
        for _pair in range(n_pairs):
            placed = False
            for _ in range(200):
                de = find_nearest_edges_pbc(
                    rng.random(3), graph, lattice_sites, cell
                )
                le = find_nearest_edges_pbc(
                    rng.random(3), graph, lattice_sites, cell
                )
                if de[0] in used or le[0] in used or de[0] == le[0]:
                    continue
                used.add(de[0])
                used.add(le[0])
                d_edges.append(de)
                l_edges.append(le)
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue
        sites = [e[0] for e in d_edges] + [e[0] for e in l_edges]
        if _min_pair_distance(graph, sites) < min_graph_distance:
            continue
        genice.bjerrum_D_edges = []
        genice.bjerrum_L_edges = []
        try:
            genice.bjerrum_D_edges = d_edges
            genice.bjerrum_L_edges = l_edges
            _ = genice.fixed_edges
        except Exception:
            continue
        return True
    return False


def _try_connect(
    method: ConnectMethod,
    fixed: nx.DiGraph,
    graph: nx.Graph,
) -> bool:
    for _ in range(method.pairing_attempts):
        result, _ = method.fn(fixed, graph)
        if result is not None:
            return True
    return False


def _try_ice_graph(
    method: ConnectMethod,
    graph: nx.Graph,
    lattice_sites: np.ndarray,
    fixed: nx.DiGraph,
    seed: int,
) -> Tuple[bool, Optional[str]]:
    """Return (success, failure_reason).

    ``ice_graph`` は connect 成功後も、分極・``force_polarize``・検証で落ちることがある。
    ベンチでは例外や ``None`` をすべて失敗として数え、理由ラベルを返す。
    """
    try:
        dg = genice_core.ice_graph(
            graph,
            vertex_positions=lattice_sites,
            is_periodic_boundary=True,
            dipole_optimization_cycles=200,
            dipole_optimization_cycles2=200,
            fixed_edges=fixed,
            pairing_attempts=method.pairing_attempts,
            target_pol=np.zeros(3),
            connect_engine=method.fn,
            seed=seed,
        )
    except AssertionError:
        return False, "ice_rules"
    except nx.NetworkXNoPath:
        return False, "no_path_polarize"
    except nx.NetworkXError:
        return False, "networkx"
    except Exception as exc:
        return False, type(exc).__name__
    if dg is None:
        return False, "connect_in_ice_graph"
    return True, None


def _configure_quiet_logging() -> None:
    """GenIce3 / genice_core / dependency_engine の INFO を抑える。"""
    basicConfig(level=WARNING, force=True)
    for name in (
        "",
        "GenIce3",
        "genice3",
        "genice_core",
        "dependency_engine",
        "replicate_fixed_edges",
    ):
        getLogger(name).setLevel(WARNING)


def _format_ice_failures(reasons: DefaultDict[str, int]) -> str:
    if not reasons:
        return ""
    parts = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
    return f"  [ice 失敗内訳: {parts}]"


def run_benchmark(
    percents: Sequence[float],
    trials: int,
    pairing_attempts_random: int,
    placement: str,
    min_graph_distance: int,
    base_seed: int,
    ice_graph_test: bool,
) -> None:
    methods = (
        ConnectMethod("MCF", connect_matching_paths_mcf, 1),
        ConnectMethod(
            "従来 (random)",
            connect_matching_paths,
            pairing_attempts_random,
        ),
    )

    genice = GenIce3(replication_matrix=np.diag([2, 2, 2]).astype(int))
    genice.set_unitcell("A15")
    genice.seed = base_seed

    graph = genice.graph
    lattice_sites = genice.lattice_sites
    cell = genice.cell
    n_water = graph.number_of_nodes()

    print(f"A15 2×2×2: N={n_water} waters, {graph.number_of_edges()} H-bonds")
    print(
        f"trials={trials}, placement={placement}, "
        f"min_graph_distance={min_graph_distance}, "
        f"random pairing_attempts={pairing_attempts_random}"
    )
    print()

    def place(n_pairs: int, rng: np.random.Generator) -> bool:
        if placement == "sites":
            return place_defects_by_sites(
                genice, graph, n_pairs, rng, min_graph_distance
            )
        return place_defects_by_fractional_positions(
            genice, graph, lattice_sites, cell, n_pairs, rng, min_graph_distance
        )

    # 外側: connect 手法（MCF → 従来）
    for method in methods:
        print(f"======== {method.label} (pairing_attempts={method.pairing_attempts}) ========")
        summary: List[ConcentrationStats] = []

        # 内側: 欠陥濃度
        for percent in percents:
            n_pairs = _n_pairs(n_water, percent)
            defect_site_pct = 200.0 * n_pairs / n_water
            placed = connect_ok = ice_ok = 0
            ice_fail_reasons: DefaultDict[str, int] = defaultdict(int)

            for t in range(trials):
                rng = np.random.default_rng(
                    base_seed + t + int(percent * 1000) + hash(method.label) % 10_000
                )
                if not place(n_pairs, rng):
                    continue

                placed += 1
                fixed = genice.fixed_edges
                if _try_connect(method, fixed, graph):
                    connect_ok += 1
                if ice_graph_test:
                    ok, reason = _try_ice_graph(
                        method, graph, lattice_sites, fixed, base_seed + t
                    )
                    if ok:
                        ice_ok += 1
                    elif reason:
                        ice_fail_reasons[reason] += 1

            summary.append(
                ConcentrationStats(
                    percent,
                    n_pairs,
                    placed,
                    connect_ok,
                    ice_ok,
                    ice_fail_reasons=ice_fail_reasons,
                )
            )
            fail_note = _format_ice_failures(ice_fail_reasons)
            if placed == 0:
                print(
                    f"  {percent:5.1f}%  → {n_pairs:3d} pairs "
                    f"({defect_site_pct:5.2f}% sites): placement failed"
                )
            elif ice_graph_test:
                print(
                    f"  {percent:5.1f}%  → {n_pairs:3d} pairs "
                    f"({defect_site_pct:5.2f}% sites): "
                    f"connect {connect_ok}/{placed}, ice_graph {ice_ok}/{placed}"
                    f"{fail_note}"
                )
            else:
                print(
                    f"  {percent:5.1f}%  → {n_pairs:3d} pairs "
                    f"({defect_site_pct:5.2f}% sites): "
                    f"connect {connect_ok}/{placed}"
                )

        print()
        print(f"  [{method.label}] 濃度 vs 成功率 (connect)")
        print(f"  {'%':>6}  {'pairs':>5}  {'connect':>12}")
        for row in summary:
            rate = (
                f"{row.connect_ok}/{row.placed}"
                if row.placed
                else "n/a"
            )
            print(f"  {row.percent:5.1f}  {row.n_pairs:5d}  {rate:>12}")
        if ice_graph_test:
            print(f"  [{'ice_graph':^12}]")
            for row in summary:
                rate = (
                    f"{row.ice_ok}/{row.placed}"
                    if row.placed
                    else "n/a"
                )
                print(f"  {row.percent:5.1f}  {row.n_pairs:5d}  {rate:>12}")
        print()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--percents",
        type=float,
        nargs="+",
        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
        help="欠陥サイト割合 (水分子数に対する %%)。",
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--pairing-attempts",
        type=int,
        default=100,
        help="従来法の pairing_attempts。",
    )
    parser.add_argument(
        "--placement",
        choices=("sites", "fractional"),
        default="sites",
    )
    parser.add_argument("--min-graph-distance", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-ice-graph",
        action="store_true",
        help="connect 段階のみ計測（高速）。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="ice_graph 失敗時に genice_core の INFO も表示する。",
    )
    args = parser.parse_args(argv)

    if not args.verbose:
        _configure_quiet_logging()
    # 比較用に従来法を呼ぶが、DeprecationWarning は表示しない
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=".*connect_matching_paths_mcf.*",
    )

    run_benchmark(
        percents=args.percents,
        trials=args.trials,
        pairing_attempts_random=args.pairing_attempts,
        placement=args.placement,
        min_graph_distance=args.min_graph_distance,
        base_seed=args.seed,
        ice_graph_test=not args.skip_ice_graph,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
