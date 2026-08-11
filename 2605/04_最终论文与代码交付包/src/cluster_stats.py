#!/usr/bin/env python3
"""渗流的结构证据：最大团簇占比随体积分数的变化。

导通概率 P 只告诉我们"通没通"，看不出通的时候内部长什么样。渗流理论里刻画这件事的量
是**序参量**——最大连通团簇所含碎片占全部碎片的比例 $S_{\\max}$。若体系真的经历渗流相变，
$S_{\\max}$ 会在与 $P$ 跃变同一位置由接近 0 迅速升到 O(1)；若只是有限尺寸下的偶然贯通，
两者的位置就不会对上。

这是一条独立于导通判定的检查：$S_{\\max}$ 的计算完全不涉及带电面。

结果写入 results/cluster_stats.json。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from microstructure import (EDGE, GAP, R_A, DSU, PRIMARY_ORIENTATION,
                            contact_pairs, percolates,
                            sample_rods)
from simulate import n_rods_for_phi

RESULTS = Path(__file__).resolve().parents[1] / "results"
BOX = np.full(3, EDGE)
FRACTIONS = [0.0030, 0.0040, 0.0050, 0.0060, 0.0070, 0.0080, 0.0090, 0.0100]
TRIALS = 200


def largest_cluster_fraction(sp: np.ndarray, sq: np.ndarray) -> tuple[float, int]:
    """返回（最大团簇碎片占比, 团簇数）。只看介质之间的连通性，不涉及带电面。"""
    n = len(sp)
    if n == 0:
        return 0.0, 0
    dsu = DSU(n)
    for a, b in contact_pairs(sp, sq, 2 * R_A + GAP):
        dsu.union(int(a), int(b))
    sizes = Counter(dsu.find(i) for i in range(n))
    return max(sizes.values()) / n, len(sizes)


def main() -> int:
    rng = np.random.default_rng(20260810)
    rows = []
    for phi in FRACTIONS:
        n_a = n_rods_for_phi(phi)
        smax, ncl, hits = [], [], 0
        smax_on, smax_off = [], []      # 分别记录导通/不导通实现的 S_max
        for _ in range(TRIALS):
            sp, sq, _ = sample_rods(n_a, rng, BOX, orientation=PRIMARY_ORIENTATION)
            f, k = largest_cluster_fraction(sp, sq)
            smax.append(f)
            ncl.append(k)
            on = bool(percolates(sp, sq))
            hits += on
            (smax_on if on else smax_off).append(f)
        # 论文第 5.3 节要论证的是"导通时内部长什么样"，需要的是条件期望而非无条件均值：
        # 无条件均值里混着大量未导通的实现，在跃变段会把 S_max 拉低。
        row = {
            "phi": phi, "n_a": n_a, "trials": TRIALS,
            "s_max_mean": float(np.mean(smax)), "s_max_sd": float(np.std(smax, ddof=1)),
            "s_max_mean_given_percolating": float(np.mean(smax_on)) if smax_on else None,
            "s_max_mean_given_not": float(np.mean(smax_off)) if smax_off else None,
            "n_percolating": len(smax_on),
            "n_clusters_mean": float(np.mean(ncl)),
            "p_percolate": hits / TRIALS,
        }
        rows.append(row)
        cond = row["s_max_mean_given_percolating"]
        print(f"  φ={phi:.2%} N_A={n_a:4d}  S_max={row['s_max_mean']:.3f}"
              f"±{row['s_max_sd']:.3f}  (导通时 {cond if cond is None else round(cond, 3)})"
              f"  团簇数={row['n_clusters_mean']:.1f}  P={row['p_percolate']:.3f}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cluster_stats.json").write_text(
        json.dumps({"trials": TRIALS, "orientation": PRIMARY_ORIENTATION, "rows": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
