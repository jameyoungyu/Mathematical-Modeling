#!/usr/bin/env python3
"""把仿真结果和渗流理论的独立估计对一对。

蒙特卡洛能给出概率，但给不出"这个数合不合理"。经典的排除体积判据
（Balberg 等，1984）给出随机取向细长杆的渗流阈值满足

    n_c · <V_ex> ≈ B_c ≈ 1.4,

其中 <V_ex> 是两根杆按连通判据的平均排除体积。用它算一个独立的阈值估计，
和仿真给出的跃变中点比较：两者量级一致，说明仿真没有系统性地错；
两者的差异方向（仿真阈值更低）也应当能被解释——本题中是有限尺寸效应
（杆长恰为盒边一半）和方向偏向带电面法向共同造成的。

同时把问题二结果表里的跃变中点用线性插值算出来，避免论文里出现手推的数。
结果写入 results/theory_check.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microstructure import GAP, H_A, R_A, V_A, V_BOX

RESULTS = Path(__file__).resolve().parents[1] / "results"
B_C = 1.4          # 细长杆极限下的临界总排除体积


def excluded_volume_threshold() -> dict:
    """按排除体积判据估计只填介质 A 时的渗流阈值体积分数。"""
    L = H_A
    # 连通判据等价于把杆加粗到"轴间距 ≤ D'"，D' = 2r + δ
    d_eff = 2 * R_A + GAP
    # 两个随机取向球柱体的平均排除体积（<sin γ> = π/4）
    v_ex = (np.pi / 2) * L ** 2 * d_eff + 2 * np.pi * L * d_eff ** 2 + (4 * np.pi / 3) * d_eff ** 3
    n_c = B_C / v_ex                      # 临界数密度 (1/nm^3)
    n_rods = n_c * V_BOX
    return {
        "B_c": B_C,
        "d_eff_nm": d_eff,
        "mean_excluded_volume_nm3": float(v_ex),
        "critical_number_density_per_nm3": float(n_c),
        "critical_n_rods": float(n_rods),
        "critical_volume_fraction": float(n_rods * V_A / V_BOX),
    }


def midpoint_from_simulation() -> dict | None:
    """从问题二的结果表线性插值出 P=0.5 的体积分数。"""
    p = RESULTS / "p2_probabilities.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    out = {}
    for mode, rows in data["runs"].items():
        xs = [r["phi_a"] for r in rows]
        ys = [r["p"] for r in rows]
        mid = None
        for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
            if y0 <= 0.5 <= y1:
                mid = x0 + (x1 - x0) * (0.5 - y0) / (y1 - y0)
                break
        out[mode] = {"phi_at_P50": mid}
    return out


def main() -> int:
    ev = excluded_volume_threshold()
    mid = midpoint_from_simulation()
    out = {"excluded_volume_estimate": ev, "simulation_midpoint": mid}
    print("排除体积判据（独立于仿真的理论估计）：")
    print(f"  平均排除体积 <V_ex> = {ev['mean_excluded_volume_nm3']:.4e} nm^3")
    print(f"  临界根数 N_c ≈ {ev['critical_n_rods']:.0f}，"
          f"对应体积分数 ≈ {ev['critical_volume_fraction']:.4%}")
    if mid:
        for m, v in mid.items():
            if v["phi_at_P50"]:
                print(f"  仿真跃变中点 ({m}): φ(P=0.5) ≈ {v['phi_at_P50']:.4%}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "theory_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
