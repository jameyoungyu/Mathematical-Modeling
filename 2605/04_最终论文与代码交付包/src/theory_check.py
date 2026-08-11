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


def mode_contrast() -> dict | None:
    """两套方向口径在各档体积分数上的概率差，以及主口径的逐档增量。

    论文正文引用了这些差值；在这里算出来，它们才有出处，而不是正文里手推的数。
    """
    p = RESULTS / "p2_probabilities.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    pol = {r["phi_nominal"]: r["p"] for r in d["runs"]["polar_uniform"]}
    iso = {r["phi_nominal"]: r["p"] for r in d["runs"]["isotropic"]}
    phis = sorted(pol)
    return {
        "gap_polar_minus_isotropic": {f"{k:.2%}": pol[k] - iso[k] for k in phis},
        "increment_polar": {f"{a:.2%}->{b:.2%}": pol[b] - pol[a]
                            for a, b in zip(phis[:-1], phis[1:])},
    }


def marginal_efficiency() -> dict | None:
    """从问题四的网格算两种介质的"每元换来多少导通概率"。

    这是问题四推荐方案的机理解释：介质 B 单价便宜，但在已取样的阈值附近，
    介质 A 的边际收益更陡。该局部差分用于解释候选排序，不构成全局最优证明。
    """
    p = RESULTS / "p4_cost_optimum.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    c_a, c_b = d["cost_per_rod_yuan"], d["cost_per_sphere_yuan"]
    grid = {(g["n_a"], g["n_b"]): g["p"] for g in d["grid"]}
    nas = sorted({k[0] for k in grid})
    nbs = sorted({k[1] for k in grid})
    out = {}
    for i, na in enumerate(nas):
        row = {}
        if i > 0:                                   # dP/dN_A（沿 N_B=0）
            prev = nas[i - 1]
            dp = grid[(na, 0)] - grid[(prev, 0)]
            row["dP_dNA_per_yuan"] = (dp / (na - prev)) / c_a
        if len(nbs) > 1:                            # dP/dN_B（在 N_B=0 处）
            nb1 = nbs[1]
            dp = grid[(na, nb1)] - grid[(na, 0)]
            row["dP_dNB_per_yuan"] = (dp / nb1) / c_b
        if row:
            row["P_at_NB0"] = grid[(na, 0)]
            out[str(na)] = row
    return out


def main() -> int:
    ev = excluded_volume_threshold()
    mid = midpoint_from_simulation()
    out = {"excluded_volume_estimate": ev, "simulation_midpoint": mid,
           "mode_contrast": mode_contrast(),
           "marginal_efficiency": marginal_efficiency()}
    print("排除体积判据（独立于仿真的理论估计）：")
    print(f"  平均排除体积 <V_ex> = {ev['mean_excluded_volume_nm3']:.4e} nm^3")
    print(f"  临界根数 N_c ≈ {ev['critical_n_rods']:.0f}，"
          f"对应体积分数 ≈ {ev['critical_volume_fraction']:.4%}")
    if mid:
        for m, v in mid.items():
            if v["phi_at_P50"]:
                print(f"  仿真跃变中点 ({m}): φ(P=0.5) ≈ {v['phi_at_P50']:.4%}")
    if out["mode_contrast"]:
        print("两套方向口径的概率差：",
              {k: round(v, 4) for k, v in out["mode_contrast"]["gap_polar_minus_isotropic"].items()})
        print("主口径逐档增量：",
              {k: round(v, 4) for k, v in out["mode_contrast"]["increment_polar"].items()})
    if out["marginal_efficiency"]:
        print("每元边际收益（ΔP/元）：")
        for na, v in out["marginal_efficiency"].items():
            a = v.get("dP_dNA_per_yuan"); b = v.get("dP_dNB_per_yuan")
            print(f"  N_A={na:>4}  P(N_B=0)={v['P_at_NB0']:.3f}  "
                  f"介质A={'—' if a is None else f'{a:.3f}'}  "
                  f"介质B={'—' if b is None else f'{b:.3f}'}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "theory_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
