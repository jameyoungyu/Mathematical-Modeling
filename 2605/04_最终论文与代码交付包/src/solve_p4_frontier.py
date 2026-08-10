#!/usr/bin/env python3
"""问题四的补充：把可行边界测准，并算出介质 B 的盈亏平衡价格。

`solve_p4.py` 的网格搜索给出最优点是纯 A，但沿可行边界的成本差别只有百分之一量级，
这种"几乎打平"的结论不能只靠网格拟合下结论——必须把边界本身量出来。

做法：对若干个 $N_A$，二分求出使 $P\\ge0.90$ 的最小 $N_B$（即边界 $g(N_A)$），
然后：

* 直接比较边界上各点的成本，看最优是不是真的在 $N_B=0$ 处；
* 用边界斜率 $g'$ 算介质 B 的盈亏平衡单价：
  成本 $C(N_A)=c_AN_A+c_Bg(N_A)$，$\\mathrm dC/\\mathrm dN_A=c_A+c_Bg'$。
  $g'<0$，所以当 $c_B<-c_A/g'$ 时成本随 $N_A$ 增大而上升，此时该多用 B；
  反之该走纯 A。盈亏平衡点 $c_B^\\*=-c_A/g'$。

结果写入 results/p4_frontier.json。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from microstructure import COST_A, COST_B, V_A, V_B, V_BOX
from simulate import Config, estimate_p

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGET = 0.90
N_A_LIST = [512, 532, 552]
BISECT_TRIALS = 5000
NB_HI = 900


def min_nb(n_a: int, mode: str = "polar_uniform") -> dict:
    """二分求使 P≥TARGET 的最小 N_B。P 对 N_B 单调不减，二分成立。"""
    probes: list[dict] = []

    def p_at(nb: int) -> float:
        r = estimate_p(Config(n_a=n_a, n_b=nb, orientation=mode), BISECT_TRIALS)
        probes.append({"n_b": nb, "p": r["p"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
        print(f"    N_A={n_a} N_B={nb:4d} -> P={r['p']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
        return r["p"]

    if p_at(0) >= TARGET:
        return {"n_a": n_a, "n_b_min": 0, "probes": probes}
    hi = NB_HI
    if p_at(hi) < TARGET:
        return {"n_a": n_a, "n_b_min": None, "probes": probes}
    lo = 0
    while hi - lo > 24:                       # 24 颗 ≈ 0.04 元，已细于成本比较所需
        mid = (lo + hi) // 2
        if p_at(mid) >= TARGET:
            hi = mid
        else:
            lo = mid
    return {"n_a": n_a, "n_b_min": hi, "probes": probes}


def main() -> int:
    mode = "polar_uniform"
    print("=== 测量可行边界 N_B = g(N_A) ===")
    frontier = []
    for n_a in N_A_LIST:
        t0 = time.time()
        r = min_nb(n_a, mode)
        if r["n_b_min"] is not None:
            r["cost_yuan"] = n_a * COST_A + r["n_b_min"] * COST_B
            r["phi_a"] = n_a * V_A / V_BOX
            r["phi_b"] = r["n_b_min"] * V_B / V_BOX
            print(f"  N_A={n_a}: 最小 N_B={r['n_b_min']}，成本 {r['cost_yuan']:.4f} 元"
                  f"（{time.time()-t0:.0f}s）")
        frontier.append(r)

    usable = [r for r in frontier if r.get("n_b_min") is not None]
    out = {"mode": mode, "target": TARGET, "bisect_trials": BISECT_TRIALS,
           "frontier": frontier}

    if len(usable) >= 2:
        xs = np.array([r["n_a"] for r in usable], float)
        ys = np.array([r["n_b_min"] for r in usable], float)
        slope, intercept = np.polyfit(xs, ys, 1)
        c_b_star = -COST_A / slope if slope < 0 else None
        out["boundary_slope_dNB_dNA"] = float(slope)
        out["break_even_cost_per_sphere_yuan"] = c_b_star
        if c_b_star:
            out["break_even_unit_price_yuan_per_um3"] = float(c_b_star / (V_B / 1e9))
        best = min(usable, key=lambda r: r["cost_yuan"])
        out["best_on_frontier"] = best
        out["cost_spread_yuan"] = float(max(r["cost_yuan"] for r in usable)
                                        - min(r["cost_yuan"] for r in usable))
        print(f"\n边界斜率 g' = {slope:.2f} 颗/根")
        print(f"边界上成本最低点：N_A={best['n_a']}, N_B={best['n_b_min']}, "
              f"{best['cost_yuan']:.4f} 元；边界成本极差 {out['cost_spread_yuan']:.4f} 元")
        if c_b_star:
            print(f"介质 B 的盈亏平衡：单颗 {c_b_star:.6f} 元，"
                  f"即单价 {out['break_even_unit_price_yuan_per_um3']:.4f} 元/μm³"
                  f"（现价 0.05 元/μm³）")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p4_frontier.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
