#!/usr/bin/env python3
"""介质 B 的盈亏平衡价：多便宜才值得掺？

问题四的答案是"纯 A"，但这是**价格**的结论而不是几何的结论。把它说清楚需要一个数：
介质 B 的单价降到多少，最优解才会从纯 A 变成混填。

设可行边界为 $N_B=g(N_A)$（即达到 $P\\ge0.90$ 所需的最小球数），边界上的成本

    C(N_A) = c_A N_A + c_B g(N_A),   dC/dN_A = c_A + c_B g'.

$g'<0$。若 $c_A+c_B g'<0$，成本随 $N_A$ 增大而下降，最优在边界右端点（纯 A）；
反之应当少用 A、多用 B。盈亏平衡价即 $c_B^\\*=-c_A/g'$。

$g'$ 直接由 `solve_p4.py` 阶段 C **已验证通过**的边界点线性拟合得到——
这些点每个都用 6000 次试验确认了 $P\\ge0.90$，比再拟合一层代理模型可靠。

结果写入 results/p4_break_even.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microstructure import COST_A, COST_B, V_B

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> int:
    src = RESULTS / "p4_cost_optimum.json"
    if not src.exists():
        print("先运行 solve_p4.py")
        return 1
    d = json.loads(src.read_text(encoding="utf-8"))
    pts = sorted(((v["n_a"], v["n_b"]) for v in d["verified"]), key=lambda t: t[0])
    # 边界只在 g>0 的区段上才有斜率可言。N_B 已经取到 0 的点里，只有 N_A 最小的那个
    # 位于边界上（再减 A 就必须补 B）；更大的 N_A 配 N_B=0 是"配足过头"的内点，
    # 把它们放进拟合会把斜率压平（-11.4 → -9.1），进而把盈亏平衡价算高。
    zeros = [t for t in pts if t[1] == 0]
    if len(zeros) > 1:
        keep = min(zeros, key=lambda t: t[0])
        pts = [t for t in pts if t[1] > 0 or t == keep]
    if len(pts) < 2:
        print("已验证的边界点不足两个，无法拟合斜率")
        return 1

    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    slope, intercept = np.polyfit(x, y, 1)
    c_b_star = -COST_A / slope if slope < 0 else None
    unit_price_now = COST_B / (V_B / 1e9)          # 元/μm³，应为 0.05

    out = {
        "boundary_points_verified": [{"n_a": int(a), "n_b": int(b)} for a, b in pts],
        "slope_dNB_dNA": float(slope),
        "intercept": float(intercept),
        "cost_per_rod_yuan": COST_A,
        "cost_per_sphere_yuan": COST_B,
        "unit_price_now_yuan_per_um3": float(unit_price_now),
    }
    print(f"已验证的边界点：{[(int(a), int(b)) for a, b in pts]}")
    print(f"边界斜率 g' = {slope:.4f} 颗/根（少用 1 根 A 需补约 {-slope:.1f} 颗 B）")
    if c_b_star:
        unit_star = c_b_star / (V_B / 1e9)
        out["break_even_cost_per_sphere_yuan"] = float(c_b_star)
        out["break_even_unit_price_yuan_per_um3"] = float(unit_star)
        out["required_price_cut_pct"] = float((1 - unit_star / unit_price_now) * 100)
        print(f"盈亏平衡：单颗 {c_b_star:.6e} 元，即单价 {unit_star:.4f} 元/μm³")
        print(f"现价 {unit_price_now:.4f} 元/μm³，需再降 "
              f"{out['required_price_cut_pct']:.1f}% 才值得掺 B")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p4_break_even.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
