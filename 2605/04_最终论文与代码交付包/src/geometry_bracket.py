#!/usr/bin/env python3
"""球柱体近似的严格上下界：把"近似是否影响答案"从估计变成证明。

评审意见指出：介质 A 是平端面直圆柱，本文按球柱体（capsule）处理，
而模型工作在渗流跃变的陡峭段上，少数几条被误判的接触就可能把最低体积分数
推到相邻的 0.01% 网格。这条担心是对的，但"约 3.3% 的临界接触受影响"只是量级估计，
不足以说明答案稳不稳。

这里改用**几何包含关系**给出严格的双侧界，不需要实现平端面圆柱的精确距离：

设真实圆柱 $C$ 的轴长 $h$、半径 $r$。记
  外界 $K^+$ = 以整条轴线段（长 $h$）为骨架、半径 $r$ 的球柱体；
  内界 $K^-$ = 以缩短后的轴线段（长 $h-2r$）为骨架、半径 $r$ 的球柱体。
则

    K^- ⊆ C ⊆ K^+ .

证明（内界）：设 $x$ 到缩短轴线段的距离 $\\le r$，即 $x=p+v$，$p$ 在缩短段上，$|v|\\le r$。
把 $v$ 分解为轴向 $v_\\parallel$ 与径向 $v_\\perp$，则 $x$ 的轴向坐标满足
$|t_p+v_\\parallel|\\le (h/2-r)+r=h/2$，径向满足 $|v_\\perp|\\le|v|\\le r$，故 $x\\in C$。
外界同理（$C$ 中任一点到轴线段的距离不超过 $r$）。

包含关系对介质—介质与介质—带电面两类判定同时成立，因此

    P(K^-) \\le P(\\text{真实圆柱}) \\le P(K^+),

而两端都只是把轴长换掉，完全复用同一套内核。若两端给出同一个 0.01% 网格答案，
则该答案对"球柱体近似"这一项是**被证明**稳健的，而不是被估计为稳健。

结果写入 results/geometry_bracket.json。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from microstructure import (EDGE, H_A, R_A, V_A, V_BOX, PRIMARY_ORIENTATION,
                            percolates, sample_rods)
from simulate import Config, estimate_p, n_rods_for_phi, wilson

RESULTS = Path(__file__).resolve().parents[1] / "results"
BOX = np.full(3, EDGE)

L_OUTER = H_A                 # 外界：轴长 5000（本文主口径）
L_INNER = H_A - 2 * R_A       # 内界：轴长 4940

P2_FRACTIONS = [0.0050, 0.0060, 0.0070, 0.0100]
# 问题三的扫描窗口**不写死**：从 p3_threshold.json 里读主口径的答案，
# 再向两侧各展开若干格。写死会在换方向口径后静默扫错区间——
# 旧版把窗口固定在 0.76%–0.80%（极角均匀口径的答案邻域），
# 换成球面均匀（答案 0.86%）后整段都够不到 90%，两端都返回 None。
P3_SPAN = 3          # 答案两侧各展开几个 0.01% 网格
TRIALS_P2 = 6000
TRIALS_P3 = 6000
TARGET = 0.90


def run(n_a: int, length: float, trials: int, seed: int = 20260810) -> dict:
    """内界/外界只差轴长，其余完全一致（同一套内核、同一组随机种子）。"""
    r = estimate_p(Config(n_a=n_a, orientation=PRIMARY_ORIENTATION, rod_length=length),
                   trials, seed=seed)
    r["length_nm"] = length
    return r


def p3_grid() -> list[float]:
    """以主口径的问题三答案为中心构造 0.01% 网格窗口。"""
    f = RESULTS / "p3_threshold.json"
    if not f.exists():
        raise SystemExit("先运行 solve_p3.py")
    phi0 = json.loads(f.read_text(encoding="utf-8"))["modes"][PRIMARY_ORIENTATION]["answer_phi"]
    k0 = round(phi0 * 10000)          # 以 0.01% 为单位的整数格
    return [round((k0 + d) / 10000, 6) for d in range(-P3_SPAN, P3_SPAN + 1)]


def main() -> int:
    out = {"L_outer_nm": L_OUTER, "L_inner_nm": L_INNER,
           "note": "K^- ⊆ 真实圆柱 ⊆ K^+，故 P(K^-) ≤ P(真实) ≤ P(K^+)",
           "problem2": [], "problem3": []}

    # 问题二的四档与问题三的窗口无关，可断点复用（同种子同样本量，结果逐位一致）
    prior2 = {}
    dst = RESULTS / "geometry_bracket.json"
    if dst.exists():
        try:
            old = json.loads(dst.read_text(encoding="utf-8"))
            if old.get("L_inner_nm") == L_INNER and old.get("L_outer_nm") == L_OUTER:
                for r in old.get("problem2", []):
                    if r["upper"].get("orientation") == PRIMARY_ORIENTATION \
                       and r["upper"].get("trials") == TRIALS_P2:
                        prior2[r["n_a"]] = r
        except Exception:
            prior2 = {}

    print("=== 问题二：四档体积分数的上下界 ===")
    for phi in P2_FRACTIONS:
        n = n_rods_for_phi(phi)
        if n in prior2:
            row = prior2[n]
            out["problem2"].append(row)
            print(f"  φ={phi:.2%} N_A={n:4d}  P∈[{row['lower']['p']:.4f}, "
                  f"{row['upper']['p']:.4f}]  带宽={row['width']:.4f}  （复用已算结果）")
            continue
        t0 = time.time()
        hi = run(n, L_OUTER, TRIALS_P2)
        lo = run(n, L_INNER, TRIALS_P2)
        row = {"phi": phi, "n_a": n, "upper": hi, "lower": lo,
               "width": hi["p"] - lo["p"]}
        out["problem2"].append(row)
        print(f"  φ={phi:.2%} N_A={n:4d}  P∈[{lo['p']:.4f}, {hi['p']:.4f}]"
              f"  带宽={row['width']:.4f}  ({time.time()-t0:.0f}s)")

    grid = p3_grid()
    print(f"\n=== 问题三：0.01% 网格上的上下界（窗口 {grid[0]:.2%}–{grid[-1]:.2%}）===")
    for phi in grid:
        n = n_rods_for_phi(phi)
        t0 = time.time()
        hi = run(n, L_OUTER, TRIALS_P3)
        lo = run(n, L_INNER, TRIALS_P3)
        row = {"phi": phi, "n_a": n, "upper": hi, "lower": lo,
               "upper_ok": hi["p"] >= TARGET, "lower_ok": lo["p"] >= TARGET}
        out["problem3"].append(row)
        print(f"  φ={phi:.2%} N_A={n:4d}  P∈[{lo['p']:.4f}, {hi['p']:.4f}]"
              f"  下界达标={row['lower_ok']} 上界达标={row['upper_ok']}"
              f"  ({time.time()-t0:.0f}s)")

    ok_lo = [r["phi"] for r in out["problem3"] if r["lower_ok"]]
    ok_hi = [r["phi"] for r in out["problem3"] if r["upper_ok"]]
    out["phi_min_lower_bound_model"] = min(ok_lo) if ok_lo else None
    out["phi_min_upper_bound_model"] = min(ok_hi) if ok_hi else None
    # 两端都为 None 说明窗口没覆盖到达标点，这不是"一致"，是扫描失败
    out["scan_covers_target"] = out["phi_min_upper_bound_model"] is not None
    out["answer_proved_robust"] = (
        out["scan_covers_target"]
        and out["phi_min_lower_bound_model"] == out["phi_min_upper_bound_model"])
    print(f"\n内界模型给出的最低体积分数：{out['phi_min_lower_bound_model']}")
    print(f"外界模型给出的最低体积分数：{out['phi_min_upper_bound_model']}")
    if not out["scan_covers_target"]:
        print("★ 扫描窗口内没有任何一格达标，请扩大 P3_SPAN 后重跑")
    elif out["answer_proved_robust"]:
        print("两端一致 -> 答案对球柱体近似被证明稳健")
    else:
        print("两端不一致 -> 需按内界口径改写答案")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "geometry_bracket.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
