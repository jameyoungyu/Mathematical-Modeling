#!/usr/bin/env python3
"""问题四预算线审计：记录已严格覆盖范围，并显式保留未形成证书的区段。

原做法的漏洞（评审意见指出，属实）：代理模型只在 8×6 的稀疏网格上训练，
训练点上的最大残差 0.030 并不能约束网格之外的误差；即使把代理模型排在前面的候选
逐个大样本验证，也排除不掉"代理模型根本没看见的更便宜的可行点"。

严格做法只需要用到已经证明的单调性：**P 对 $N_A$、$N_B$ 都单调不减**。
记当前最优成本 $C^\\*=c_AN_A^\\*$（纯 A 方案）。要证明它是全局最优成本，
只需说明"所有成本低于 $C^\\*$ 的整数点都不可行"。对固定的 $N_A$，
成本低于 $C^\\*$ 等价于

    N_B \\le N_B^{\\max}(N_A) = \\lceil (C^\\* - c_A N_A)/c_B \\rceil - 1 .

由单调性，只要 $P(N_A, N_B^{\\max}(N_A)) < 0.90$，该 $N_A$ 下所有更便宜的点都不可行。

进一步，不必对每个 $N_A$ 都算一次。取网格 $a_1<a_2$，对任意
$N_A\\in[a_1,a_2]$ 且成本低于 $C^\\*$ 的点 $(N_A,N_B)$ 有

    N_A \\le a_2,   N_B \\le N_B^{\\max}(N_A) \\le N_B^{\\max}(a_1),

于是由单调性 $P(N_A,N_B)\\le P(a_2,\\;N_B^{\\max}(a_1))$。
所以**只要在"角点" $(a_2, N_B^{\\max}(a_1))$ 上验证 $P<0.90$，整条区间就被排除**。
角点本身的成本高于 $C^\\*$，它不是候选解，只是一个上界。

这样，$O(551)$ 个 $N_A$ 的穷举被压缩成 $\\lceil 551/\\text{step}\\rceil$ 次仿真，
而结论仍然是严格的（相对于蒙特卡洛的统计误差——因此角点判据取 Wilson 上界 < 0.90，
而不是点估计 < 0.90）。

结果写入 results/p4_global_audit.json。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from microstructure import COST_A, COST_B
from simulate import Config, estimate_p

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGET = 0.90
STEP = 24          # N_A 网格步长；角点判据保证步长内的所有点都被覆盖
TRIALS = 2500


def nb_max(cost_star: float, n_a: int) -> int:
    """成本严格低于 cost_star 时，该 N_A 允许的最大 N_B。"""
    v = (cost_star - n_a * COST_A) / COST_B
    return max(-1, math.ceil(v) - 1)


def main() -> int:
    p4 = RESULTS / "p4_cost_optimum.json"
    if not p4.exists():
        print("先运行 solve_p4.py")
        return 1
    best = json.loads(p4.read_text(encoding="utf-8"))["best"]
    n_a_star, cost_star = best["n_a"], best["cost_yuan"]
    print(f"待证最优：N_A={n_a_star}, N_B={best['n_b']}, 成本={cost_star:.4f} 元")
    print(f"需排除所有成本 < {cost_star:.4f} 元的整数点\n")

    grid = list(range(0, n_a_star + 1, STEP))
    if grid[-1] != n_a_star:
        grid.append(n_a_star)

    # 角点阶段可断点续跑：结果文件里已有的 checks 直接复用（同种子同样本量，结果一致）
    prior, prior_line = {}, {}
    dst = RESULTS / "p4_global_audit.json"
    if dst.exists():
        try:
            old = json.loads(dst.read_text(encoding="utf-8"))
            for c in old.get("checks", []):
                prior[(c["a1"], c["a2"])] = c
            for c in old.get("budget_line_checks", []):
                prior_line[(c["n_a"], c["n_b"])] = c
        except Exception:
            prior, prior_line = {}, {}

    checks, excluded_all = [], True
    for a1, a2 in zip(grid[:-1], grid[1:]):
        if (a1, a2) in prior:
            c = prior[(a1, a2)]
            checks.append(c); excluded_all &= c["excluded"]
            print(f"  N_A∈[{a1:3d},{a2:3d}] （复用已算结果）"
                  f"{'已排除' if c['excluded'] else '★未排除'}")
            continue
        nb = nb_max(cost_star, a1)
        if nb < 0:
            checks.append({"a1": a1, "a2": a2, "nb_corner": nb,
                           "note": "该区间内无成本更低的点", "excluded": True})
            continue
        t0 = time.time()
        r = estimate_p(Config(n_a=a2, n_b=nb), TRIALS)
        ok = r["ci_hi"] < TARGET          # Wilson 上界都够不到 0.90 才算排除
        excluded_all &= ok
        checks.append({"a1": a1, "a2": a2, "corner_n_a": a2, "corner_n_b": nb,
                       "p": r["p"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                       "trials": TRIALS, "excluded": bool(ok)})
        print(f"  N_A∈[{a1:3d},{a2:3d}] 角点=({a2:3d},{nb:5d})  "
              f"P={r['p']:.4f} (上界 {r['ci_hi']:.4f})  "
              f"{'已排除' if ok else '★未排除'}  ({time.time()-t0:.0f}s)")

    # 角点界 P(a2, N_B^max(a1)) 在区间较宽时过松：a1→a2 之间 N_B 上限要多算
    # (a2-a1)·c_A/c_B ≈ 24×8.86 ≈ 213 颗球，足以把上界抬过 0.90。
    # 对未排除的区间，改为直接在**预算线本身**上取点：由 P 对 N_B 单调，
    # 只要 P(N_A, N_B^max(N_A)) < 0.90，该 N_A 下所有更便宜的点都不可行。
    unresolved = [c for c in checks if not c["excluded"]]
    line_checks = []
    if unresolved:
        lo = min(c["a1"] for c in unresolved)
        hi = max(c["a2"] for c in unresolved)
        print(f"\n=== 对未排除区间 N_A∈[{lo},{hi}] 直接取预算线上的点 ===")
        for a in range(lo, hi + 1, 16):
            nb = nb_max(cost_star, a)
            if nb < 0:
                continue
            if (a, nb) in prior_line:
                c = prior_line[(a, nb)]
                line_checks.append(c)
                print(f"  预算线点 ({a:3d},{nb:4d}) （复用已算结果）"
                      f"P={c['p']:.4f}  {'已排除' if c['excluded'] else '★未排除'}")
                continue
            t0 = time.time()
            r = estimate_p(Config(n_a=a, n_b=nb), TRIALS)
            ok = r["ci_hi"] < TARGET
            line_checks.append({"n_a": a, "n_b": nb, "cost": a * COST_A + nb * COST_B,
                                "p": r["p"], "ci_hi": r["ci_hi"], "excluded": bool(ok)})
            print(f"  预算线点 ({a:3d},{nb:4d}) 成本={a*COST_A+nb*COST_B:.4f}  "
                  f"P={r['p']:.4f} (上界 {r['ci_hi']:.4f})  "
                  f"{'已排除' if ok else '★未排除'}  ({time.time()-t0:.0f}s)")

    out = {"target": TARGET, "step": STEP, "trials": TRIALS,
           "budget_line_checks": line_checks,
           "incumbent": {"n_a": n_a_star, "n_b": best["n_b"], "cost_yuan": cost_star},
           "checks": checks}
    # 预算线离散取样只能排除被实际检查的列及其分量支配点，不能替代未取样整数列。
    # 旧版把所有已取样列均排除误写成“整个未决区间均排除”，属于覆盖范围错误。
    out_all = excluded_all
    out["all_cheaper_points_excluded"] = bool(out_all)
    if out_all:
        msg = ("在球柱体外界 K+ 口径下，成本低于候选值的整数点全部被排除，"
               f"成本 {cost_star:.4f} 元为该口径的全局最小，(N_A,N_B)=({n_a_star},{best['n_b']}) 达到该值。"
               "该结论不构成真实平端面圆柱的可行性证明。")
    else:
        bad = [c for c in line_checks if not c["excluded"]]
        msg = (f"角点法仅严格覆盖 N_A≤480；预算线离散检查另排除部分列。"
               f"N_A∈[481,{n_a_star - 1}] 尚未逐列形成完整证书，"
               f"且已检查点中有 {len(bad)} 个 Wilson 上界未低于 0.90。"
               f"因此不能声称成本下界或全局最优；只能说 ({n_a_star},{best['n_b']})、"
               f"成本 {cost_star:.4f} 元是球柱体外界 K+ 下经抽样确认的候选。"
               "真实平端面圆柱的可行性须另看几何双侧界。")
        out["strictly_covered_n_a_max"] = 480
        out["not_fully_certified_n_a_interval"] = [481, n_a_star - 1]
        out["budget_line_sampled_columns"] = [c["n_a"] for c in line_checks]
        out["budget_line_excluded_columns"] = [c["n_a"] for c in line_checks if c["excluded"]]
    print("\n结论：" + msg)
    out["verdict"] = msg
    out["model_scope"] = "spherocylinder outer-bound model K+"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p4_global_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
