#!/usr/bin/env python3
"""对预算线上"擦边"的点做大样本复核。

p4_global_audit.py 用 2500 次试验做区间排除，这对远离阈值的点足够，
但对最靠近最优解的两三个点不够：那里 Wilson 半宽（约 ±0.012）与
"点估计到 0.90 的距离"同量级，于是出现两种都不可靠的判读——

  (576,283) 上界 0.8994，仅比 0.90 低 0.0006 —— 名义排除，实则擦边；
  (592,141) 点估计 0.9004，上界 0.9115 —— 未排除，且其成本 9.0239 元
            **低于**当前最优 9.0252 元，若它真可行，则最优解要换人。

本脚本对这些点各做 TRIALS 次试验（默认 20000，半宽约 ±0.004），
把结论建立在能分辨 0.89/0.90/0.91 的样本量上，而不是靠 2500 次的擦边。

结果写入 results/p4_marginal_recheck.json。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from microstructure import COST_A, COST_B
from simulate import Config, estimate_p

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGET = 0.90
TRIALS = 20000
SEED = 20260811          # 与审计不同的种子，兼作独立复核

# 预算线上最靠近最优解的两个点（由 p4_global_audit 的 budget_line_checks 挑出）
POINTS = [(576, 283), (592, 141)]


def main() -> int:
    p4 = RESULTS / "p4_cost_optimum.json"
    if not p4.exists():
        print("先运行 solve_p4.py")
        return 1
    best = json.loads(p4.read_text(encoding="utf-8"))["best"]
    c_star = best["cost_yuan"]
    print(f"当前最优：({best['n_a']},{best['n_b']})  成本={c_star:.4f} 元")
    print(f"每点 {TRIALS} 次试验，种子 {SEED}\n")

    rows = []
    for n_a, n_b in POINTS:
        t0 = time.time()
        r = estimate_p(Config(n_a=n_a, n_b=n_b), TRIALS, seed=SEED)
        cost = n_a * COST_A + n_b * COST_B
        # 三种判读，取决于区间落在 0.90 的哪一侧
        if r["ci_hi"] < TARGET:
            verdict = "确认不可行（上界仍低于 0.90）→ 可排除"
        elif r["ci_lo"] >= TARGET:
            verdict = "确认可行（下界已达 0.90）→ 比当前最优更便宜，最优解需更换"
        else:
            verdict = "区间仍跨越 0.90 → 与 0.90 统计上不可区分，最优解退化"
        rows.append({"n_a": n_a, "n_b": n_b, "cost_yuan": cost,
                     "cheaper_than_incumbent": bool(cost < c_star),
                     "p": r["p"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                     "trials": TRIALS, "seed": SEED, "verdict": verdict})
        print(f"  ({n_a},{n_b}) 成本={cost:.4f} 元"
              f"（{'低于' if cost < c_star else '不低于'}最优）  "
              f"P={r['p']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  "
              f"{verdict}  ({time.time()-t0:.0f}s)")

    out = {"target": TARGET, "trials": TRIALS, "seed": SEED,
           "incumbent": {"n_a": best["n_a"], "n_b": best["n_b"], "cost_yuan": c_star},
           "points": rows}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p4_marginal_recheck.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
