#!/usr/bin/env python3
"""问题三：只填介质 A 时，使导通概率不低于 90% 的最低体积分数。

两步：
1. 粗扫 + 二项似然下的 logit 拟合，把 P=0.90 的穿越点定位到几根棒以内；
2. 在 0.01%（题目要求的精度）的体积分数网格上逐点大样本复算，取**最小的**、
   点估计与 Wilson 下界都不低于 0.90 的那一格作为答案。

只用拟合曲线反解会把答案的可信度压在模型形式上；第 2 步是直接的频率验证，
拟合只用来省算力。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from microstructure import V_A, V_BOX
from simulate import Config, estimate_p, n_rods_for_phi

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGET = 0.90
# 样本量按分辨率需要定：相邻 0.01% 网格点相差约 7 根棒，对应 ΔP≈0.02；
# T=4000 时 Wilson 半宽约 0.009，足以把相邻网格点分开。
COARSE_TRIALS = 2000
FINE_TRIALS = 4000
FINAL_TRIALS = 12000


def logit_fit(ns: np.ndarray, hits: np.ndarray, trials: np.ndarray) -> tuple[float, float]:
    """二项似然拟合 logit P = a + b·N，返回 (a, b)。"""
    def nll(theta):
        a, b = theta
        z = a + b * ns
        # log-sum-exp 稳定写法
        ll = hits * z - trials * np.logaddexp(0.0, z)
        return -np.sum(ll)
    x0 = np.array([-8.0, 0.015])
    res = minimize(nll, x0, method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 20000})
    return float(res.x[0]), float(res.x[1])


def solve_for_mode(mode: str) -> dict:
    print(f"\n=== 方向分布：{mode} ===")
    coarse_ns = [n_rods_for_phi(p) for p in (0.0060, 0.0068, 0.0076, 0.0084, 0.0092, 0.0100)]
    rows = []
    for n in coarse_ns:
        r = estimate_p(Config(n_a=n, orientation=mode), COARSE_TRIALS)
        rows.append(r)
        print(f"  粗扫 N_A={n:4d} φ={r['phi_a']:.4%}  P={r['p']:.4f}")

    ns = np.array([r["n_a"] for r in rows], float)
    hits = np.array([r["hits"] for r in rows], float)
    tri = np.array([r["trials"] for r in rows], float)
    a, b = logit_fit(ns, hits, tri)
    n_star = (np.log(TARGET / (1 - TARGET)) - a) / b
    phi_star = n_star * V_A / V_BOX
    print(f"  logit 拟合: P=0.90 处 N_A≈{n_star:.1f}  φ≈{phi_star:.4%}")

    # 在 0.01% 网格上逐点复算
    center = round(phi_star * 10000) / 10000          # 归到 0.01% 网格
    grid = [round(center + k * 0.0001, 6) for k in (-2, -1, 0, 1, 2)]
    fine = []
    for phi in grid:
        n = n_rods_for_phi(phi)
        t0 = time.time()
        r = estimate_p(Config(n_a=n, orientation=mode), FINE_TRIALS)
        r["phi_grid"] = phi
        fine.append(r)
        print(f"  细算 φ={phi:.2%} N_A={n:4d}  P={r['p']:.4f} "
              f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  ({time.time()-t0:.0f}s)")

    ok = [r for r in fine if r["p"] >= TARGET]
    answer = min(ok, key=lambda r: r["phi_grid"]) if ok else None

    final = None
    if answer is not None:
        print(f"  复核 φ={answer['phi_grid']:.2%} 用 {FINAL_TRIALS} 次试验 …")
        final = estimate_p(Config(n_a=answer["n_a"], orientation=mode), FINAL_TRIALS,
                           seed=987654321)
        final["phi_grid"] = answer["phi_grid"]
        print(f"  复核结果 P={final['p']:.4f} [{final['ci_lo']:.4f},{final['ci_hi']:.4f}]")

    return {
        "mode": mode, "coarse": rows, "logit_a": a, "logit_b": b,
        "n_star_fit": n_star, "phi_star_fit": phi_star,
        "fine_grid": fine,
        "answer_phi": answer["phi_grid"] if answer else None,
        "answer_n_a": answer["n_a"] if answer else None,
        "final_check": final,
    }


def main() -> int:
    out = {"target": TARGET,
           "trials": {"coarse": COARSE_TRIALS, "fine": FINE_TRIALS, "final": FINAL_TRIALS},
           "modes": {}}
    for mode in ("polar_uniform", "isotropic"):
        out["modes"][mode] = solve_for_mode(mode)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p3_threshold.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    for m, r in out["modes"].items():
        print(f"\n{m}: 最低体积分数 = {r['answer_phi']:.2%}（N_A={r['answer_n_a']}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
