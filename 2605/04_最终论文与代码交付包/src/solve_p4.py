#!/usr/bin/env python3
"""问题四：同时填 A、B 两种介质，在导通概率不低于 90% 的前提下最小化总成本。

成本：介质 A 1.05 元/μm³ → 每根 0.0148441 元；介质 B 0.05 元/μm³ → 每颗 0.00167552 元。
一颗 B 的价钱只有一根 A 的 1/8.9，但 B 是半径 200 nm 的球，跨接能力远不如长 5000 nm 的棒，
所以两者之间存在真实的权衡，而不是"谁便宜用谁"。

求解结构
--------
决策变量 (N_A, N_B) 都是整数，目标线性，可行域由 P(N_A,N_B) ≥ 0.90 给出。
P 对 N_A、N_B 都单调不减（加介质只会增加接触图的点和边，不会删掉任何通路），
因此最优解一定落在可行域边界 N_B = g(N_A) 上，问题退化为一维搜索。

直接对每个 N_A 二分求 g(N_A) 要跑几百万次仿真。这里改成两段：
  阶段 A：在 (N_A,N_B) 网格上用中等样本量估 P，拟合 logit 代理模型；
  阶段 B：用代理模型定位边界与最优点，再对最优点及其邻域做大样本直接验证。
代理模型只用来省算力，最终结论以阶段 B 的直接频率为准。
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from microstructure import COST_A, COST_B, PRIMARY_ORIENTATION, V_A, V_B, V_BOX
from simulate import Config, estimate_p

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGET = 0.90

GRID_A = [150, 250, 320, 400, 480, 550, 610, 670]
GRID_B = [0, 300, 700, 1200, 2000, 3200]
GRID_TRIALS = 700
VERIFY_TRIALS = 6000


def design_matrix(na: np.ndarray, nb: np.ndarray) -> np.ndarray:
    """代理模型的特征。

    渗流概率对根数近似呈 logit-线性，B 的边际贡献随 A 增多而变化（协同项），
    再加一个 sqrt(N_B) 项刻画球的边际收益递减。
    """
    na = np.asarray(na, float)
    nb = np.asarray(nb, float)
    return np.column_stack([
        np.ones_like(na), na, nb, np.sqrt(nb), na * nb / 1000.0,
    ])


def fit_surrogate(na, nb, hits, trials):
    X = design_matrix(na, nb)
    y = np.asarray(hits, float)
    n = np.asarray(trials, float)

    def nll(w):
        z = np.clip(X @ w, -40, 40)
        return -np.sum(y * z - n * np.logaddexp(0.0, z)) + 1e-6 * np.sum(w * w)

    best = None
    for x0 in (np.zeros(X.shape[1]), np.array([-8.0, 0.02, 0.001, 0.0, 0.0])):
        r = minimize(nll, x0, method="Nelder-Mead",
                     options={"maxiter": 60000, "fatol": 1e-9, "xatol": 1e-9})
        if best is None or r.fun < best.fun:
            best = r
    return best.x


def p_hat(w, na, nb):
    z = design_matrix(np.atleast_1d(na), np.atleast_1d(nb)) @ w
    return 1.0 / (1.0 + np.exp(-z))


def min_nb_for(w, na, nb_max=8000):
    """代理模型下满足 P≥TARGET 的最小 N_B（不可行则返回 None）。"""
    if p_hat(w, na, 0)[0] >= TARGET:
        return 0
    if p_hat(w, na, nb_max)[0] < TARGET:
        return None
    lo, hi = 0, nb_max
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if p_hat(w, na, mid)[0] >= TARGET:
            hi = mid
        else:
            lo = mid
    return hi


def main() -> int:
    mode = PRIMARY_ORIENTATION
    print("=== 阶段 A：网格取样，拟合代理模型 ===")
    grid = []
    for na, nb in itertools.product(GRID_A, GRID_B):
        t0 = time.time()
        r = estimate_p(Config(n_a=na, n_b=nb, orientation=mode), GRID_TRIALS)
        grid.append(r)
        print(f"  N_A={na:4d} N_B={nb:5d} φ_A={r['phi_a']:.3%} φ_B={r['phi_b']:.2%} "
              f"P={r['p']:.3f} 成本={r['cost_yuan']:.3f}元 ({time.time()-t0:.0f}s)")

    na = np.array([g["n_a"] for g in grid], float)
    nb = np.array([g["n_b"] for g in grid], float)
    hits = np.array([g["hits"] for g in grid], float)
    tri = np.array([g["trials"] for g in grid], float)
    w = fit_surrogate(na, nb, hits, tri)
    pred = p_hat(w, na, nb)
    resid = np.max(np.abs(pred - hits / tri))
    print(f"  代理模型最大绝对残差 = {resid:.3f}")

    print("\n=== 阶段 B：沿可行边界一维搜索 ===")
    cand = []
    for a in range(120, 780, 4):
        b = min_nb_for(w, a)
        if b is None:
            continue
        cand.append({"n_a": a, "n_b": b, "cost": a * COST_A + b * COST_B})
    cand.sort(key=lambda c: c["cost"])
    # 代理模型的最优点附近成本几乎相同（N_A 相差 4 根的点成本差不到 0.1%），
    # 全部拿去验证等于把算力花在同一个点上。按 N_A 至少相隔 40 根取互不重复的候选，
    # 既覆盖了边界的不同区段，又能看出成本沿边界是否真的平坦。
    spread: list[dict] = []
    for c in cand:
        if all(abs(c["n_a"] - s["n_a"]) >= 40 for s in spread):
            spread.append(c)
        if len(spread) >= 5:
            break
    print("  代理模型给出的候选（按成本排序，N_A 至少相隔 40）：")
    for c in spread:
        print(f"    N_A={c['n_a']:4d} N_B={c['n_b']:5d} 成本={c['cost']:.4f} 元")
    cand = spread

    print("\n=== 阶段 C：候选点大样本筛选（含向上补 B 直到通过预设筛选规则）===")
    verified = []
    all_probes = []          # 未通过的探测点同样要留痕，否则论文里的表无从溯源
    seen = set()
    for c in cand:
        a = c["n_a"]
        if a in seen:
            continue
        seen.add(a)
        b = c["n_b"]
        for _ in range(6):
            r = estimate_p(Config(n_a=a, n_b=b, orientation=mode), VERIFY_TRIALS)
            all_probes.append(r)
            print(f"    N_A={a:4d} N_B={b:5d} P={r['p']:.4f} "
                  f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] 成本={r['cost_yuan']:.4f} 元")
            if r["p"] >= TARGET and r["ci_lo"] >= TARGET - 0.01:
                verified.append(r)
                break
            b = int(b + max(60, 0.06 * max(b, 500)))
    if not verified:
        print("  没有候选点通过验证")
        return 1

    # 纯 A 方案（问题三在同一 K+ 口径下的答案根数）必须参与候选比较，而不是只作参照。
    # 代理模型的边界搜索按自己的网格取 N_A，未必落在问题三那一格上；若它取到的 N_A
    # 略小、需要补 B 才可行，补出来的方案可能反而比"多放几根 A、不放 B"更贵。
    # 旧版先对 verified 取 min、之后才算纯 A 参照，于是把这个更便宜的可行点漏在比较之外。
    pure_a = None
    p3 = RESULTS / "p3_threshold.json"
    if p3.exists():
        n3 = json.loads(p3.read_text(encoding="utf-8"))["modes"][mode]["answer_n_a"]
        if n3:
            pure_a = estimate_p(Config(n_a=int(n3), n_b=0, orientation=mode), VERIFY_TRIALS)
            print(f"  纯 A 参照 N_A={n3} P={pure_a['p']:.4f} 成本={pure_a['cost_yuan']:.4f} 元")
            if pure_a["p"] >= TARGET and pure_a["ci_lo"] >= TARGET - 0.01:
                pure_a["note"] = "纯 A 方案（问题三 K+ 口径答案），按同一口径纳入候选比较"
                verified.append(pure_a)

    best = min(verified, key=lambda r: r["cost_yuan"])
    print(f"\n已筛选候选中成本最低：N_A={best['n_a']} N_B={best['n_b']}  "
          f"φ_A={best['phi_a']:.3%} φ_B={best['phi_b']:.2%}  "
          f"P={best['p']:.4f}  总成本={best['cost_yuan']:.4f} 元")
    out = {
        "target": TARGET, "mode": mode,
        "verification_rule": "screening: p>=0.90 and ci_lo>=0.89; only ci_lo>=0.90 confirms feasibility",
        "global_optimality_proved": False,
        "cost_per_rod_yuan": COST_A, "cost_per_sphere_yuan": COST_B,
        "grid_trials": GRID_TRIALS, "verify_trials": VERIFY_TRIALS,
        "grid": grid, "surrogate_weights": w.tolist(),
        "surrogate_max_abs_resid": float(resid),
        "candidates": cand[:12], "verified": verified, "all_probes": all_probes,
        "best": best,
        "model_scope": "spherocylinder outer-bound model K+; not a feasibility certificate for the real flat-ended cylinder",
        "best_status": "lowest-cost screened K+ candidate; real-cylinder feasibility and global optimality are unproved",
        "pure_a_reference": pure_a,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p4_cost_optimum.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
