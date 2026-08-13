#!/usr/bin/env python3
"""问题四：平端圆柱真实几何下的成本线审计（可断点续跑）。

本脚本不再用球柱体上下界代替介质 A，而调用 ``rod_geometry='cylinder'``：

* 圆柱—圆柱距离由支撑映射 Gilbert/GJK 算法计算；
* 圆柱—球距离使用点到有限平端圆柱的解析公式；
* 圆柱—带电面接触使用圆柱在 X 方向的精确支撑值。

对给定成本 C 和根数 N_A，只需检查预算线上最大的整数 N_B。若该点不达标，
同一列所有成本不超过 C 的点都不达标。命令行可一次指定一条成本线和若干 N_A，
结果持续写入 ``results/p4_real_geometry.json``，适合把算力集中到阈值附近。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from microstructure import COST_A, COST_B
from simulate import Config, estimate_p, wilson

RESULTS = Path(__file__).resolve().parents[1] / "results"
DST = RESULTS / "p4_real_geometry.json"
TARGET = 0.90
Z99 = 2.5758293035489004


def budget_nb(cost: float, n_a: int, *, strict: bool = False) -> int:
    """成本不超过（或严格低于）``cost`` 时允许的最大球数。"""
    x = (cost - n_a * COST_A) / COST_B
    return math.ceil(x) - 1 if strict else math.floor(x + 1e-12)


def load() -> dict:
    if DST.exists():
        state = json.loads(DST.read_text(encoding="utf-8"))
        # 早期结果文件只保存 screening/finalists，没有 probes。
        # 新的命令行增量审计应能在这类文件上继续，而不是
        # 因 KeyError 中断。
        state.setdefault("probes", {})
        return state
    return {
        "target": TARGET,
        "geometry": "finite flat-ended cylinder after the stated centerline truncation rule",
        "distance_kernel": "support-mapping Gilbert/GJK + analytic point-cylinder distance",
        "decision_rule": "99% Wilson: lo>=0.90 confirms; hi<0.90 excludes; otherwise undecided",
        "probes": {},
    }


def save(state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def run_probe(n_a: int, n_b: int, trials: int, seed: int, workers: int) -> dict:
    t0 = time.time()
    r = estimate_p(Config(n_a=n_a, n_b=n_b, rod_geometry="cylinder"),
                   trials, seed=seed, workers=workers)
    lo99, hi99 = wilson(r["hits"], trials, z=Z99)
    return {
        **r,
        "ci99_lo": lo99,
        "ci99_hi": hi99,
        "status": ("confirmed" if lo99 >= TARGET else
                   "excluded" if hi99 < TARGET else "undecided"),
        "seconds": round(time.time() - t0, 1),
    }


def summarize(state: dict) -> None:
    groups: dict[str, list[dict]] = {}
    for r in state["probes"].values():
        if "audit_cost_yuan" in r:
            groups.setdefault(f"{r['audit_cost_yuan']:.4f}", []).append(r)
    state["cost_lines"] = {}
    for key, rows in sorted(groups.items(), key=lambda kv: float(kv[0])):
        best = max(rows, key=lambda r: r["p"])
        state["cost_lines"][key] = {
            "columns_done": len(rows),
            "n_a_min": min(r["n_a"] for r in rows),
            "n_a_max": max(r["n_a"] for r in rows),
            "best_observed": {k: best[k] for k in
                              ("n_a", "n_b", "cost_yuan", "p", "ci99_lo", "ci99_hi", "status")},
            "all_sampled_columns_excluded": all(r["status"] == "excluded" for r in rows),
            "any_confirmed": any(r["status"] == "confirmed" for r in rows),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost", type=float, required=True, help="要审计的总成本线（元）")
    ap.add_argument("--a", type=int, nargs="+", required=True, help="要检查的 N_A 列")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260815)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--strict", action="store_true", help="检查成本严格低于给定值")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    state = load()
    for n_a in args.a:
        n_b = budget_nb(args.cost, n_a, strict=args.strict)
        if n_b < 0:
            continue
        key = f"C={args.cost:.6f}|A={n_a}|B={n_b}|T={args.trials}|seed={args.seed}"
        if key not in state["probes"] or args.force:
            print(f"({n_a},{n_b}) C={n_a*COST_A+n_b*COST_B:.4f} T={args.trials}", flush=True)
            r = run_probe(n_a, n_b, args.trials, args.seed, args.workers)
            r["audit_cost_yuan"] = args.cost
            r["strict_cost"] = args.strict
            state["probes"][key] = r
            summarize(state); save(state)
        r = state["probes"][key]
        print(f"  P={r['p']:.4f} 99%=[{r['ci99_lo']:.4f},{r['ci99_hi']:.4f}] "
              f"{r['status']} ({r['seconds']:.1f}s)", flush=True)
    summarize(state); save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
