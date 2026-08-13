#!/usr/bin/env python3
"""问题四：在固定共同情景上二分搜索最小整数成本刻度。

本脚本调用 ``p4_cost_line_sweep.sweep``，以完全相同的随机情景
评估不同预算。因为预算增加只会增加可用介质前缀，“成本线上
是否存在经验导通率不低于 0.90 的整数点”对预算单调，可用
二分搜索。输出是固定有限情景的 SAA 全局最优候选，还须用独立
样本做置信下界确认。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from p4_cost_line_sweep import COST_UNIT, RESULTS, sweep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=int, default=346830,
                    help="已知经验不可行的 Q")
    ap.add_argument("--hi", type=int, default=347594,
                    help="已知经验可行的 Q")
    ap.add_argument("--a-min", type=int, default=580)
    ap.add_argument("--a-max", type=int, default=620)
    ap.add_argument("--sample-max-a", type=int, default=630)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--saa-target", type=float, default=0.90,
                    help="SAA 阶段的经验概率约束；可设 0.91 留置信缓冲")
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sphere-mode", choices=("wrap", "discard"), default="wrap")
    args = ap.parse_args()
    required_hits = math.ceil(args.saa_target * args.trials - 1e-12)
    cache: dict[int, dict] = {}

    def evaluate(q: int) -> tuple[bool, dict]:
        if q not in cache:
            query_a_max = min(args.a_max, q // 567)
            dst = RESULTS / f"p4_line_{q}_cylinder_{args.sphere_mode}.json"
            state = None
            if dst.exists():
                old = json.loads(dst.read_text(encoding="utf-8"))
                if (old.get("trials") == args.trials and old.get("seed") == args.seed
                        and old.get("a_min") == args.a_min
                        and old.get("a_max") == query_a_max
                        and old.get("sample_max_a") == args.sample_max_a):
                    state = old
            if state is None:
                state = sweep(q, args.a_min, query_a_max,
                              args.trials, args.seed, args.workers, "cylinder",
                              args.sphere_mode, args.sample_max_a)
            state["required_hits"] = required_hits
            dst.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            cache[q] = state
        state = cache[q]
        best = max(state["rows"], key=lambda r: (r["hits"], -r["cost_tick"]))
        ok = best["hits"] >= required_hits
        print(f"Q={q} C={q*COST_UNIT:.6f}  best=({best['n_a']},{best['n_b']}) "
              f"hits={best['hits']}/{args.trials}  {'FEAS' if ok else 'INFEAS'}",
              flush=True)
        return ok, best

    lo, hi = args.lo, args.hi
    lo_ok, _ = evaluate(lo)
    hi_ok, _ = evaluate(hi)
    if lo_ok or not hi_ok:
        raise RuntimeError("二分端点不满足 lo=不可行、hi=可行")
    history = []
    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, best = evaluate(mid)
        history.append({"q_tick": mid, "feasible": ok, "best": best})
        if ok:
            hi = mid
        else:
            lo = mid
    _, best = evaluate(hi)
    out = {
        "scope": "SAA global search on fixed common scenarios",
        "target": args.saa_target,
        "trials": args.trials,
        "seed": args.seed,
        "sphere_mode": args.sphere_mode,
        "a_range": [args.a_min, args.a_max],
        "sample_max_a": args.sample_max_a,
        "last_infeasible_q": lo,
        "first_feasible_q": hi,
        "first_feasible_budget_yuan": hi * COST_UNIT,
        "recommended": best,
        "history": history,
        "claim_limit": "finite-scenario SAA optimum; independent confidence validation required",
    }
    tag = f"{args.saa_target:.3f}".replace(".", "p")
    dst = RESULTS / f"p4_saa_search_{tag}_{args.sphere_mode}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SAA 最低成本刻度 Q={hi}, 候选=({best['n_a']},{best['n_b']}), "
          f"实际成本={best['cost_yuan']:.6f} 元")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
