#!/usr/bin/env python3
"""用乐观几何外界排除鲁棒 SAA 候选之下的低 A 区域。

对成本严格低于候选的预算 ``Q* - 1``，在 A<580 的每个整数列
取该预算下最大 B。介质 A 用外接球柱体，越界 B 球缺用完整球
扩大；因此每个情景中题设真实接触图都是当前图的子图。若该外界
在固定 SAA 情景上仍不达 92%，则对应列及其低成本点在该鲁棒
SAA 规则下全部排除。
"""

from __future__ import annotations

import argparse
import json
import math

from p4_cost_line_sweep import COST_UNIT, RESULTS, sweep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-limit", type=int, default=351484)
    ap.add_argument("--trials", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--target", type=float, default=0.92)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--sample-max-a", type=int, default=630)
    ap.add_argument("--exact-fallback", action="store_true", default=True,
                    help="外界未排除的分区改用平端圆柱精确距离复核")
    args = ap.parse_args()
    required = math.ceil(args.target * args.trials - 1e-12)
    # 低 A 时 B 多，分区可避免构建永远不会同时激活的棒-球边。
    partitions = [(0, 159), (160, 319), (320, 439),
                  (440, 519), (520, 559), (560, 579)]
    summaries = []
    all_excluded = True
    for lo, hi in partitions:
        hi = min(hi, args.q_limit // 567)
        if lo > hi:
            continue
        state = sweep(args.q_limit, lo, hi, args.trials, args.seed,
                      args.workers, "capsule", "wrap", args.sample_max_a)
        best = max(state["rows"], key=lambda r: r["hits"])
        excluded = best["hits"] < required
        fallback = None
        if not excluded and args.exact_fallback:
            fallback = sweep(args.q_limit, lo, hi, args.trials, args.seed,
                             args.workers, "cylinder", "wrap", args.sample_max_a)
            exact_best = max(fallback["rows"], key=lambda r: r["hits"])
            excluded = exact_best["hits"] < required
            exact_dst = RESULTS / f"p4_low_a_{lo}_{hi}_exact.json"
            exact_dst.write_text(json.dumps(fallback, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        all_excluded &= excluded
        dst = RESULTS / f"p4_low_a_{lo}_{hi}_outer.json"
        dst.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append({"a_range": [lo, hi], "outer_best": best,
                          "exact_fallback_best": (None if fallback is None else
                              max(fallback["rows"], key=lambda r: r["hits"])),
                          "all_columns_below_saa_target": excluded,
                          "details_file": dst.name})
        shown = (best if fallback is None else
                 max(fallback["rows"], key=lambda r: r["hits"]))
        print(f"A={lo:3d}..{hi:3d}: best=({shown['n_a']},{shown['n_b']}) "
              f"hits={shown['hits']}/{args.trials} "
              f"{'EXCLUDED' if excluded else 'UNRESOLVED'}", flush=True)
    out = {
        "scope": "all integer columns A<580 on Q<=candidate-1",
        "geometry_outer_bound": "capsule rods + full wrapped spheres",
        "q_limit": args.q_limit,
        "budget_yuan": args.q_limit * COST_UNIT,
        "target": args.target,
        "trials": args.trials,
        "required_hits": required,
        "seed": args.seed,
        "sample_max_a": args.sample_max_a,
        "partitions": summaries,
        "all_low_a_columns_excluded": all_excluded,
    }
    (RESULTS / "p4_low_a_outer_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("低 A 区域结论：" + ("全部排除" if all_excluded else "存在未排除分区"))
    return 0 if all_excluded else 2


if __name__ == "__main__":
    raise SystemExit(main())
