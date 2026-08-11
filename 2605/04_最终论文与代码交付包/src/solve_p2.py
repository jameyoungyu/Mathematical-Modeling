#!/usr/bin/env python3
"""问题二：只填介质 A 时，体积分数 0.50%/0.60%/0.70%/1.00% 的导通概率。

对每个体积分数做 T 次独立仿真，导通频率即概率估计，附 Wilson 95% 置信区间。
主口径用球面均匀方向（isotropic），同时给出附件标定方向（polar_uniform）的灵敏度对照——
这两个分布给出的概率差别可达 20 个百分点，是全题最敏感的建模选择。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from simulate import Config, estimate_p, n_rods_for_phi

RESULTS = Path(__file__).resolve().parents[1] / "results"
FRACTIONS = [0.0050, 0.0060, 0.0070, 0.0100]
TRIALS = 8000


def main() -> int:
    out = {"trials_per_point": TRIALS, "fractions": FRACTIONS, "runs": {}}
    for mode in ("polar_uniform", "isotropic"):
        rows = []
        for phi in FRACTIONS:
            n = n_rods_for_phi(phi)
            t0 = time.time()
            r = estimate_p(Config(n_a=n, orientation=mode), TRIALS)
            r["phi_nominal"] = phi
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"{mode:14s} φ={phi:.2%} N_A={n:4d}  P={r['p']:.4f} "
                  f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]  ({r['seconds']}s)")
        out["runs"][mode] = rows

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p2_probabilities.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
