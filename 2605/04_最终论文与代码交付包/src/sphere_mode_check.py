#!/usr/bin/env python3
"""介质 B 越界处理口径的对照（假设 D8）。

球被边界切开后，碎片其实是球缺。实现里按"整球置于其所在周期像"近似
（`sphere_mode="wrap"`），在距壁 200 nm 的薄层内会略微高估它的跨接能力。
对照口径是把球心限制在 $[-(L/2-R),\\,L/2-R]$，保证整颗球都在盒内、不产生碎片
（`sphere_mode="inside"`）。

只在含介质 B 的配置上比较才有意义；问题四的最优解 $N_B=0$，这条假设对最终答案
没有影响，但它影响可行边界的位置，因而影响搜索过程，所以仍要量化。

结果写入 results/sphere_mode_check.json。
"""

from __future__ import annotations

import json
from pathlib import Path

from simulate import Config, estimate_p

RESULTS = Path(__file__).resolve().parents[1] / "results"
CASES = [(440, 1200), (500, 700)]
TRIALS = 4000


def main() -> int:
    rows = []
    for n_a, n_b in CASES:
        pair = {}
        for mode in ("wrap", "inside"):
            r = estimate_p(Config(n_a=n_a, n_b=n_b, sphere_mode=mode), TRIALS)
            pair[mode] = r
            print(f"  N_A={n_a} N_B={n_b} sphere_mode={mode:6s} "
                  f"P={r['p']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")
        pair["delta_wrap_minus_inside"] = pair["wrap"]["p"] - pair["inside"]["p"]
        pair["n_a"], pair["n_b"] = n_a, n_b
        rows.append(pair)
        print(f"    差值（wrap − inside）= {pair['delta_wrap_minus_inside']:+.4f}")

    out = {"trials": TRIALS, "cases": rows,
           "note": "问题四最优解 N_B=0，该口径不影响最终答案，仅影响可行边界位置。"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sphere_mode_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
