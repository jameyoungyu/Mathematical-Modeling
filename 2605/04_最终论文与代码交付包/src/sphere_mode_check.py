#!/usr/bin/env python3
"""介质 B 越界处理口径的对照（假设 D8）。

球被边界切开后，碎片其实是球缺。实现里的 `sphere_mode="wrap"` 用整球近似每个球缺；
`sphere_mode="inside"` 则改变球心采样域，使整球不越界。两者都不是题面截断规则的精确实现，
只能作为近似口径的包络式对照，不能据此声称球边界误差已被严格夹住。

只在含介质 B 的配置上比较才有意义。早期 `wrap`/`inside` 对照只用于灵敏度解释；
最终推荐方案含 8 颗 B，故其证书另用严格包含的 `discard` 内界与 `wrap` 外界，
不能用本脚本的点估计代替可行性确认。

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
           "note": ("wrap 与 inside 都不是题面球缺截断的精确实现；本结果只作灵敏度对照。"
                    "最终 (619,8) 的确认改用 discard 内界，低成本排除使用 wrap 外界。")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sphere_mode_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
