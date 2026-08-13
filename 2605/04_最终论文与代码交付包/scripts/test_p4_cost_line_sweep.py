#!/usr/bin/env python3
"""问题四全局成本线求解器的结构回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microstructure import COST_A, COST_B
from p4_cost_line_sweep import COST_UNIT, one_scenario


def scenario(q: int, seed: int, sphere_mode: str) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([20260826, seed]))
    return one_scenario(q, 604, min(616, q // 567), 630, rng,
                        "cylinder", sphere_mode)


def main() -> int:
    if abs(COST_A / COST_UNIT - 567) > 1e-10 or abs(COST_B / COST_UNIT - 64) > 1e-10:
        raise AssertionError("成本不能用 567:64 的整数刻度表示")
    print("成本整数化：c_A=567u, c_B=64u")

    q_lo, q_hi = 346830, 347594
    for i in range(8):
        lo = scenario(q_lo, i, "wrap")
        hi = scenario(q_hi, i, "wrap")
        # 低预算的查询列是高预算列的前缀；同一 A 列增加预算
        # 只会多激活 B，导通事件不得从 1 变 0。
        if np.any(hi[:len(lo)] < lo):
            raise AssertionError(f"预算单调性违例：scenario={i}")

        outer = scenario(q_hi, i, "wrap")
        inner = scenario(q_hi, i, "discard")
        if np.any(inner > outer):
            raise AssertionError(f"球缺内外界违例：scenario={i}")
    print("跨预算单调性：8/8 通过")
    print("球缺包含关系：8/8 通过（discard <= true <= wrap）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
