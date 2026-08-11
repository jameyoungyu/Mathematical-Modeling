#!/usr/bin/env python3
"""验证判据阈值通过 Config 传入，且不依赖 fork 继承模块全局。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulate import Config, estimate_p  # noqa: E402


def main() -> int:
    by_gap: dict[float, int] = {}
    for gap in (0.9, 2.7):
        hits = []
        for workers in (1, 2, 4):
            result = estimate_p(
                Config(n_a=495, gap=gap), 80, seed=24680, workers=workers
            )
            hits.append(result["hits"])
        print(f"gap={gap}: workers=1/2/4 -> hits={hits}")
        if len(set(hits)) != 1:
            raise AssertionError(f"gap={gap} 时结果随 worker 改变：{hits}")
        by_gap[gap] = hits[0]

    print("PASS: gap 由 Config 显式传入，spawn/fork 均无需继承模块全局")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
