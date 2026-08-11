#!/usr/bin/env python3
"""回归检查：改变 worker 数不能改变蒙特卡洛随机流与命中数。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulate import Config, estimate_p  # noqa: E402


def main() -> int:
    rows = []
    for workers in (1, 2, 4, 8):
        r = estimate_p(Config(n_a=354), trials=80, workers=workers)
        rows.append((workers, r["hits"], r["p"]))
        print(f"workers={workers}: hits={r['hits']}, p={r['p']:.6f}")

    hits = {row[1] for row in rows}
    if len(hits) != 1:
        raise AssertionError(f"并行度改变了随机结果：{rows}")
    print("PASS: 1/2/4/8 个 worker 得到逐位一致的结果")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
