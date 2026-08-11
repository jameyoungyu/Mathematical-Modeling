#!/usr/bin/env python3
"""回归检查：随机流必须既与并行度无关，又与已发布的 results/ 逐位一致。

两件事都要查，只查第一件是不够的：把随机流数固定成任意常数都能通过"并行度不变结果"，
但只有取到与生成 results/ 时相同的流数，代码才跑得出已发布的那些数。
本仓库的 results/ 是在 4 条流下生成的（取 8 条时同一配置得 475 而非 462）。
"""

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

    # 第二件事：与已发布的 results/ 对齐。抽两个成本可控的点。
    import json
    res = ROOT / "results"
    coarse = json.loads((res / "p3_threshold.json").read_text(encoding="utf-8"))
    ref = coarse["modes"]["isotropic"]["coarse"][0]
    got = estimate_p(Config(n_a=ref["n_a"]), trials=ref["trials"])
    print(f'p3 粗扫首点 N_A={ref["n_a"]} T={ref["trials"]}: '
          f'hits={got["hits"]} vs results 记录 {ref["hits"]}')
    if got["hits"] != ref["hits"]:
        raise AssertionError(
            f'与已发布结果不一致：{got["hits"]} != {ref["hits"]}。'
            f'多半是 N_RANDOM_STREAMS 被改动了——它必须与生成 results/ 时一致。')
    print("PASS: 与 results/p3_threshold.json 逐位一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
