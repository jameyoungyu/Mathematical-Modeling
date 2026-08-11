#!/usr/bin/env python3
"""灵敏度与稳健性分析。

要回答的不是"模型是否鲁棒"这种没法证伪的话，而是三个具体问题：

S1 判据阈值 δ=1.8 nm 若变动 ±50%，导通概率变多少？（δ 是题目给死的，做这条是为了说明
   结论不是卡在阈值的某个巧合上）
S2 各条建模口径分别把导通概率推到哪里？逐条列出，包括被否掉的"碎片粘合"口径——
   它把 P 顶到 1.0，正是它被否掉的原因。
S3 蒙特卡洛样本量够不够？给出 P 随试验次数的收敛轨迹。

固定在 φ=0.70%（处于渗流跃变最陡的位置，对任何扰动都最敏感）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from microstructure import CONTROL_ORIENTATION
from simulate import Config, estimate_p, n_rods_for_phi, wilson

RESULTS = Path(__file__).resolve().parents[1] / "results"
PHI = 0.0070
TRIALS = 4000


def with_gap(gap: float, n_a: int, trials: int) -> dict:
    # gap 随 Config 显式传给 worker；spawn/fork/forkserver 下结果一致。
    r = estimate_p(Config(n_a=n_a, gap=gap), trials)
    r["gap"] = gap
    return r


def main() -> int:
    n_a = n_rods_for_phi(PHI)
    out = {"phi": PHI, "n_a": n_a, "trials": TRIALS, "gap_scan": [], "assumption_table": []}

    print(f"φ={PHI:.2%}, N_A={n_a}")
    print("--- S1 判据阈值 δ ---")
    for gap in (0.9, 1.35, 1.8, 2.25, 2.7):
        # 样本量减半以控制时间。
        r = with_gap(gap, n_a, TRIALS // 2)
        out["gap_scan"].append(r)
        print(f"  δ={gap:4.2f} nm  P={r['p']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")

    print("--- S2 建模口径 ---")
    variants = [
        ("基准：球面均匀方向 + 碎片各自独立", Config(n_a=n_a)),
        ("方向改为附件标定的极角均匀分布", Config(n_a=n_a, orientation=CONTROL_ORIENTATION)),
        ("碎片粘合为同一导体（已否决）", Config(n_a=n_a, bond_fragments=True)),
    ]
    for name, cfg in variants:
        r = estimate_p(cfg, TRIALS)
        r["name"] = name
        out["assumption_table"].append(r)
        print(f"  {name}: P={r['p']:.4f} [{r['ci_lo']:.4f},{r['ci_hi']:.4f}]")

    print("--- S3 样本量收敛 ---")
    conv = []
    for t in (250, 500, 1000, 2000, 4000, 8000):
        r = estimate_p(Config(n_a=n_a), t, seed=13579)
        conv.append({"trials": t, "p": r["p"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                     "half_width": r["half_width"]})
        print(f"  T={t:5d}  P={r['p']:.4f}  ±{r['half_width']:.4f}")
    out["convergence"] = conv

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
