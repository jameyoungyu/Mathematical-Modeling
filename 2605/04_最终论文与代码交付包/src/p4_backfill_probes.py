#!/usr/bin/env python3
"""把问题四阶段 C 中**未通过**的探测点补记进结果文件。

`solve_p4.py` 最初只把通过验证的点写进 `verified`，未通过的探测点只留在运行日志里。
但论文表 13 要给出"哪些候选没通过、差多少"——只写通过的点会让读者无法判断
最优解是否真的被比较过。脚本已改为记录全部探测点（`all_probes`），
本文件用于把当时那几个未通过的点补算回来。

`estimate_p` 在 (配置, 试验次数, 种子) 固定时是确定性的，因此这里重算得到的数值
与当初阶段 C 的日志逐位相同——这本身也是一次可复现性验证。
"""

from __future__ import annotations

import json
from pathlib import Path

from simulate import Config, estimate_p

RESULTS = Path(__file__).resolve().parents[1] / "results"
P4 = RESULTS / "p4_cost_optimum.json"

# 阶段 C 日志中未通过 P≥0.90 的探测点
REJECTED = [(512, 408), (472, 826), (432, 1301)]


def main() -> int:
    if not P4.exists():
        print("先运行 solve_p4.py")
        return 1
    data = json.loads(P4.read_text(encoding="utf-8"))
    trials = data["verify_trials"]
    mode = data["mode"]

    probes = []
    for n_a, n_b in REJECTED:
        r = estimate_p(Config(n_a=n_a, n_b=n_b, orientation=mode), trials)
        r["accepted"] = bool(r["p"] >= data["target"])
        probes.append(r)
        print(f"  N_A={n_a:4d} N_B={n_b:5d} P={r['p']:.4f} "
              f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] 成本={r['cost_yuan']:.4f} 元 "
              f"-> {'通过' if r['accepted'] else '未通过'}")

    data["rejected_probes"] = probes
    existing = {(p["n_a"], p["n_b"]) for p in data.get("verified", [])}
    data["all_probes"] = sorted(
        data.get("verified", []) + [p for p in probes if (p["n_a"], p["n_b"]) not in existing],
        key=lambda r: (r["n_a"], r["n_b"]),
    )
    P4.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已补记 {len(probes)} 个未通过的探测点 -> {P4.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
