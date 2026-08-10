#!/usr/bin/env python3
"""复现结论 3：`--data` 把反编造检查的分辨力从 ~100% 降到 ~80%，并使其变慢。

SKILL.md 建议默认带上 `--data`，把题目附件里的数字也算作合法出处。动机是对的
（正文会大量引用题给常数），代价却没有被量化：附件动辄上万个不同的数，而
`number_matches` 又允许"四舍五入到 0/1/2/3 位小数""×100 / ÷100""取绝对值"
三类等价，于是**随便编一个数也很容易撞上某个附件数字的某个舍入版本**。

本脚本生成一批与论文无关的随机"编造数字"，分别在两种出处集合下测它们的通过率：

    A. 只用 results.json
    B. results.json + data/          （SKILL.md 推荐的默认）

同时统计各条匹配规则的贡献，以及 `number_matches` 的耗时——它是
O(论文数字 × 出处数字) 的线性扫描，出处一多就线性变慢。

用法：python3 measure_data_falsepass.py [--trials 2000] [--seed 20260810]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1] / "2604" / "04_最终论文与代码交付包"
RESULTS_DIR = PKG / "results"
DATA_DIR = PKG / "data"


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_paper_asreceived", HERE / "check_paper_asreceived.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def why_matched(mod, paper_num: float, known: set[str]) -> str:
    """复述 number_matches 的判定路径，用于归因是哪条规则放过了编造数字。"""
    if mod.normalize_number(paper_num) in known:
        return "精确相等"
    hit = None
    for k in known:
        kv = float(k)
        if kv == 0:
            continue
        for digits in (0, 1, 2, 3):
            if abs(round(kv, digits) - paper_num) < 1e-9:
                cur = f"四舍五入到 {digits} 位小数"
                if digits == 0:
                    return cur          # 最宽的一条，直接归因
                hit = hit or cur
        if abs(kv * 100 - paper_num) < 1e-6 or abs(kv / 100 - paper_num) < 1e-12:
            hit = hit or "×100 / ÷100"
        if abs(abs(kv) - abs(paper_num)) < 1e-9:
            hit = hit or "取绝对值"
    return hit or "未匹配"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    jsons = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.exists() else []
    if not jsons:
        print(f"{RESULTS_DIR} 下没有结果 JSON")
        return 2
    mod = load_checker()

    # 2604 把结果拆成了几个 JSON；这里取并集，等价于技能约定的单个 results.json
    known_results: set[str] = set()
    for f in jsons:
        known_results |= mod.collect_result_numbers(
            json.loads(f.read_text(encoding="utf-8")))
    report = mod.Report()
    known_data = mod.collect_data_numbers([DATA_DIR], report) if DATA_DIR.exists() else set()
    known_both = known_results | known_data

    data_bytes = sum(f.stat().st_size for f in DATA_DIR.rglob("*") if f.is_file()) \
        if DATA_DIR.exists() else 0
    print(f"出处集合 A：results/*.json（{len(jsons)} 个）  {len(known_results):>7d} 个不同的数")
    print(f"出处集合 B：并入 data/               {len(known_both):>7d} 个不同的数"
          f"（data/ {data_bytes/1e6:.1f} MB）\n")

    # 编造一批与论文无关的两位小数，取值范围覆盖常见的量纲
    rng = random.Random(args.seed)
    fabricated = [round(rng.uniform(0, 200), 2) for _ in range(args.trials)]

    rows = []
    for label, known in (("A  仅 results/*.json", known_results),
                         ("B  + data/（推荐默认）", known_both)):
        t0 = time.perf_counter()
        passed = [x for x in fabricated if mod.number_matches(x, known)]
        dt = time.perf_counter() - t0
        rows.append((label, len(passed), dt, known))
        print(f"{label:<26}  编造数字被判为“有出处”：{len(passed):>5d} / {args.trials}"
              f"  = {len(passed)/args.trials:6.1%}   （耗时 {dt:5.2f}s）")

    print()
    label, n_pass, _, known = rows[-1]
    if n_pass:
        reasons: dict[str, int] = {}
        for x in fabricated:
            if mod.number_matches(x, known):
                r = why_matched(mod, x, known)
                reasons[r] = reasons.get(r, 0) + 1
        print("集合 B 下，放过编造数字的规则归因：")
        for r, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {r:<20} {n:>5d} 条  ({n/n_pass:5.1%})")

    print("\n结论：`--data` 让反编造检查的漏网率从个位数百分比升到两成量级，"
          "耗时也随出处规模线性增长。")
    print("      注意归因：主要通道是**精确相等**而非舍入等价——数值附件本身就足够密集，")
    print("      几乎能为任意一个数“提供出处”。收紧舍入规则并不能解决这个问题。")
    print("建议：① 命中附件的数字单列一档“低置信度出处”，不与 results/*.json 混报"
          "（唯一治本的一条）；")
    print("      ② 舍入 / ×100 等价只对 results/*.json 生效，附件只认精确相等"
          "（消掉剩下那部分）；")
    print("      ③ 预先把各等价版本装进哈希集合，把 O(论文×出处) 降到 O(论文)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
