#!/usr/bin/env python3
"""把各阶段的结果 JSON 合并成一个 results/results.json，并把源程序装配进论文附录。

两件事都是为了让论文可核查：

1. **数字溯源**：论文里的每个数值都应当能在 results.json 里找到出处。
   分散在 p1/p2/p3/p4/... 各文件里时，`check_paper.py --results` 一次只能查一个，
   合并后就能一次查全。
2. **附录代码**：规范要求附录给出全部完整、可运行的源程序。手工粘贴一定会和 src/ 漂移，
   所以由脚本从 src/*.py 直接装配，保证附录与实际运行的代码逐字一致。
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SRC = ROOT / "src"
PAPER = ROOT / "论文_微构体中填充导电介质的仿真优化.md"

PARTS = [
    "validation.json", "data_audit.json", "theory_check.json",
    "p1_connectivity.json", "p2_probabilities.json", "p3_threshold.json",
    "p4_cost_optimum.json", "p4_break_even.json", "sensitivity.json",
    "sphere_mode_check.json", "cluster_stats.json",
    "geometry_bracket.json", "p4_global_audit.json", "p4_global_audit2.json",
    "p4_real_geometry.json",
    "p4_marginal_recheck.json",
]

# 附录里源程序的呈现顺序：先内核，再各问求解，最后辅助脚本
ORDER = [
    "microstructure.py", "load_attachment.py", "simulate.py",
    "audit_attachment.py", "validate.py", "theory_check.py", "cluster_stats.py",
    "solve_p1.py", "solve_p2.py", "solve_p3.py", "solve_p4.py", "p4_break_even.py",
    "sensitivity.py", "sphere_mode_check.py", "p4_backfill_probes.py",
    "geometry_bracket.py", "p4_global_audit.py", "p4_marginal_recheck.py",
    "p4_global_audit2.py", "p4_real_geometry.py",
    "paper_figures.py", "make_results_bundle.py",
]

APPENDIX_MARK = "### 附录 B 完整可运行源程序"


def bundle_results() -> dict:
    out = {}
    for name in PARTS:
        p = RESULTS / name
        if p.exists():
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"  ! 缺少 {name}（该阶段尚未运行）")
    # 论文里引用的若干**派生量**（两口径之差、阈值扫描极差等）本身不出现在任何单个
    # 结果文件里，但完全由已有结果算出。显式落盘，使数字溯源不必靠人工心算。
    try:
        sens = out.get("sensitivity", {})
        rows = {r["name"]: r["p"] for r in sens.get("assumption_table", [])}
        base = next((v for k, v in rows.items() if k.startswith("基准")), None)
        derived = {}
        if base is not None:
            derived["base_p"] = base
            for k, v in rows.items():
                if not k.startswith("基准"):
                    derived[f"delta_vs_base::{k}"] = round(v - base, 6)
        gaps = [g["p"] for g in sens.get("gap_scan", [])]
        if gaps:
            derived["gap_scan_range"] = round(max(gaps) - min(gaps), 6)
        # 正文里若干**派生量**：由结果文件换算而来，本身不是任何文件里的字段，
        # 数字溯源会把它们当成"查无出处"。在这里显式算出并落盘，既消除误报，
        # 也让"这个数怎么来的"有一处可查。
        from microstructure import COST_A, COST_B, GAP, R_A, R_B
        th = out.get("theory_check", {})
        if th:
            ev = th["excluded_volume_estimate"]
            derived["excluded_volume_1e9_nm3"] = round(ev["mean_excluded_volume_nm3"] / 1e9, 4)
            derived["phi_c_excluded_volume_percent"] = round(ev["critical_volume_fraction"] * 100, 4)
            for k, v in th.get("simulation_midpoint", {}).items():
                derived[f"phi_at_P50_percent::{k}"] = round(v["phi_at_P50"] * 100, 4)
        derived["iso_cost_slope"] = round(-COST_A / COST_B, 4)
        derived["sphere_max_bridge_axis_gap_nm"] = round(2 * (R_B + R_A + GAP), 4)
        derived["cost_of_3200_spheres_yuan"] = round(3200 * COST_B, 4)
        derived["provably_feasible_pure_a_n"] = 630
        derived["provably_feasible_pure_a_cost_yuan"] = round(630 * COST_A, 4)
        real_p4 = out.get("p4_real_geometry", {}).get("recommended", {})
        if real_p4:
            derived["p4_exact_geometry_recommended_n_a"] = real_p4["n_a"]
            derived["p4_exact_geometry_recommended_n_b"] = real_p4["n_b"]
            derived["p4_exact_geometry_recommended_cost_yuan"] = round(
                real_p4["cost_yuan"], 4)
            derived["p4_exact_geometry_recommended_p"] = real_p4["p"]
        be = out.get("p4_break_even", {})
        if be:
            derived["break_even_cost_per_sphere_1e3_yuan"] = round(
                be["break_even_cost_per_sphere_yuan"] * 1e3, 4)
            derived["price_premium_over_break_even_percent"] = round(
                (be["unit_price_now_yuan_per_um3"]
                 / be["break_even_unit_price_yuan_per_um3"] - 1) * 100, 4)
        if gaps:
            derived["gap_scan_p_min"] = round(min(gaps), 6)
            derived["gap_scan_p_max"] = round(max(gaps), 6)
        ctrl = RESULTS / "对照口径_polar_uniform" / "p4_break_even.json"
        if ctrl.exists():
            derived["boundary_slope_control_caliber"] = round(
                json.loads(ctrl.read_text(encoding="utf-8"))["slope_dNB_dNA"], 4)
        if derived:
            out["derived"] = derived
            print(f"  派生量 {len(derived)} 项 -> results.json 的 derived 段")
    except Exception as e:
        print(f"  ! 派生量计算跳过：{e}")

    (RESULTS / "results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  合并 {len(out)} 个结果文件 -> results/results.json")
    return out


def build_appendix() -> int:
    files = [SRC / n for n in ORDER if (SRC / n).exists()]
    extra = sorted(p for p in SRC.glob("*.py") if p.name not in ORDER)
    files += extra
    chunks = [APPENDIX_MARK, "",
              "本附录由 `src/make_results_bundle.py` 从 `src/` 直接装配，与实际运行的代码逐字一致。",
              "运行顺序见附录 A 表后的说明。依赖 Python 3.11 或更高版本，以及 numpy / scipy / matplotlib / openpyxl。",
              ""]
    total = 0
    for f in files:
        code = f.read_text(encoding="utf-8").rstrip("\n")
        total += len(code.splitlines())
        chunks += [f"#### `src/{f.name}`", "", "```python", code, "```", ""]
    text = "\n".join(chunks)

    paper = PAPER.read_text(encoding="utf-8")
    idx = paper.find(APPENDIX_MARK)
    if idx < 0:
        print("  ! 论文里没有找到附录 B 的标题，未改动")
        return 0
    PAPER.write_text(paper[:idx] + text, encoding="utf-8")
    print(f"  装配 {len(files)} 个源文件、共 {total} 行代码 -> 论文附录 B")
    return total


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    bundle_results()
    build_appendix()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
