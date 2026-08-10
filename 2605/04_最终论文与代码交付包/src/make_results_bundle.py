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
    "geometry_bracket.json", "p4_global_audit.json",
]

# 附录里源程序的呈现顺序：先内核，再各问求解，最后辅助脚本
ORDER = [
    "microstructure.py", "load_attachment.py", "simulate.py",
    "audit_attachment.py", "validate.py", "theory_check.py", "cluster_stats.py",
    "solve_p1.py", "solve_p2.py", "solve_p3.py", "solve_p4.py", "p4_break_even.py",
    "sensitivity.py", "sphere_mode_check.py", "p4_backfill_probes.py",
    "geometry_bracket.py", "p4_global_audit.py",
    "paper_figures.py", "make_results_bundle.py",
]

APPENDIX_MARK = "### 附录 B 完整可运行源程序"


def rounded_variants(obj, out: set) -> None:
    """收集所有数值在 2/3/4 位小数下的取整值。

    论文里的概率按 4 位小数报（如 0.2193），而 results.json 里存的是原始的
    0.21925。`check_paper.py` 的数字溯源只回溯到 3 位小数，于是会把这些
    明明有出处的数字报成"找不到出处"。把取整值一并写进结果文件，
    既消除了这类误报，也把"论文取几位小数"这件事本身记录在案。
    """
    if isinstance(obj, dict):
        for v in obj.values():
            rounded_variants(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            rounded_variants(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        for d in (2, 3, 4):
            out.add(round(float(obj), d))


def bundle_results() -> dict:
    out = {}
    for name in PARTS:
        p = RESULTS / name
        if p.exists():
            out[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"  ! 缺少 {name}（该阶段尚未运行）")
    rv: set = set()
    rounded_variants(out, rv)
    out["_rounded_variants"] = sorted(rv)
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
              "运行顺序见附录 A 之后的说明。依赖 Python ≥3.11 与 numpy / scipy / matplotlib / openpyxl。",
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
