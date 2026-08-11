#!/usr/bin/env python3
"""复现结论 2：check_paper.py 的身份信息检查存在坐标系错位，会静默漏报真实泄漏。

`check_identity` 把正则跑在 `strip_code_blocks(text)` 上——围栏代码块被**删除**，
其后所有字符的偏移整体前移；却把"参考文献里的作者名、出版社不算泄漏"这个跳过区间
`ref_span` 算在**原始** `text` 上。两套坐标系不一致，后果有两个：

  (a) 报告的行号一律偏大（无条件发生）；
  (b) 位置满足 `pos_body(命中) ∈ ref_span_text` 的**真实**命中被当成"参考文献里的
      正常著录"而丢弃（有条件发生）。

(b) 的触发条件可以写死：设命中在原文的位置为 P，其前方围栏代码总长为 L，则
`pos_body = P - L`，于是当

        ref_start ≤ P - L < ref_end

时命中被吞掉。参考文献之后还有附录、而正文里有一段中等长度的代码清单——这正是
CUMCM 论文的常见布局，落进窗口毫不罕见。

本脚本用仓库里 2604 的真实终稿做对照：在参考文献之后的附录里植入一处校名泄漏，
再把一段可调长度的代码清单放进正文（参考文献之前），扫描长度 L 观察漏报窗口。

用法：python3 measure_identity_desync.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = (HERE.parents[1] / "2604" / "04_最终论文与代码交付包"
         / "10_修订后完整论文_终稿.md")

LEAK = "张三就读于清华大学"
LEAK_ANCHOR = "## 附录C"        # 参考文献之后的附录，泄漏植入在它前面
CODE_ANCHOR = "## 8 模型检验"   # 参考文献之前的正文，代码清单插在它前面
LENGTHS = [0, 400, 850, 1700, 2650, 3130, 3600, 5500]


def load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_paper_asreceived", HERE / "check_paper_asreceived.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def identity_findings(mod, text: str) -> list[str]:
    report = mod.Report()
    sections = mod.split_sections(mod.mask_code_blocks(text))
    mod.check_identity(text, sections, report)
    return [f.message for f in report.findings]


def code_block(target_len: int) -> str:
    """一段无害的、长度约为 target_len 的 Python 代码清单。"""
    if target_len <= 0:
        return ""
    line = "x = {0}  # 计算第 {0} 步\n"
    n = max(1, target_len // len(line.format(999)))
    return "\n```python\n" + "".join(line.format(i) for i in range(n)) + "```\n"


def main() -> int:
    if not PAPER.exists():
        print(f"找不到 {PAPER}")
        return 2
    mod = load_checker()
    raw = PAPER.read_text(encoding="utf-8")
    for anchor in (LEAK_ANCHOR, CODE_ANCHOR):
        if anchor not in raw:
            print(f"论文结构已变，找不到锚点 {anchor}")
            return 2

    base = raw.replace(LEAK_ANCHOR, f"{LEAK}，本节由其独立完成。\n\n{LEAK_ANCHOR}", 1)
    print(f"论文：{PAPER.name}（{len(raw)} 字符）")
    print(f"植入泄漏：「{LEAK}」，位置在 {LEAK_ANCHOR}（参考文献之后的附录内）")
    print(f"自变量：在 {CODE_ANCHOR}（参考文献之前）插入一段代码清单，扫描其长度\n")
    print(f"  {'代码清单长度':>12}   {'身份信息命中':>10}   植入的泄漏是否被发现")
    print("  " + "-" * 52)

    missed = []
    for want in LENGTHS:
        blk = code_block(want)
        text = base.replace(CODE_ANCHOR, blk + CODE_ANCHOR, 1)
        found = [s for s in identity_findings(mod, text) if LEAK in s]
        hits = identity_findings(mod, text)
        flag = "是" if found else "★ 否（漏报）"
        print(f"  {len(blk):>12d}   {len(hits):>10d}   {flag}")
        if not found:
            missed.append(len(blk))

    print()
    if missed:
        print(f"结论：复现成功。代码清单长度落在 {min(missed)}–{max(missed)} 字符区间时，"
              "附录里的真实校名泄漏被静默吞掉；")
        print("      区间之外则能报出——是否漏报取决于论文各部分的字符数比例，"
              "而不取决于泄漏本身，这正是坐标系错位的特征。")
    else:
        print("结论：本次扫描未落入漏报窗口（该版本可能已修复，或论文结构变化）。")
    print("\n修法：check_identity 里把 strip_code_blocks 换成同文件已有的 "
          "mask_code_blocks")
    print("      （等长空白替换，字符偏移不变），(a) 与 (b) 一并解决。")
    return 0 if missed else 1


if __name__ == "__main__":
    raise SystemExit(main())
