#!/usr/bin/env python3
"""把终稿 Markdown 编译成提交用 PDF（pandoc + XeLaTeX）。

排版规范照搬竞赛论文的通行要求：A4、四边留白 ≥2.5 cm、页脚居中连续页码、
正文无目录、图题在图下方、表题在表上方。字体用 texlive 自带的 Fandol 开源字族，
不依赖任何私有字体，换台装了 texlive 的机器能编出同样的版面。

用法：
    python3 src/build_paper_pdf.py              # 完整版（含附录 B 全部源程序）
    python3 src/build_paper_pdf.py --no-code    # 去掉附录 B，便于快速预览正文版面

中间文件放 build/，产物是与论文同名的 PDF。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "论文_微构体中填充导电介质的仿真优化.md"
HEADER = ROOT / "header_paper.tex"
BUILD = ROOT / "build"
APPENDIX_B = "### 附录 B 完整可运行源程序"


# LaTeX 特殊字符（数学环境之外才需要转义）
_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex_escape(text: str) -> str:
    """转义 LaTeX 特殊字符，但保留 $...$ 行内公式原样。

    题注里既有中文和百分号，又有 $N_A$ 这样的行内公式；整体当原始 LaTeX 会被
    百分号注释掉半行，整体转义又会把公式打死。所以按 $...$ 切开分别处理。
    """
    out = []
    for i, part in enumerate(re.split(r"(\$[^$]*\$)", text)):
        if i % 2 == 1:                      # 落在 $...$ 里，原样保留
            out.append(part)
        else:
            out.append("".join(_TEX_ESC.get(ch, ch) for ch in part))
    return "".join(out)


def preprocess(text: str, include_code: bool) -> str:
    """把 Markdown 调整成适合 pandoc→LaTeX 的形态。"""
    if not include_code:
        idx = text.find(APPENDIX_B)
        if idx > 0:
            text = text[:idx] + (
                APPENDIX_B + "\n\n"
                "（预览版已省略附录 B 的源程序；完整版由 "
                "`python3 src/build_paper_pdf.py` 生成。）\n")

    lines = text.split("\n")
    out: list[str] = []
    i = 0
    img_re = re.compile(r"!\[([^\]]*)\]\((figures_paper/[^)]+)\)")
    cap_re = re.compile(r"^(图|表)\s*\d+(?:-\d+)?\s+\S")

    while i < len(lines):
        line = lines[i]
        m = img_re.match(line.strip())
        if m:
            # 向后找紧随的 "图 N …" 题注，和图打包进同一个 [H] 浮动体。
            # 分成两个块的话，LaTeX 可能在图与题注之间分页——预览版里就出现过
            # 图在第 4 页、题注在第 5 页开头的情况。
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            cap = lines[j] if j < len(lines) and cap_re.match(lines[j]) else ""
            path = (ROOT / m.group(2)).as_posix()
            out += ["", "\\begin{figure}[H]", "\\centering",
                    f"\\includegraphics[width=0.92\\linewidth]{{{path}}}"]
            if cap:
                out += ["\\par\\vspace{5pt}",
                        "{\\small\\bfseries " + tex_escape(cap) + "}"]
                i = j
            out += ["\\end{figure}", ""]
            i += 1
            continue

        if cap_re.match(line):              # 表题：居中小字加粗，不浮动
            out += ["", "\\begin{center}\\small\\bfseries "
                    + tex_escape(line) + "\\end{center}", ""]
            i += 1
            continue

        out.append(line)
        i += 1
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="编译论文 PDF")
    ap.add_argument("--no-code", action="store_true", help="省略附录 B 的源程序")
    ap.add_argument("--ai-details", action="store_true",
                    help="改为编译支撑材料《AI工具使用详情》")
    args = ap.parse_args()

    for tool in ("pandoc", "xelatex"):
        if shutil.which(tool) is None:
            sys.exit(f"缺少 {tool}；请先安装 pandoc 与 texlive-xetex/texlive-lang-chinese")

    BUILD.mkdir(parents=True, exist_ok=True)

    if args.ai_details:
        # 支撑材料里要求文件名精确为 AI工具使用详情.pdf
        src_md = ROOT / "AI工具使用详情.md"
        out_pdf = ROOT / "AI工具使用详情.pdf"
        cmd = ["pandoc", str(src_md), "--from", "markdown+pipe_tables",
               "--pdf-engine", "xelatex", "--top-level-division", "section",
               "--include-in-header", str(HEADER), "-V", "documentclass=article",
               "-V", "fontsize=12pt", "-o", str(out_pdf)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stderr[-2000:])
            sys.exit("编译失败")
        print(f"  ✓ {out_pdf.name}（{out_pdf.stat().st_size/1024:.0f} KB）")
        return 0

    staged = BUILD / "paper.md"
    staged.write_text(preprocess(MANUSCRIPT.read_text(encoding="utf-8"),
                                 include_code=not args.no_code), encoding="utf-8")

    out_pdf = ROOT / ("微构体中填充导电介质的仿真优化_预览版.pdf" if args.no_code
                      else "微构体中填充导电介质的仿真优化_终稿.pdf")
    cmd = [
        "pandoc", str(staged),
        "--from", "markdown+tex_math_dollars+pipe_tables+raw_tex",
        "--to", "latex",
        "--pdf-engine", "xelatex",
        "--top-level-division", "section",
        "--include-in-header", str(HEADER),
        "--resource-path", str(ROOT),
        "--highlight-style", "monochrome",
        "-V", "documentclass=article",
        "-V", "fontsize=12pt",
        "-o", str(out_pdf),
    ]
    print("  " + " ".join(cmd[:6]) + " …")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        (BUILD / "pandoc_error.log").write_text(res.stdout + res.stderr, encoding="utf-8")
        print(res.stderr[-3000:])
        sys.exit("编译失败，详见 build/pandoc_error.log")

    size = out_pdf.stat().st_size / 1024 / 1024
    print(f"  ✓ {out_pdf.name}（{size:.2f} MB）")
    if size > 20:
        print("  ! 超过 20 MB，提交前需压缩或改用预览版")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
