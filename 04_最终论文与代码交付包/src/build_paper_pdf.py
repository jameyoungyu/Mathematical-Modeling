#!/usr/bin/env python3
"""Build the CUMCM submission PDF from the final Markdown manuscript.

Converts ``10_修订后完整论文_终稿.md`` into XeLaTeX and compiles it.
Layout follows the CUMCM paper format specification: A4, >=2.5 cm margins,
abstract alone on page 1, continuous arabic page numbers centred in the footer.

Usage:  python3 src/build_paper_pdf.py [--no-code]
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANUSCRIPT = ROOT / "10_修订后完整论文_终稿.md"
BUILD = ROOT / "build"
SRC = ROOT / "src"
CODE_MODULES = ["scenario_model.py", "paper_figures.py", "plot_utils.py"]

# ---------------------------------------------------------------- inline text

_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def esc(text: str) -> str:
    return "".join(_SPECIALS.get(ch, ch) for ch in text)


def inline(text: str) -> str:
    """Convert inline Markdown to LaTeX, leaving $...$ math untouched."""
    out: list[str] = []
    for i, seg in enumerate(re.split(r"(\$[^$]*\$)", text)):
        if i % 2:                       # math segment - pass through verbatim
            out.append(seg)
            continue
        # code spans first: their content must be escaped but not styled
        parts = []
        for j, piece in enumerate(re.split(r"(`[^`]*`)", seg)):
            if j % 2:
                parts.append(r"\texttt{" + esc(piece[1:-1]) + "}")
            else:
                piece = esc(piece)
                piece = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", piece)
                parts.append(piece)
        out.append("".join(parts))
    return "".join(out)


# ---------------------------------------------------------------- block level

def convert_table(header: list[str], align: list[str], rows: list[list[str]],
                  number: str, title: str) -> str:
    spec = "".join({"r": "r", "c": "c"}.get(a, "l") for a in align)
    wide = len(header) >= 9          # 宽表按页宽等比缩放，避免溢出版心
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption*{\normalsize 表" + number + "\u3000" + inline(title) + "}",
             r"\vspace{-2pt}"]
    if wide:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [r"\begin{tabular}{" + spec + "}", r"\toprule",
              " & ".join(inline(h) for h in header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(inline(c) for c in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if wide:
        lines.append("}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def parse_align(sep_row: list[str]) -> list[str]:
    out = []
    for cell in sep_row:
        cell = cell.strip()
        if cell.startswith(":") and cell.endswith(":"):
            out.append("c")
        elif cell.endswith(":"):
            out.append("r")
        else:
            out.append("l")
    return out


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def convert(md: str) -> tuple[str, str, str]:
    """Return (title, abstract_block, body)."""
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    lines = md.split("\n")

    title = ""
    abstract: list[str] = []
    body: list[str] = []
    target = body

    i = 0
    pending_table: tuple[str, str] | None = None   # (number, title)
    while i < len(lines):
        line = lines[i].rstrip()

        # ---- headings
        m = re.match(r"^# (.+)$", line)
        if m:
            title = m.group(1).strip()
            i += 1
            continue
        m = re.match(r"^## (.+)$", line)
        if m:
            head = m.group(1).strip()
            if head == "摘要":
                target = abstract
                i += 1
                continue
            target = body
            hm = re.match(r"^(\d+)\s+(.*)$", head)
            if hm:
                target.append(r"\section{" + inline(hm.group(2)) + "}")
            else:
                target.append(r"\section*{" + inline(head) + "}")
                target.append(r"\addcontentsline{toc}{section}{" + inline(head) + "}")
            i += 1
            continue
        m = re.match(r"^### (.+)$", line)
        if m:
            head = m.group(1).strip()
            hm = re.match(r"^([\d.]+|[A-Z]\.\d+)\s+(.*)$", head)
            target.append(r"\subsection*{" + inline(head if not hm else head) + "}")
            i += 1
            continue

        # ---- fenced code
        if line.startswith("```"):
            j = i + 1
            buf = []
            while j < len(lines) and not lines[j].startswith("```"):
                buf.append(lines[j])
                j += 1
            target.append(r"\begin{lstlisting}[style=pseudo]")
            target.extend(buf)
            target.append(r"\end{lstlisting}")
            i = j + 1
            continue

        # ---- display math
        if line.strip() == "$$":
            j = i + 1
            buf = []
            while j < len(lines) and lines[j].strip() != "$$":
                buf.append(lines[j])
                j += 1
            expr = "\n".join(buf).strip().rstrip(",.")
            target.append(r"\begin{equation}")
            target.append(expr)
            target.append(r"\end{equation}")
            i = j + 1
            continue

        # ---- image  ->  figure float
        m = re.match(r"^!\[图(\d+)\s*([^\]]*)\]\(([^)]+)\)\s*$", line)
        if m:
            num, alt, path = m.group(1), m.group(2), m.group(3)
            cap = alt
            k = i + 1
            while k < len(lines) and not lines[k].strip():
                k += 1
            cm = re.match(r"^图%s\u3000(.+)$" % num, lines[k].strip()) if k < len(lines) else None
            if cm:
                cap = cm.group(1)
                i = k
            width = "0.96" if path.endswith("00_esp_schematic.pdf") else "0.92"
            target += [r"\begin{figure}[htbp]", r"\centering",
                       r"\includegraphics[width=%s\textwidth]{%s}" % (width, path),
                       r"\caption*{\normalsize 图" + num + "\u3000" + inline(cap) + "}",
                       r"\end{figure}"]
            i += 1
            continue

        # ---- standalone figure caption already consumed above; skip strays
        if re.match(r"^图\d+\u3000", line.strip()):
            i += 1
            continue

        # ---- table title line
        m = re.match(r"^表(\d+|A\d+)\u3000(.+)$", line.strip())
        if m:
            pending_table = (m.group(1), m.group(2))
            i += 1
            continue

        # ---- table body
        if line.strip().startswith("|"):
            header = split_row(line)
            align = parse_align(split_row(lines[i + 1]))
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            num, ttl = pending_table if pending_table else ("", "")
            target.append(convert_table(header, align, rows, num, ttl))
            pending_table = None
            i = j
            continue

        # ---- blockquote
        if line.startswith(">"):
            buf = [line.lstrip("> ").strip()]
            j = i + 1
            while j < len(lines) and lines[j].startswith(">"):
                buf.append(lines[j].lstrip("> ").strip())
                j += 1
            target.append(r"\begin{quote}\small " + inline(" ".join(buf)) + r"\end{quote}")
            i = j
            continue

        # ---- ordered / unordered list
        if re.match(r"^\d+\.\s+", line) or re.match(r"^[-*]\s+", line):
            ordered = bool(re.match(r"^\d+\.\s+", line))
            env = "enumerate" if ordered else "itemize"
            items = []
            j = i
            while j < len(lines) and (re.match(r"^\d+\.\s+", lines[j]) or re.match(r"^[-*]\s+", lines[j])):
                items.append(re.sub(r"^(\d+\.|[-*])\s+", "", lines[j].rstrip()))
                j += 1
            target.append(r"\begin{%s}[itemsep=1pt,topsep=3pt]" % env)
            target += [r"\item " + inline(it) for it in items]
            target.append(r"\end{%s}" % env)
            i = j
            continue

        # ---- blank
        if not line.strip():
            target.append("")
            i += 1
            continue

        # ---- plain paragraph
        target.append(inline(line.strip()))
        i += 1

    return title, "\n".join(abstract).strip(), "\n".join(body).strip()


# ---------------------------------------------------------------- preamble

PREAMBLE = r"""
\documentclass[UTF8,zihao=5,a4paper]{ctexart}
\usepackage[a4paper,top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{geometry}
\usepackage{amsmath,amssymb,bm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{caption}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{setspace}
\usepackage{array}
\usepackage{fancyhdr}

\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans Mono CJK SC}
\setmainfont{TeX Gyre Termes}
\setmonofont{Noto Sans Mono CJK SC}[Scale=0.82]

\onehalfspacing
\setlength{\parindent}{2em}
\captionsetup{skip=4pt}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\thepage}

\ctexset{
  section/format = {\raggedright\zihao{4}\bfseries},
  subsection/format = {\raggedright\zihao{5}\bfseries},
  section/beforeskip = 12pt plus 2pt minus 2pt,
  section/afterskip = 8pt,
}

\lstdefinestyle{pseudo}{
  basicstyle=\ttfamily\zihao{-5},
  breaklines=true, breakatwhitespace=false,
  columns=fullflexible, keepspaces=true,
  frame=single, rulecolor=\color{gray!55},
  framesep=4pt, xleftmargin=6pt, xrightmargin=2pt,
  showstringspaces=false, extendedchars=true,
}
\lstdefinestyle{pycode}{
  language=Python,
  basicstyle=\ttfamily\zihao{6},
  keywordstyle=\color{blue!65!black},
  commentstyle=\color{green!42!black}\itshape,
  stringstyle=\color{orange!75!black},
  breaklines=true, breakatwhitespace=false,
  breakindent=12pt, postbreak=\mbox{\textcolor{gray}{$\hookrightarrow$}\space},
  columns=fullflexible, keepspaces=true,
  showstringspaces=false, extendedchars=true,
  numbers=left, numberstyle=\tiny\color{gray}, numbersep=6pt,
  frame=leftline, rulecolor=\color{gray!45}, framesep=5pt,
  xleftmargin=17pt, tabsize=4, upquote=true,
}

% 手工编号的图表题注（与正文交叉引用一致）
\captionsetup[figure]{labelformat=empty,justification=centering}
\captionsetup[table]{labelformat=empty,justification=centering}
"""


def build_tex(title: str, abstract: str, body: str, code: str) -> str:
    parts = [PREAMBLE, r"\begin{document}", ""]
    # --- page 1: title + abstract only
    parts += [
        r"\begin{center}",
        r"{\zihao{3}\bfseries " + inline(title) + r"}",
        r"\end{center}",
        r"\vspace{6pt}",
        r"\begin{center}{\zihao{4}\bfseries 摘\quad 要}\end{center}",
        r"\vspace{2pt}",
    ]
    abs_lines = [l for l in abstract.split("\n") if l.strip()]
    kw = ""
    keep = []
    for l in abs_lines:
        if l.startswith(r"\textbf{关键词："):
            kw = l
        else:
            keep.append(l)
    parts += keep
    if kw:
        parts += ["", r"\vspace{8pt}", "", r"\noindent " + kw]
    parts += [r"\clearpage", ""]
    parts.append(body)
    if code:
        parts.append(code)
    parts += ["", r"\end{document}", ""]
    return "\n".join(parts)


def code_appendix() -> str:
    out = [r"\clearpage",
           r"\section*{附录D\quad 完整源程序}",
           r"\addcontentsline{toc}{section}{附录D\quad 完整源程序}",
           r"本附录给出生成本文全部结果与图表的完整可运行源程序，与电子支撑材料 \texttt{src/} 目录一致。"]
    for n, name in enumerate(CODE_MODULES, start=1):
        path = SRC / name
        if not path.exists():
            continue
        out += [r"\subsection*{D.%d\quad \texttt{%s}}" % (n, esc(name)),
                r"\lstinputlisting[style=pycode]{%s}" % (path.as_posix())]
    return "\n".join(out)


def main() -> None:
    include_code = "--no-code" not in sys.argv
    md = MANUSCRIPT.read_text(encoding="utf-8")
    title, abstract, body = convert(md)
    tex = build_tex(title, abstract, body, code_appendix() if include_code else "")

    BUILD.mkdir(exist_ok=True)
    tex_path = BUILD / "paper.tex"
    tex_path.write_text(tex, encoding="utf-8")

    for run in range(2):
        proc = subprocess.run(
            ["xelatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(BUILD), str(tex_path)],
            cwd=ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            log = (BUILD / "paper.log").read_text(encoding="utf-8", errors="replace")
            errs = [l for l in log.split("\n") if l.startswith("!")][:12]
            print("XeLaTeX 失败：\n" + "\n".join(errs))
            sys.exit(1)
    print("已生成", BUILD / "paper.pdf")


if __name__ == "__main__":
    main()
