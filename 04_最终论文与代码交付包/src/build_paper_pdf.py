#!/usr/bin/env python3
"""Build the CUMCM submission PDF from the final Markdown manuscript.

Converts ``10_修订后完整论文_终稿.md`` into XeLaTeX and compiles it.
Layout follows the CUMCM paper format specification: A4, >=2.5 cm margins,
abstract alone on page 1, continuous arabic page numbers centred in the footer.

Usage:  python3 src/build_paper_pdf.py [--no-code | --core-code]
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
CODE_MODULES = ["scenario_model.py", "paper_figures.py", "plot_utils.py",
                "compute_frontier.py", "integrity_check.py", "build_paper_pdf.py"]

# ---------------------------------------------------------------- inline text

_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    # symbols the CJK/serif fonts do not carry -> render them as maths
    "\u2103": r"$^\circ$C", "\u221d": r"$\propto$", "\u2260": r"$\neq$",
    "\u2264": r"$\le$", "\u2265": r"$\ge$", "\u2248": r"$\approx$",
    "\u207b": r"$^-$", "\u00b9": r"$^1$", "\u2070": r"$^0$",
    "\u2192": r"$\rightarrow$", "\u2191": r"$\uparrow$", "\u2193": r"$\downarrow$",
}


def esc(text: str) -> str:
    return "".join(_SPECIALS.get(ch, ch) for ch in text)


#: bare links and DOIs are single unbreakable words for TeX; \url can break them
_URL_RE = re.compile(r"(https?://\S+?)(?=[\s\u3002\uff0c]|$)|(?<=DOI: )(10\.\S+?)(?=[\s\u3002]|$)")


def _urlify(piece: str) -> str:
    r"""Escape text, but wrap any bare URL or DOI in a breakable url command."""
    out: list[str] = []
    last = 0
    for m in _URL_RE.finditer(piece):
        out.append(esc(piece[last:m.start()]))
        link = m.group(0).rstrip(".")
        trailing = m.group(0)[len(link):]
        out.append(r"\url{" + link + "}" + esc(trailing))
        last = m.end()
    out.append(esc(piece[last:]))
    return "".join(out)


#: Markdown bold may wrap a maths span ("**若把$x$作为约束**"), so the markers
#: cannot be resolved inside a single plain-text piece. They are replaced by a
#: sentinel first and paired up after every segment has been rendered.
_BOLD = "\x00"


def _smart_quotes(piece: str, state: list[bool]) -> str:
    """Straight double quotes in Chinese prose -> paired curly quotes.

    ``state`` carries the open/close flag across the whole line, so a quoted
    phrase that wraps a maths span still gets a closing quote at its end.
    """
    out = []
    for ch in piece:
        if ch == '"':
            out.append("\u201c" if state[0] else "\u201d")
            state[0] = not state[0]
        else:
            out.append(ch)
    return "".join(out)


def inline(text: str) -> str:
    """Convert inline Markdown to LaTeX, leaving $...$ math untouched."""
    out: list[str] = []
    quote_open = [True]
    for i, seg in enumerate(re.split(r"(\$[^$]*\$)", text)):
        if i % 2:                       # math segment - pass through verbatim
            out.append(seg)
            continue
        # code spans next: their content must be escaped but not styled
        for j, piece in enumerate(re.split(r"(`[^`]*`)", seg)):
            if j % 2:
                out.append(r"\texttt{" + esc(piece[1:-1]) + "}")
            else:
                piece = piece.replace("**", _BOLD)
                out.append(_urlify(_smart_quotes(piece, quote_open)))
    rendered = "".join(out)

    # pair the sentinels; an unmatched trailing marker is dropped
    parts = rendered.split(_BOLD)
    if len(parts) == 1:
        return rendered
    result = parts[0]
    for k, chunk in enumerate(parts[1:], start=1):
        result += (r"\textbf{" if k % 2 else "}") + chunk
    if len(parts) % 2 == 0:             # odd number of markers
        result += "}"
    return result


# ---------------------------------------------------------------- block level

def _display_width(text: str) -> int:
    """Rough typeset width in en-units (CJK glyphs are double width)."""
    text = re.sub(r"\$[^$]*\$", "xx", text)          # math sets compactly
    text = re.sub(r"[*`]", "", text)
    return sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)


# A row of ``l`` columns cannot wrap, so any table whose natural width exceeds
# the text block silently runs off the page. Text-heavy columns therefore
# become wrapping ``X`` columns and the table is set to exactly \textwidth.
_LINE_BUDGET = 84          # en-units that fit across the 16 cm text block
_WRAP_THRESHOLD = 16       # a column wider than this must be allowed to wrap


def convert_table(header: list[str], align: list[str], rows: list[list[str]],
                  number: str, title: str) -> str:
    ncol = len(header)
    widths = [max([_display_width(header[j])]
                  + [_display_width(r[j]) for r in rows if j < len(r)])
              for j in range(ncol)]
    natural = sum(widths) + 2 * ncol

    wrappable = [j for j in range(ncol)
                 if align[j] == "l" and widths[j] > _WRAP_THRESHOLD]
    use_tabularx = natural > _LINE_BUDGET and bool(wrappable)

    if use_tabularx:
        # Share the wrapping width between the text columns in proportion to
        # how much text each of them actually carries.
        # Proportional-to-length shares starve short-but-not-short-enough
        # columns, so damp the ratio with a square root and floor it.
        raw = {j: max(widths[j], 1) ** 0.5 for j in wrappable}
        total_wrap = sum(raw.values())
        shares = {j: max(len(wrappable) * raw[j] / total_wrap, 0.55) for j in wrappable}
        scale = len(wrappable) / sum(shares.values())
        parts = []
        for j in range(ncol):
            if j in wrappable:
                parts.append(r">{\hsize=%.3f\hsize\raggedright\arraybackslash}X"
                             % (shares[j] * scale))
            else:
                parts.append({"r": "r", "c": "c"}.get(align[j], "l"))
        open_env = r"\begin{tabularx}{\textwidth}{" + "".join(parts) + "}"
        close_env = r"\end{tabularx}"
        shrink = False
    else:
        spec = "".join({"r": "r", "c": "c"}.get(a, "l") for a in align)
        open_env = r"\begin{tabular}{" + spec + "}"
        close_env = r"\end{tabular}"
        shrink = natural > _LINE_BUDGET        # all-numeric wide table: scale it

    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption*{\normalsize 表" + number + "\u3000" + inline(title) + "}",
             r"\vspace{-2pt}", r"\zihao{-5}"]
    if shrink:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [open_env, r"\toprule",
              " & ".join(inline(h) for h in header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(inline(c) for c in row) + r" \\")
    lines += [r"\bottomrule", close_env]
    if shrink:
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
\usepackage{tabularx}
\usepackage{fancyhdr}
\usepackage{xurl}

%%FONTS%%

\onehalfspacing
\setlength{\parindent}{2em}
\captionsetup{skip=4pt}
\emergencystretch=3em
\sloppy

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyfoot[C]{\thepage}

\ctexset{
  section/format = {\centering\zihao{4}\bfseries},
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



# ---------------------------------------------------------------- fonts

#: Preferred first, portable fallback second. The submission machine normally
#: has TeX Gyre Termes and Noto CJK; a bare TeX Live may not.
FONT_CHOICES = {
    "main": ["TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"],
    "mono": ["Noto Sans Mono CJK SC", "DejaVu Sans Mono"],
    "cjkmain": ["Noto Serif CJK SC", "Fandol Song", "Noto Sans CJK SC"],
    "cjksans": ["Noto Sans CJK SC", "Fandol Hei"],
    "cjkmono": ["Noto Sans Mono CJK SC", "Noto Sans CJK SC"],
}


def _installed_families() -> set[str]:
    try:
        out = subprocess.run(["fc-list", ":", "family"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    return {name.strip() for line in out.splitlines() for name in line.split(",")}


def font_block() -> str:
    available = _installed_families()

    def pick(key: str) -> str:
        for name in FONT_CHOICES[key]:
            if not available or name in available:
                return name
        return FONT_CHOICES[key][-1]

    return "\n".join([
        r"\setCJKmainfont{%s}" % pick("cjkmain"),
        r"\setCJKsansfont{%s}" % pick("cjksans"),
        r"\setCJKmonofont{%s}" % pick("cjkmono"),
        r"\setmainfont{%s}" % pick("main"),
        r"\setmonofont{%s}[Scale=0.82]" % pick("mono"),
    ])


def build_tex(title: str, abstract: str, body: str, code: str) -> str:
    parts = [PREAMBLE.replace("%%FONTS%%", font_block()), r"\begin{document}", ""]
    # --- page 1: title + abstract only
    parts += [
        r"\begin{center}",
        r"{\zihao{3}\bfseries " + inline(title) + r"}",
        r"\end{center}",
        r"\vspace{6pt}",
        r"\begin{center}{\zihao{4}\bfseries 摘\quad 要}\end{center}",
        r"\vspace{2pt}",
    ]
    kw = ""
    keep = []
    for l in abstract.split("\n"):          # 保留空行以维持摘要的分段结构
        if l.startswith(r"\textbf{关键词："):
            kw = l
        else:
            keep.append(l)
    while keep and not keep[-1].strip():
        keep.pop()
    parts += keep
    if kw:
        parts += ["", r"\vspace{8pt}", "", r"\noindent " + kw]
    parts += [r"\clearpage", ""]
    parts.append(body)
    if code:
        parts.append(code)
    parts += ["", r"\end{document}", ""]
    return "\n".join(parts)


def _extract(module: str, names: list[str]) -> str:
    """Pull whole top-level definitions out of a source file, in order."""
    text = (SRC / module).read_text(encoding="utf-8").split("\n")
    out: list[str] = []
    for name in names:
        try:
            i = next(k for k, l in enumerate(text)
                     if l.startswith(("def " + name + "(", "class " + name)))
        except StopIteration:
            continue
        j = i + 1
        while j < len(text) and (not text[j].strip() or text[j][:1] in " )]}\t"):
            j += 1
        while j > i and not text[j - 1].strip():
            j -= 1
        out.extend(text[i:j] + [""])
    return "\n".join(out).rstrip() + "\n"


#: Appendix E carries the full listing by default; ``--core-code`` prints only
#: the two definitions that determine every number in the paper.
CORE_EXTRACTS = [
    ("scenario_model.py", ["PhysicalPowerModel"]),
    ("scenario_model.py", ["scenario_emission", "field_priority"]),
]


def code_appendix(full: bool = True) -> str:
    title = "\u9644\u5f55E\\quad " + ("\u5b8c\u6574\u6e90\u7a0b\u5e8f" if full else "\u6838\u5fc3\u6e90\u7a0b\u5e8f\u8282\u9009")
    out = [r"\clearpage",
           r"\section*{%s}" % title,
           r"\addcontentsline{toc}{section}{%s}" % title]
    if full:
        out.append("\u672c\u9644\u5f55\u7ed9\u51fa\u751f\u6210\u672c\u6587\u5168\u90e8\u7ed3\u679c\u4e0e\u56fe\u8868\u7684\u5b8c\u6574\u53ef\u8fd0\u884c\u6e90\u7a0b\u5e8f\uff0c\u4e0e\u7535\u5b50\u652f\u6491\u6750\u6599 \\texttt{src/} \u76ee\u5f55\u9010\u5b57\u4e00\u81f4\u3002"
                   "\u6309\u4f9d\u8d56\u987a\u5e8f\u6392\u5217\uff1a\u60c5\u666f\u6a21\u578b\u4e0e\u4f18\u5316\u3001\u6392\u653e\u2014\u529f\u7387\u524d\u6cbf\u3001\u7ed8\u56fe\u6837\u5f0f\u4e0e\u56fe\u96c6\u751f\u6210\u3001\u786e\u5b9a\u6027\u6838\u9a8c\u3001\u63d0\u4ea4\u7a3f\u88c5\u914d\u3002")
        for n, name in enumerate(CODE_MODULES, start=1):
            path = SRC / name
            if not path.exists():
                continue
            out += [r"\subsection*{E.%d\quad \texttt{%s}}" % (n, esc(name)),
                    r"\lstinputlisting[style=pycode]{%s}" % (path.as_posix())]
        return "\n".join(out)

    out.append("\u4e3a\u63a7\u5236\u7bc7\u5e45\uff0c\u672c\u9644\u5f55\u53ea\u7ed9\u51fa\u51b3\u5b9a\u5168\u6587\u6570\u503c\u7ed3\u679c\u7684\u4e24\u6bb5\u6838\u5fc3\u5b9a\u4e49\uff1b"
               "\u5b8c\u6574\u53ef\u8fd0\u884c\u6e90\u7801\u968f\u7535\u5b50\u652f\u6491\u6750\u6599\u63d0\u4ea4\u3002")
    BUILD.mkdir(exist_ok=True)
    for n, (module, names) in enumerate(CORE_EXTRACTS, start=1):
        snippet = BUILD / ("code_excerpt_%d.py" % n)
        snippet.write_text(_extract(module, names), encoding="utf-8")
        out += [r"\subsection*{E.%d\quad \texttt{%s}\uff1a%s}"
                % (n, esc(module), "\u3001".join(esc(x) for x in names)),
                r"\lstinputlisting[style=pycode]{%s}" % snippet.as_posix()]
    return "\n".join(out)


def main() -> None:
    include_code = "--no-code" not in sys.argv
    full_code = "--core-code" not in sys.argv
    md = MANUSCRIPT.read_text(encoding="utf-8")
    title, abstract, body = convert(md)
    tex = build_tex(title, abstract, body, code_appendix(full_code) if include_code else "")

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
