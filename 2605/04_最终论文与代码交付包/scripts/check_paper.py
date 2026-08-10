#!/usr/bin/env python3
"""国赛论文格式与数字溯源检查。

论文里有一批规则可以做机械预检：不能出现校名、摘要原则上不超过一页、参考文献要双向闭合、
附录要有可运行代码。人工核对这些既慢又容易漏，尤其在交稿前几小时。把它们变成脚本，
是把"要记住的纪律"降级成"跑一下就知道"的机制。

最有价值的一项是数字溯源：把论文里出现的结果型数字和 results/results.json 对照，
把找不到出处的挑出来。挑出来不等于错（题目给定的常数、年份都会被挑出来），
但每一条都应该能说清来源 —— 说不清的那个，往往就是编的。

用法:
    python scripts/check_paper.py 论文.md
    python scripts/check_paper.py 论文.md --results results/results.json
    python scripts/check_paper.py 论文.md --results results/results.json --strict

退出码: 0 = 无资格风险; 1 = 存在资格风险（或 --strict 下存在任何扣分项）。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 结果模型

RISK = "资格风险"
DEDUCT = "扣分项"
HINT = "提示"

SEVERITY_ORDER = [RISK, DEDUCT, HINT]


@dataclass
class Finding:
    severity: str
    rule: str
    message: str
    line: int | None = None


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, severity: str, rule: str, message: str, line: int | None = None) -> None:
        self.findings.append(Finding(severity, rule, message, line))

    def by_severity(self, severity: str) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]


# ---------------------------------------------------------------- 文本读取

def read_paper(path: Path) -> str:
    """读取论文。支持 md/txt，装了对应库时也支持 docx/pdf。"""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt", ".tex"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".docx":
        try:
            import docx  # type: ignore
        except ImportError:
            sys.exit("读取 .docx 需要 python-docx：pip install python-docx")
        return "\n".join(p.text for p in docx.Document(str(path)).paragraphs)
    if suffix == ".pdf":
        try:
            import pdfplumber  # type: ignore
        except ImportError:
            sys.exit("读取 .pdf 需要 pdfplumber：pip install pdfplumber")
        with pdfplumber.open(str(path)) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    sys.exit(f"不支持的文件类型：{suffix}")


def strip_code_blocks(text: str) -> str:
    """去掉围栏代码块。附录代码里的路径、变量名不该按正文规则检查。"""
    return re.sub(r"```.*?```", "", text, flags=re.S)


def normalize_minus(text: str) -> str:
    """把排版用的各种减号统一成 ASCII 连字符。

    论文里写 −0.412（U+2212 真减号）很常见，不归一化的话数字提取会漏掉负号，
    把它当成 0.412 去和结果文件比对，于是一个明明有出处的数字被报成"找不到出处"。
    """
    return text.replace("−", "-").replace("－", "-").replace("–", "-").replace("—", "-")


def split_keywords(raw: str) -> list[str]:
    """切分关键词行。

    关键词之间可能用中文分号、顿号、逗号或多个空格分隔，但关键词本身也可能含空格
    （"LASSO 回归"）。所以有显式分隔符时只按分隔符切，没有时才退回按多空格切。
    """
    raw = re.sub(r"[*_`]", "", raw).strip()
    if re.search(r"[;；、,，]", raw):
        parts = re.split(r"[;；、,，]+", raw)
    else:
        parts = re.split(r"\s{2,}|\t|　+", raw) if re.search(r"\s{2,}|　", raw) else raw.split()
    return [p.strip() for p in parts if p.strip()]


HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s*(.+?)\s*$|^\s*(?:[一二三四五六七八九十]+)\s*[、.]\s*(.+?)\s*$", re.M)
APPENDIX_HEADING_RE = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*|[一二三四五六七八九十]+[、.]\s*)?附录(?:\s|$).*",
    re.M,
)


def mask_code_blocks(text: str) -> str:
    """把围栏代码块替换成等长空白，保持字符偏移不变。

    附录里的 Python 注释以 # 开头，和 markdown 标题长得一模一样。不屏蔽的话，
    「# 读取附件数据」会被当成一个新章节，把附录从中间劈开——于是脚本报告
    「附录中没有找到源程序代码」这个根本不存在的资格风险。
    """
    return re.sub(r"```.*?```", lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=re.S)


def split_sections(text: str) -> dict[str, str]:
    """按标题切分论文，返回 {标题: 正文}。标题匹配 markdown # 和"一、xxx"两种写法。"""
    marks: list[tuple[int, str]] = []
    for m in HEADING_RE.finditer(mask_code_blocks(text)):
        title = (m.group(2) or m.group(3) or "").strip()
        if title:
            marks.append((m.start(), title))
    sections: dict[str, str] = {}
    for i, (pos, title) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        sections.setdefault(title, "")
        sections[title] += text[pos:end]
    if not sections:
        sections["__全文__"] = text
    return sections


def find_section(sections: dict[str, str], *keywords: str) -> str:
    """取标题含任一关键词的章节内容，找不到返回空串。"""
    parts = [body for title, body in sections.items() if any(k in title for k in keywords)]
    return "\n".join(parts)


def appendix_tail(text: str) -> str:
    """返回从第一个附录标题到文末的内容。

    ``split_sections`` 是扁平切分；若论文写成 ``# 附录`` 后再写
    ``## 支撑材料文件列表``，只按标题名找“附录”会漏掉二级小节。
    """
    match = APPENDIX_HEADING_RE.search(mask_code_blocks(text))
    return text[match.start():] if match else ""


def line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


# ---------------------------------------------------------------- 检查项

IDENTITY_PATTERNS: list[tuple[str, str]] = [
    (r"[一-龥]{2,12}(?:大学|学院|职业技术学校|中学)(?!出版社)", "疑似校名"),
    (r"指导(?:教师|老师)\s*[:：]?\s*\S+", "指导教师"),
    (r"学\s*号\s*[:：]?\s*\w+", "学号"),
    (r"参赛队号|报名号|队伍编号", "队号（编号专用页除外）"),
    (r"/Users/[A-Za-z0-9_.\-]+", "绝对路径含用户名"),
    (r"[Cc]:\\\\?Users\\\\?[A-Za-z0-9_.\-]+", "绝对路径含用户名"),
    (r"/home/[A-Za-z0-9_.\-]+", "绝对路径含用户名"),
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", "邮箱地址"),
]


def check_identity(text: str, sections: dict[str, str], report: Report) -> None:
    """身份信息泄漏。规范里这是取消资格的条件，不是扣分。"""
    refs = find_section(sections, "参考文献")
    ref_span: tuple[int, int] | None = None
    if refs:
        start = text.find(refs[:200])
        if start >= 0:
            ref_span = (start, start + len(refs))

    body = strip_code_blocks(text)
    for pattern, label in IDENTITY_PATTERNS:
        for m in re.finditer(pattern, body):
            # 参考文献里出现出版社、作者名是正常的，跳过
            if ref_span is not None and ref_span[0] <= m.start() < ref_span[1]:
                continue
            snippet = m.group(0)[:40]
            report.add(RISK, "身份信息", f"{label}：「{snippet}」", line_of(body, m.start()))

    # 代码块里的绝对路径同样泄漏身份，且会让评委跑不通
    for m in re.finditer(r"```.*?```", text, flags=re.S):
        for pattern, label in IDENTITY_PATTERNS[3:7]:
            for hit in re.finditer(pattern, m.group(0)):
                report.add(
                    RISK,
                    "身份信息",
                    f"附录代码中{label}：「{hit.group(0)[:40]}」",
                    line_of(text, m.start() + hit.start()),
                )


def check_abstract(sections: dict[str, str], report: Report) -> None:
    """摘要：存在、字数提示、结果表述与关键词。"""
    abstract = find_section(sections, "摘要")
    if not abstract.strip():
        report.add(RISK, "摘要", "未找到摘要章节。摘要页是评委看的第一页，必须有。")
        return

    body = re.sub(r"^#+.*$", "", abstract, flags=re.M)
    chars = len(re.sub(r"\s", "", body))
    if chars > 1200:
        report.add(HINT, "摘要", f"摘要约 {chars} 字，可能超出一页。字数不能代替排版页数，请打开最终 PDF/Word 确认摘要（含标题和关键词）原则上不超过一页。")
    elif chars > 1000:
        report.add(HINT, "摘要", f"摘要约 {chars} 字，接近一页上限，排版后请确认没有溢出。")
    elif chars < 250:
        report.add(HINT, "摘要", f"摘要仅约 {chars} 字，请确认已回答各小问并给出方法与结论；短不等于违规。")

    # 数值题的摘要通常需要关键数值；纯证明或定性任务不强制每段出现数字
    paragraphs = [
        p for p in re.split(r"\n\s*\n", body)
        if len(re.sub(r"\s", "", p)) > 30 and "关键词" not in p
    ]
    numeric = re.compile(r"\d+\.\d+|\d{2,}")
    weak = [p for p in paragraphs if not numeric.search(p)]
    if paragraphs and len(weak) == len(paragraphs):
        report.add(HINT, "摘要", "摘要中没有具体数字。若题目要求数值、预测、排序或优化结果，请补入可复现的关键数值；纯证明或定性任务可忽略。")
    elif weak:
        for p in weak[:3]:
            report.add(HINT, "摘要", f"这一段没有具体数字：「{p.strip()[:45]}…」")

    # 空话检测
    for phrase in ["较好的结果", "较优的方案", "具有广泛的应用前景", "取得了良好的效果", "合理的数学模型"]:
        if phrase in body:
            report.add(HINT, "摘要", f"出现空泛表述「{phrase}」，建议换成具体数字。")

    kw = re.search(r"关键词\s*[:：]?\s*(.+)", body)
    if not kw:
        report.add(DEDUCT, "摘要", "摘要末尾缺少关键词。")
    else:
        words = split_keywords(kw.group(1))
        if not 3 <= len(words) <= 6:
            report.add(HINT, "摘要", f"关键词 {len(words)} 个。官网未统一规定数量，通常建议 3–6 个，并服从赛区模板。")
        vague = {"数学模型", "优化", "分析", "模型", "算法", "建模"}
        for w in words:
            if w in vague:
                report.add(HINT, "摘要", f"关键词「{w}」没有区分度，建议换成具体方法名或问题对象。")


def check_structure(text: str, sections: dict[str, str], report: Report) -> None:
    """正文结构建议、目录禁令与篇幅提示。"""
    required = [
        (("问题重述", "问题的重述"), HINT),
        (("问题分析", "问题的分析"), HINT),
        (("模型假设", "基本假设"), HINT),
        (("符号说明", "符号约定"), HINT),
        (("灵敏度", "敏感性"), HINT),
        (("模型的评价", "模型评价"), HINT),
        (("参考文献",), HINT),
        (("附录",), RISK),
    ]
    for names, severity in required:
        if not find_section(sections, *names):
            suffix = "官网要求正文后有附录；无程序或支撑材料时也要在附录明确说明。" if names[0] == "附录" else "这是建议结构，不是官网统一规定的固定标题；若内容已在别处清楚呈现可忽略。"
            report.add(severity, "结构", f"未找到「{names[0]}」章节。{suffix}")

    for title in sections:
        if title.strip() in {"目录", "目 录", "Contents"}:
            report.add(DEDUCT, "结构", "正文出现目录。国赛规范要求正文不要目录。")

    # 字数无法可靠换算成页数；这里只做过长预警，不能据此判定违反 30 页上限
    appendix = find_section(sections, "附录")
    main = strip_code_blocks(text)
    if appendix:
        idx = main.find(appendix[:120])
        if idx > 0:
            main = main[:idx]
    chars = len(re.sub(r"\s", "", main))
    if chars > 24000:
        report.add(HINT, "篇幅", f"正文约 {chars} 个非空白字符，可能接近或超过 30 页。字符数受公式、图表和排版影响很大，必须以最终 PDF/Word 的正文实际页数为准。")


def check_appendix(sections: dict[str, str], report: Report, source_suffix: str = ".md") -> None:
    """附录：识别代码或官方允许的“未使用程序”说明。"""
    appendix = find_section(sections, "附录")
    full_text = "\n".join(sections.values())
    tail = appendix_tail(full_text)
    if tail:
        appendix = tail
    if not appendix:
        return
    if re.search(r"本论文(?:确实)?没有用到程序|本论文未使用程序", appendix):
        if not re.search(r"本论文没有支撑材料|支撑材料", appendix):
            report.add(HINT, "附录", "已声明未使用程序；请继续说明支撑材料文件列表，或明确写“本论文没有支撑材料”。")
        return
    blocks = re.findall(r"```(\w*)\n(.*?)```", appendix, flags=re.S)
    if not blocks:
        if source_suffix in {".md", ".markdown"}:
            report.add(RISK, "附录", "Markdown 附录中没有找到围栏代码块，也没有“本论文没有用到程序”说明。若建模使用了程序，规范要求附上全部完整、可运行代码。")
        else:
            report.add(HINT, "附录", "文本抽取无法可靠判断 Word/PDF/TeX 中是否包含完整代码。请人工核对：有程序就附全部可运行代码；无程序就明确说明。")
        return
    total = sum(len(b[1].splitlines()) for b in blocks)
    if total < 30:
        report.add(DEDUCT, "附录", f"附录代码仅 {total} 行，很可能不是完整可运行的程序。")
    if not re.search(r"文件列表|支撑材料", appendix):
        report.add(HINT, "附录", "附录建议给出支撑材料文件列表，说明每个文件对应论文哪一部分。")


AI_TOOL_NAMES = [
    "ChatGPT", "GPT-4", "GPT-5", "OpenAI", "Claude", "Anthropic", "DeepSeek", "Gemini",
    "Copilot", "Cursor", "LLaMA", "Qwen", "通义千问", "文心一言", "讯飞星火", "Kimi",
    "豆包", "智谱", "ChatGLM", "GLM-4", "Ollama", "Grok", "Codex", "AI智能体", "AI 智能体",
    "大语言模型", "生成式AI", "生成式 AI",
]

AI_DECLARATION_HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*|[一二三四五六七八九十]+[、.]\s*)?AI\s*工具使用声明\s*$",
    re.M,
)
REFERENCE_HEADING = re.compile(
    r"^\s{0,3}(?:#{1,6}\s*|[一二三四五六七八九十]+[、.]\s*)?参考文献\s*$",
    re.M,
)
NO_AI_DECLARATION = re.compile(
    r"本参赛队在竞赛过程中未使用任何\s*AI\s*工具\s*[。.]?"
)
USED_AI_DECLARATION = re.compile(
    r"本参赛队在竞赛过程中使用了\s*AI\s*工具\s*[，,]\s*主要用于"
    r"(?P<purpose>[^。\n]{1,160}?)[，,]\s*详细使用情况见支撑材料\s*[。.]?"
)

# 用于发现“声明未使用”与论文其他内容明显矛盾的线索；不能据此替代人工认定。
AI_USAGE_HINT = re.compile(
    r"(?:由|经|借助|使用|采用|通过)\s*(?:AI|人工智能)\s*(?:工具)?\s*(?:生成|辅助|完成|撰写|编写|优化)"
    r"|(?:AI|人工智能)\s*工具\s*(?:生成|辅助|使用)"
    r"|AI\s*智能体"
)


def check_ai_compliance(text: str, sections: dict[str, str], report: Report) -> None:
    """AI 工具使用规定机械预检（依据 2026 年试行规定；比赛当日仍需复核）。

    每篇论文都要在参考文献之前设置“AI工具使用声明”，使用官方二选一文字。
    使用 AI 时，支撑材料还必须包含文件名准确的 ``AI工具使用详情.pdf``。
    脚本不能判断是否真正做到核心工作由队伍主导、逐项人工审查与核实。
    """
    body = strip_code_blocks(text)
    refs = find_section(sections, "参考文献")
    reference_heading = REFERENCE_HEADING.search(body)
    if not refs and reference_heading:
        # PDF/Word 文本抽取后，标题层级信息可能丢失；从“参考文献”行回退截取，
        # 至少保证声明位置与旧版 AI 著录提示不会静默漏检。
        refs = body[reference_heading.start():]
    if not refs:
        return  # 不是一篇完整论文（比如只写了摘要），这项检查无从谈起

    no_matches = list(NO_AI_DECLARATION.finditer(body))
    used_matches = list(USED_AI_DECLARATION.finditer(body))
    declaration_headings = list(AI_DECLARATION_HEADING.finditer(body))
    tools_in_refs = sorted({n for n in AI_TOOL_NAMES if n in refs})

    if not declaration_headings:
        report.add(RISK, "AI合规",
                   "未找到标题“AI工具使用声明”。2026 年试行规定要求在参考文献之前设置该声明。")

    exact_count = len(no_matches) + len(used_matches)
    if exact_count == 0:
        report.add(RISK, "AI合规",
                   "没有找到 2026 年试行规定给出的二选一声明文字。请使用官方句式，不要沿用 2025 年旧声明。")
        return
    if no_matches and used_matches:
        report.add(RISK, "AI合规",
                   "同时出现“未使用 AI”和“使用了 AI”两种声明，必须二选一并与实际情况一致。")
        return
    if exact_count > 1:
        report.add(DEDUCT, "AI合规", "AI 工具使用声明重复出现，请只保留一份官方声明。")

    declaration = (used_matches or no_matches)[0]
    reference_pos = reference_heading.start() if reference_heading else body.find(refs[:120])
    if reference_pos >= 0 and declaration.start() > reference_pos:
        report.add(RISK, "AI合规",
                   "AI工具使用声明位于参考文献之后。2026 年试行规定要求将其放在参考文献之前。")
    elif declaration_headings and reference_pos >= 0 and declaration_headings[0].start() > reference_pos:
        report.add(RISK, "AI合规", "标题“AI工具使用声明”位于参考文献之后，位置不符合 2026 年试行规定。")

    if no_matches:
        without_declaration = NO_AI_DECLARATION.sub("", body)
        without_declaration = AI_DECLARATION_HEADING.sub("", without_declaration)
        if AI_USAGE_HINT.search(without_declaration) or re.search(r"AI工具使用详情\.pdf", without_declaration, re.I):
            report.add(RISK, "AI合规",
                       "声明称未使用 AI，但论文其他位置出现 AI 辅助/生成或 AI工具使用详情.pdf 的线索。"
                       "请核实实际情况，避免虚假声明。")
    else:
        purpose = used_matches[0].group("purpose").strip()
        if (not purpose or len(purpose) < 2 or re.search(r"[【】\[\]]|简要用途|如语言润色", purpose)):
            report.add(RISK, "AI合规",
                       "“主要用于”后的简要用途仍为空或含示例占位符，请替换成真实用途。")
        appendix = appendix_tail(body) or find_section(sections, "附录")
        if not re.search(r"AI工具使用详情\.pdf", appendix, re.I):
            report.add(RISK, "AI合规",
                       "使用了 AI，但附录的支撑材料文件列表中没有找到精确文件名“AI工具使用详情.pdf”。"
                       "请把该 PDF 放入支撑材料并在文件列表中列明。")
        report.add(HINT, "AI合规",
                   "脚本只能检查论文中的声明和文件清单，不能确认 AI工具使用详情.pdf 是否实际存在、"
                   "内容是否完整，也不能判断 AI 输出是否已逐项人工审查与核实；提交前必须人工复核。")

    if tools_in_refs:
        report.add(HINT, "AI合规",
                   f"参考文献中出现 AI 工具（{', '.join(tools_in_refs[:4])}）。2026 年试行规定不再强制"
                   "把 AI 工具作为参考文献著录；若这些条目不是正文实际引用的信息来源，可删除以免影响引用闭合。")


def check_references(text: str, sections: dict[str, str], report: Report) -> None:
    """参考文献与正文引用双向闭合。"""
    refs = find_section(sections, "参考文献")
    if not refs:
        return
    listed = {int(n) for n in re.findall(r"^\s*\[(\d+)\]", refs, flags=re.M)}
    if not listed:
        report.add(DEDUCT, "参考文献", "参考文献没有使用 [编号] 格式。")
        return

    body = strip_code_blocks(text)
    idx = body.find(refs[:120])
    main = body[:idx] if idx > 0 else body
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", main)}

    for n in sorted(listed - cited):
        report.add(DEDUCT, "参考文献", f"文献 [{n}] 在正文中从未被引用。参考文献应按正文引用次序列出。")
    for n in sorted(cited - listed):
        report.add(DEDUCT, "参考文献", f"正文引用了 [{n}]，但参考文献列表中没有这一条。")

    for m in re.finditer(r"^\s*\[(\d+)\]\s*(.+)$", refs, flags=re.M):
        entry = m.group(2)
        if "出版社" in entry and not re.search(r"\d+\s*[-–—]\s*\d+\s*页|第\s*\d+", entry):
            report.add(HINT, "参考文献", f"[{m.group(1)}] 是书籍但未标页码，规范要求引用书籍须指出页码。")


def check_figures_tables(text: str, report: Report) -> None:
    """图题在图下方、表题在表上方，且图表都被正文引用。"""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"\s*!\[.*\]\(.+\)", line):
            after = "\n".join(lines[i + 1 : i + 4])
            before = "\n".join(lines[max(0, i - 3) : i])
            if re.search(r"图\s*\d+", before) and not re.search(r"图\s*\d+", after):
                report.add(HINT, "图表", "图题似乎在图上方，学术惯例是图题在图下方。", i + 1)
            elif not re.search(r"图\s*\d+", after):
                report.add(HINT, "图表", "这张图下方没有找到「图 N」形式的图题。", i + 1)

    fig_nums = {int(n) for n in re.findall(r"图\s*(\d+)", text)}
    tbl_nums = {int(n) for n in re.findall(r"表\s*(\d+)", text)}
    for label, nums in (("图", fig_nums), ("表", tbl_nums)):
        if nums and max(nums) != len(nums):
            missing = sorted(set(range(1, max(nums) + 1)) - nums)
            if missing:
                report.add(HINT, "图表", f"{label}编号不连续，缺少：{', '.join(map(str, missing))}。")

    if re.search(r"图\s*\d+", text) and not re.search(r"(如|见)图\s*\d+", text):
        report.add(HINT, "图表", "正文中没有出现「如图 N 所示」形式的引用，图表应在正文中被引用。")


def check_leftovers(text: str, report: Report) -> None:
    """模板残留与占位符。"""
    if re.search(r"<!--", text):
        n = len(re.findall(r"<!--", text))
        report.add(DEDUCT, "残留", f"存在 {n} 处 HTML 注释未删除（模板写作提示）。")
    for m in re.finditer(r"【([^】]{1,30})】", text):
        report.add(DEDUCT, "残留", f"模板占位符未填写：「【{m.group(1)}】」", line_of(text, m.start()))
    for kw in ("待计算", "TODO", "TBD", "XXX"):
        for m in re.finditer(re.escape(kw), text):
            report.add(RISK, "残留", f"存在未完成标记「{kw}」，提交前必须处理。", line_of(text, m.start()))


# ---------------------------------------------------------------- 数字溯源

YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def collect_result_numbers(obj) -> set[str]:
    """递归收集 results.json 里的所有数字（含字符串里嵌的数字）。"""
    out: set[str] = set()
    if isinstance(obj, dict):
        for v in obj.values():
            out |= collect_result_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out |= collect_result_numbers(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.add(normalize_number(obj))
    elif isinstance(obj, str):
        for tok in re.findall(r"-?\d+(?:\.\d+)?", obj):
            out.add(normalize_number(float(tok)))
    return out


def normalize_number(x: float | int | str) -> str:
    v = float(x)
    return f"{v:.10g}"


def number_matches(paper_num: float, known: set[str]) -> bool:
    """论文里的数字能否在结果集合中找到出处（允许四舍五入）。"""
    if normalize_number(paper_num) in known:
        return True
    for k in known:
        kv = float(k)
        if kv == 0:
            continue
        # 论文常写成保留 1-2 位小数的版本
        for digits in (0, 1, 2, 3):
            if abs(round(kv, digits) - paper_num) < 1e-9:
                return True
        # 比例/百分号换算
        if abs(kv * 100 - paper_num) < 1e-6 or abs(kv / 100 - paper_num) < 1e-12:
            return True
        # 正负号：论文常写"系数绝对值 0.412"或把负号写进文字里
        if abs(abs(kv) - abs(paper_num)) < 1e-9:
            return True
    return False


def collect_data_numbers(paths: list[Path], report: Report) -> set[str]:
    """收集题目附件里的数字。

    论文正文会大量引用题目给定的常数（单位利润 44.0、单位消耗 2.8……），
    这些数字天然不在 results.json 里。不把它们算作合法出处，溯源检查就会被
    几十条误报淹没，真正编造的那一个反而看不见了。
    """
    known: set[str] = set()
    for p in paths:
        for f in sorted(p.rglob("*")) if p.is_dir() else [p]:
            if f.suffix.lower() not in {".csv", ".tsv", ".json", ".txt"} or not f.is_file():
                continue
            try:
                raw = normalize_minus(f.read_text(encoding="utf-8-sig", errors="replace"))
            except OSError as e:
                report.add(HINT, "数字溯源", f"读不了数据文件 {f}：{e}")
                continue
            for tok in re.findall(r"-?\d+(?:\.\d+)?", raw):
                known.add(normalize_number(float(tok)))
    return known


def check_number_provenance(
    text: str,
    sections: dict[str, str],
    results_path: Path | None,
    report: Report,
    data_paths: list[Path] | None = None,
) -> None:
    """论文里的结果型数字是否都能在 results.json（或题目附件）里找到出处。"""
    if results_path is None:
        report.add(
            HINT,
            "数字溯源",
            "未提供 --results，跳过数字溯源检查。建议求解脚本把所有要写进论文的数字落到 results/results.json，"
            "论文从那里取数——这是防止编造数字最有效的一步。",
        )
        return
    if not results_path.exists():
        report.add(DEDUCT, "数字溯源", f"结果文件不存在：{results_path}")
        return

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        report.add(DEDUCT, "数字溯源", f"结果文件不是合法 JSON：{e}")
        return

    known = collect_result_numbers(data)
    if not known:
        report.add(DEDUCT, "数字溯源", "结果文件里没有任何数字。")
        return

    n_results = len(known)
    n_data = 0
    if data_paths:
        data_known = collect_data_numbers(data_paths, report)
        n_data = len(data_known - known)
        known |= data_known

    # 只查最要命的区域：摘要和各问的结果/检验部分
    targets = {
        "摘要": find_section(sections, "摘要"),
        "结果": find_section(sections, "结果", "求解", "检验", "灵敏度", "敏感性"),
    }
    if not any(targets.values()):
        targets = {"全文": text}

    unexplained: dict[str, list[str]] = {}
    for area, body in targets.items():
        if not body:
            continue
        body = strip_code_blocks(body)
        body = re.sub(r"\[\d+\]", "", body)                  # 文献编号
        body = re.sub(r"(图|表|式|问题|第|章|节)\s*\d+(\.\d+)*", "", body)  # 图表式章节编号
        body = re.sub(r"^\s*#+.*$", "", body, flags=re.M)    # 标题里的编号

        for m in re.finditer(r"-?\d+(?:\.\d+)?", body):
            tok = m.group(0)
            if "." not in tok and (len(tok.lstrip("-")) < 3 or YEAR_RE.match(tok)):
                continue  # 小整数与年份不算结果型数字
            val = float(tok)
            if not number_matches(val, known):
                unexplained.setdefault(area, []).append(tok)

    for area, nums in unexplained.items():
        uniq = sorted(set(nums), key=lambda s: nums.index(s))
        # 自动匹配允许四舍五入但仍会误报题给常数、公式参数和排版编号；先要求人工核对，
        # 不能仅凭脚本把它定为资格问题。
        severity = DEDUCT if area == "摘要" else HINT
        report.add(
            severity,
            "数字溯源",
            f"{area}中有 {len(uniq)} 个数字在 results.json 里找不到出处："
            f"{', '.join(uniq[:12])}{' …' if len(uniq) > 12 else ''}。"
            "题目给定的常数会被误报，但每一个都应能说清来源——说不清的那个通常就是编的。",
        )

    if not unexplained:
        src = f"results.json（{n_results} 个数值）"
        if n_data:
            src += f" + 题目附件（另 {n_data} 个数值）"
        report.add(HINT, "数字溯源", f"通过：论文中的结果型数字均能在 {src} 中找到出处。")


# ---------------------------------------------------------------- 主流程

def run_checks(
    text: str,
    results_path: Path | None,
    data_paths: list[Path] | None = None,
    source_suffix: str = ".md",
) -> Report:
    report = Report()
    text = normalize_minus(text)
    sections = split_sections(text)
    check_identity(text, sections, report)
    check_abstract(sections, report)
    check_structure(text, sections, report)
    check_appendix(sections, report, source_suffix)
    check_ai_compliance(text, sections, report)
    check_references(text, sections, report)
    check_figures_tables(text, report)
    check_leftovers(text, report)
    check_number_provenance(text, sections, results_path, report, data_paths)
    return report


def print_report(report: Report, paper: Path) -> None:
    print(f"\n{'=' * 62}")
    print(f"  论文检查：{paper}")
    print(f"{'=' * 62}")

    icons = {RISK: "✗", DEDUCT: "!", HINT: "·"}
    headers = {
        RISK: "资格风险 —— 规范里可能导致取消评奖资格的问题",
        DEDUCT: "扣分项 —— 违反格式规范或评阅要点",
        HINT: "提示 —— 建议改进",
    }
    for sev in SEVERITY_ORDER:
        items = report.by_severity(sev)
        if not items:
            continue
        print(f"\n【{headers[sev]}】共 {len(items)} 条")
        for f in items:
            loc = f" (第 {f.line} 行)" if f.line else ""
            print(f"  {icons[sev]} [{f.rule}]{loc} {f.message}")

    n_risk = len(report.by_severity(RISK))
    n_deduct = len(report.by_severity(DEDUCT))
    n_hint = len(report.by_severity(HINT))
    print(f"\n{'-' * 62}")
    print(f"  资格风险 {n_risk} · 扣分项 {n_deduct} · 提示 {n_hint}")
    if n_risk == 0 and n_deduct == 0:
        print("  未发现资格风险与扣分项。")
    print(f"{'-' * 62}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="国赛论文格式与数字溯源检查")
    ap.add_argument("paper", type=Path, help="论文文件（.md/.txt/.tex/.docx/.pdf）")
    ap.add_argument("--results", type=Path, default=None, help="results.json 路径，用于数字溯源")
    ap.add_argument("--data", type=Path, nargs="*", default=None,
                    help="题目附件（目录或 csv/json/txt 文件），其中的数字也算合法出处，避免题给常数被误报")
    ap.add_argument("--strict", action="store_true", help="存在任何扣分项也返回非零退出码")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出，便于程序处理")
    args = ap.parse_args()

    if not args.paper.exists():
        sys.exit(f"论文文件不存在：{args.paper}")

    text = read_paper(args.paper)
    report = run_checks(text, args.results, args.data, args.paper.suffix.lower())

    if args.json:
        print(json.dumps(
            [{"severity": f.severity, "rule": f.rule, "message": f.message, "line": f.line}
             for f in report.findings],
            ensure_ascii=False, indent=2,
        ))
    else:
        print_report(report, args.paper)

    if report.by_severity(RISK):
        return 1
    if args.strict and report.by_severity(DEDUCT):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
