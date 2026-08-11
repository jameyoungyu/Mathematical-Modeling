"""matplotlib 中文显示与统一绘图样式。

数模比赛里最稳定复现的浪费时间方式，是画完图发现中文标签全是方块。
原因是 matplotlib 默认字体不含 CJK 字形。这个模块在导入时找一个系统上
真实存在的中文字体并配好，同时统一全文图表的字号与配色，让论文里的图
看起来像一套而不是七拼八凑。

用法：

    import sys; sys.path.insert(0, "scripts")
    from setup_cn_plot import setup, COLORS
    setup()

    fig, ax = plt.subplots()
    ax.plot(x, y, color=COLORS[0], label="实际值")
    ax.set_xlabel("时间 / 天")
    ax.set_ylabel("产量 / 吨")
    ax.legend()
    savefig(fig, "figures/p1_result.png")

若系统确实没有任何中文字体，setup() 会明确告警并给出安装建议，
而不是静默产出一堆方块图 —— 静默失败要到交论文前才会被发现。
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 按优先级排列的候选中文字体，覆盖 Windows / macOS / Linux 常见安装
CANDIDATE_FONTS = [
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "STHeiti",
    "Arial Unicode MS",
]

# Okabe-Ito 色盲友好配色。黑白打印仍需同时使用线型或标记，不能只靠颜色。
COLORS = [
    "#0072B2",  # 蓝
    "#D55E00",  # 朱红
    "#009E73",  # 蓝绿
    "#E69F00",  # 橙
    "#CC79A7",  # 紫红
    "#000000",  # 黑
    "#56B4E9",  # 天蓝
    "#F0E442",  # 黄
]

# 线型与标记，配合颜色使用，保证黑白打印下仍可区分
LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

# 表示"大小"的连续量必须用**单色深浅**渐变，不能用彩虹型（如 viridis/jet）：
# 多色渐变会把数值差异编码成色相差异，读者无法判断哪一端更大，
# 且在灰度打印下不单调。下面是蓝色系单色梯度，亮→暗对应小→大。
SEQ_BLUE_STOPS = [
    "#eaf2fd", "#cde2fb", "#9ec5f4", "#6da7ec",
    "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b",
]

# 图面 chrome：网格与轴线用比背景仅深一档的实线细发丝线，不用虚线
# （虚线会被误读成"阈值"或"外推"，而它只是网格）。
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def seq_cmap():
    """单色蓝渐变 colormap（亮=小，暗=大）。"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE_STOPS)


def find_chinese_font() -> str | None:
    """返回系统上第一个可用的中文字体名，找不到返回 None。"""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in CANDIDATE_FONTS:
        if name in installed:
            return name

    # 候选表全落空时，扫描字体文件看谁真的含有 CJK 字形。
    # 比再加一堆字体名到候选表更可靠，因为发行版命名差异很大。
    for font in font_manager.fontManager.ttflist:
        try:
            from matplotlib.ft2font import FT2Font

            if FT2Font(font.fname).get_char_index(ord("中")):
                return font.name
        except Exception:
            continue
    return None


def setup(font_size: int = 11, dpi: int = 300, style: str = "paper") -> str | None:
    """配置中文字体与论文绘图样式，返回实际使用的字体名。

    font_size: 正文字号。论文里的图会被缩小排版，10-12 比较合适。
    dpi:       保存图片的分辨率。论文插图用 300。
    style:     "paper" 为论文风格（白底、细网格）；"plain" 只配字体不改样式。
    """
    # 中文字体常缺少 weight 元数据，matplotlib 会为每个文本刷屏 findfont 警告。
    # 字体本身是能用的，这些警告只会淹没真正的报错。
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    font = find_chinese_font()
    if font:
        matplotlib.rcParams["font.sans-serif"] = [font] + CANDIDATE_FONTS
        matplotlib.rcParams["font.family"] = "sans-serif"
    else:
        warnings.warn(
            "未找到任何中文字体，图中的中文将显示为方块。\n"
            "  Ubuntu/Debian: sudo apt-get install fonts-noto-cjk\n"
            "  CentOS/RHEL:   sudo yum install google-noto-sans-cjk-ttc-fonts\n"
            "  conda:         conda install -c conda-forge font-ttf-source-han-sans\n"
            "  或把 SimHei.ttf 放到当前目录后调用 register_font_file('SimHei.ttf')",
            stacklevel=2,
        )

    # 中文字体的负号字形常缺失，会把 -1 画成方块1
    matplotlib.rcParams["axes.unicode_minus"] = False

    if style == "paper":
        matplotlib.rcParams.update(
            {
                "figure.dpi": 110,
                "savefig.dpi": dpi,
                "savefig.bbox": "tight",
                "savefig.facecolor": "white",
                "figure.figsize": (7.0, 4.3),
                "font.size": font_size,
                "axes.titlesize": font_size + 1,
                "axes.labelsize": font_size,
                "xtick.labelsize": font_size - 1,
                "ytick.labelsize": font_size - 1,
                "legend.fontsize": font_size - 1,
                "axes.grid": True,
                "grid.color": GRID,
                "grid.alpha": 1.0,
                "grid.linewidth": 0.5,
                "grid.linestyle": "-",
                "axes.edgecolor": AXIS,
                "axes.linewidth": 0.8,
                "xtick.color": INK_MUTED,
                "ytick.color": INK_MUTED,
                "xtick.labelcolor": "#52514e",
                "ytick.labelcolor": "#52514e",
                "axes.axisbelow": True,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.prop_cycle": matplotlib.cycler(color=COLORS),
                "lines.linewidth": 1.6,
                "lines.markersize": 4.5,
                "legend.frameon": False,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
    return font


def register_font_file(path: str | Path) -> str:
    """手动注册一个字体文件（比赛现场装不了系统字体时的兜底）。

    把 SimHei.ttf / NotoSansCJK.ttc 之类拷到项目里，调用本函数即可。
    """
    path = Path(path)
    font_manager.fontManager.addfont(str(path))
    name = font_manager.FontProperties(fname=str(path)).get_name()
    matplotlib.rcParams["font.sans-serif"] = [name] + CANDIDATE_FONTS
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["axes.unicode_minus"] = False
    return name


def savefig(fig, path: str | Path, close: bool = True, **kwargs) -> Path:
    """保存单个图片，自动建目录；默认保存后关闭图对象。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, **{"bbox_inches": "tight", **kwargs})
    if close:
        plt.close(fig)
    return path


def savefig_bundle(
    fig,
    path_stem: str | Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    **kwargs,
) -> list[Path]:
    """同时保存论文位图和矢量图，返回生成路径。

    ``path_stem`` 不带扩展名；PNG 用于预览和兼容，PDF/SVG 用于论文排版。
    """
    stem = Path(path_stem)
    outputs: list[Path] = []
    for index, fmt in enumerate(formats):
        suffix = fmt.lower().lstrip(".")
        outputs.append(savefig(fig, stem.with_suffix(f".{suffix}"), close=index == len(formats) - 1, **kwargs))
    return outputs


def check() -> None:
    """自检：画一张含中文和负号的测试图，确认字体配置生效。"""
    font = setup()
    print(f"使用字体：{font or '（未找到中文字体）'}")
    fig, ax = plt.subplots()
    for i in range(3):
        ax.plot(
            [-2, -1, 0, 1, 2],
            [v * (i + 1) for v in (-1.5, -0.5, 0.5, 1.5, 2.5)],
            color=COLORS[i],
            linestyle=LINESTYLES[i],
            marker=MARKERS[i],
            label=f"方案{'一二三'[i]}",
        )
    ax.set_xlabel("参数偏移 / %")
    ax.set_ylabel("目标函数值 / 万元")
    ax.set_title("中文字体与负号显示自检")
    ax.legend()
    out = savefig(fig, "figures/_font_check.png")
    print(f"已保存 {out}，打开确认中文与负号（-2、-1）显示正常。")


if __name__ == "__main__":
    check()
