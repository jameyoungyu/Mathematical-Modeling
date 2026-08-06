"""Shared publication plotting style for the cement ESP paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from cycler import cycler


# Okabe-Ito 色序：色觉障碍友好，灰度打印仍可区分
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]
GRAY = "#5B6472"
LIGHT_GRAY = "#D1D5DB"
NEUTRAL_FILL = "#DFE3E8"


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Linux 优先 Noto，macOS 回落到系统中文字体
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "PingFang SC",
                "Arial Unicode MS",
                "WenQuanYi Zen Hei",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 9.0,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.4,
            "legend.borderaxespad": 0.4,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#2F3640",
            "axes.labelcolor": "#1F2933",
            "axes.prop_cycle": cycler(color=PALETTE),
            "lines.linewidth": 1.7,
            "lines.markersize": 5.0,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.color": "#2F3640",
            "ytick.color": "#2F3640",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.9,
            "figure.dpi": 160,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_axes(ax, grid_axis: str = "y") -> None:
    if grid_axis != "none":
        ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_subpanel_label(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    """Add a compact (a), (b), ... identifier to a multi-panel figure."""
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


def headroom(ax, frac: float = 0.18) -> None:
    """在数据上方留出空白，避免图例或注释压住柱顶与折线。"""
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, lo + (hi - lo) * (1.0 + frac))


def figure_legend(fig, handles, labels, ncol: int, bottom: float = -0.02) -> None:
    """把多面板共用图例统一放到画布下方，杜绝与数据重叠。"""
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, bottom),
        frameon=False,
    )


def note(ax, text: str, loc: str = "upper left", color: str | None = None) -> None:
    """角落说明文字，带白底以免压住网格与数据。"""
    anchor = {
        "upper left": (0.02, 0.98, "left", "top"),
        "upper right": (0.98, 0.98, "right", "top"),
        "lower left": (0.02, 0.03, "left", "bottom"),
        "lower right": (0.98, 0.03, "right", "bottom"),
    }[loc]
    ax.text(
        anchor[0],
        anchor[1],
        text,
        transform=ax.transAxes,
        ha=anchor[2],
        va=anchor[3],
        fontsize=8.0,
        color=color or GRAY,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": LIGHT_GRAY,
            "linewidth": 0.6,
            "alpha": 0.92,
        },
    )


def save_figure(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png")
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
