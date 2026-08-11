#!/usr/bin/env python3
"""生成论文正文图件（PNG 预览 + PDF 矢量），并写出图注清单。

每张图只承载一个结论，图注写结论而不是图名。所有数据图都从 results/*.json 或附件读取，
没有任何一条曲线是手画的。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

from microstructure import CONTROL_ORIENTATION, PRIMARY_ORIENTATION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from setup_cn_plot import (COLORS, LINESTYLES, MARKERS, savefig_bundle,  # noqa: E402
                           seq_cmap, setup)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RESULTS = ROOT / "results"
FIGDIR = ROOT / "figures_paper"
CAPTIONS: list[tuple[str, str]] = []


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def emit(fig, stem: str, caption: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    savefig_bundle(fig, FIGDIR / stem, formats=("png", "pdf"))
    CAPTIONS.append((stem, caption))
    print(f"  ✓ {stem}")


# ------------------------------------------------------------------ 图1 方向分布
def fig_orientation() -> None:
    audit = load("data_audit.json")
    if not audit:
        return
    g = audit["groups"]["组3"]
    from load_attachment import group_by_medium, load_pieces
    pieces = load_pieces()["组3"]
    ux = []
    for idx in group_by_medium(pieces):
        d = pieces[idx[0]][3:] - pieces[idx[0]][:3]
        ux.append(abs(d[0] / np.linalg.norm(d)))
    ux = np.sort(np.array(ux))
    emp = np.arange(1, len(ux) + 1) / len(ux)

    t = np.linspace(0, 1, 400)
    cdf_iso = t
    cdf_pol = (np.pi - 2 * np.arccos(np.clip(t, 0, 1))) / np.pi

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.43))
    ax = axes[0]
    ax.step(ux, emp, where="post", color=COLORS[5], lw=2.0, label="附件组 3 经验分布 (n=354)")
    ax.plot(t, cdf_iso, color=COLORS[1], ls=LINESTYLES[1], lw=1.8, label="球面均匀（各向同性）")
    ax.plot(t, cdf_pol, color=COLORS[0], ls=LINESTYLES[2], lw=1.8, label="θ~U(0,π)，极轴为 X")
    ax.set_xlabel(r"$|u_x|$（介质轴与带电面法向夹角的余弦绝对值）")
    ax.set_ylabel("累积分布 F")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"KS 检验：各向同性 p={g['ks_isotropic']['p']:.1e}，"
                 f"极角均匀 p={g['ks_polar_uniform']['p']:.2f}", fontsize=9)

    ax = axes[1]
    labels = [r"$E|u_x|$", r"$E|u_y|$", r"$E|u_z|$"]
    x = np.arange(3)
    ax.bar(x - 0.26, g["mean_abs_u"], 0.26, color=COLORS[5], label="附件组 3")
    ax.bar(x, audit["reference_means"]["polar_uniform_about_x"], 0.26,
           color=COLORS[0], label="θ~U(0,π)")
    ax.bar(x + 0.26, audit["reference_means"]["isotropic"], 0.26,
           color=COLORS[1], label="各向同性")
    ax.set_xticks(x, labels)
    ax.set_ylabel("方向余弦绝对值的期望")
    ax.legend(fontsize=8)
    ax.set_title("介质明显偏向带电面法向（X）", fontsize=9)
    fig.tight_layout()
    emit(fig, "03_orientation_audit",
         "组 3 介质方向的经验分布与两种候选分布")


# ------------------------------------------------------------------ 图2 碎片审计
def fig_fragments() -> None:
    audit = load("data_audit.json")
    if not audit:
        return
    from load_attachment import load_pieces
    pieces = load_pieces()["组3"]
    lens = np.linalg.norm(pieces[:, 3:] - pieces[:, :3], axis=1)
    dropped = np.array(audit["groups"]["组3"]["dropped_per_medium_nm"])

    fig, ax = plt.subplots(figsize=(7.2, 3.43))
    bins = np.linspace(0, 5000, 51)
    ax.hist(lens, bins=bins, color=COLORS[0], alpha=0.85, label="附件中保留的碎片 (n=535)")
    ax.hist(dropped, bins=bins, color=COLORS[1], alpha=0.85,
            label="由长度守恒反推出的缺失碎片 (n=49)")
    ax.axvline(500, color=COLORS[5], ls=LINESTYLES[1], lw=1.6)
    ax.annotate("500 nm：保留的最短碎片为 526 nm，\n缺失碎片 46/49 短于 500 nm",
                xy=(500, ax.get_ylim()[1] * 0.62), xytext=(1500, ax.get_ylim()[1] * 0.72),
                fontsize=8, arrowprops=dict(arrowstyle="->", color=COLORS[5], lw=1.2))
    ax.set_xlabel("碎片轴向长度 / nm")
    ax.set_ylabel("碎片数")
    ax.legend(fontsize=8)
    fig.tight_layout()
    emit(fig, "02_fragment_audit",
         "附件保留碎片与反推出的缺失碎片的长度分布")


# ------------------------------------------------------------------ 图3 问题一
def fig_p1() -> None:
    """三组接触图在 X–Y 平面的投影，按**真实的棒—棒团簇**着色。

    这里的着色口径必须讲清楚，早先的版本在这上面出过错。判定导通时会往并查集里加两个
    虚拟电极节点 L、R，凡触及同一带电面的碎片都会被它们合并成一个连通块——那是**判定用的
    辅助结构，不是物理团簇**。若按那个口径着色，组 3 会有 258/535（48.2%）的碎片被涂成
    同一个"贯通团簇"，看上去像一个吞掉半个盒子的巨团簇，与第 5.3 节"贯通靠短链、
    此时尚未形成巨团簇"的结论正好相反。

    本函数只用**碎片之间的接触**建团簇（不含电极节点），再看每个团簇是否触到带电面：
    同时触到左右两面的团簇才是真正的贯通团簇。按这个口径，组 3 的贯通团簇只有 17 个碎片
    （3.2%），组 2 只有 5 个（10.2%）——图与第 5.3 节这才是一致的。
    """
    res = load("p1_connectivity.json")
    if not res:
        return
    from collections import Counter
    from load_attachment import GROUP_BOX, load_pieces
    from microstructure import EDGE, GAP, R_A, DSU, contact_pairs
    data = load_pieces()

    # 组 1、组 2 的截面只有 1000 nm（长宽比 10:1），按各组真实长宽比分配面板高度。
    # 注意由此三个面板的纵轴量程相差 10 倍，题注里已注明。
    fig, axes = plt.subplots(3, 1, figsize=(7.1, 5.4),
                             gridspec_kw={"height_ratios": [0.72, 0.72, 2.35]})
    for ax, name in zip(axes, ["组1", "组2", "组3"]):
        pieces = data[name]
        sp, sq = pieces[:, :3], pieces[:, 3:]
        n = len(sp)
        half = EDGE / 2

        # 只用碎片之间的接触建团簇——不加电极节点
        dsu = DSU(n)
        for a, b in contact_pairs(sp, sq, 2 * R_A + GAP):
            dsu.union(int(a), int(b))
        root = [dsu.find(i) for i in range(n)]
        lo = np.minimum(sp[:, 0], sq[:, 0]) - R_A
        hi = np.maximum(sp[:, 0], sq[:, 0]) + R_A
        touch_l = {root[i] for i in np.nonzero(lo <= -half + GAP)[0]}
        touch_r = {root[i] for i in np.nonzero(hi >= half - GAP)[0]}
        spanning = touch_l & touch_r            # 同时触到两面 = 真正的贯通团簇
        n_span = sum(Counter(root)[r] for r in spanning)

        for i in range(n):
            r = root[i]
            if r in spanning:
                c, lw, z = COLORS[2], 2.6, 4
            elif r in touch_l:
                c, lw, z = COLORS[1], 1.2, 2
            elif r in touch_r:
                c, lw, z = COLORS[0], 1.2, 2
            else:
                c, lw, z = "0.80", 0.7, 1
            ax.plot([sp[i, 0], sq[i, 0]], [sp[i, 1], sq[i, 1]], color=c, lw=lw, zorder=z)
        ax.axvline(-half, color="k", lw=2.2)
        ax.axvline(half, color="k", lw=2.2)
        hy = GROUP_BOX[name][1] / 2
        ax.set_ylim(-hy * 1.05, hy * 1.05)
        ax.set_xlim(-half * 1.04, half * 1.04)
        g = res[name]
        span_txt = (f"贯通团簇 {n_span}/{n} 个碎片" if spanning else "无贯通团簇")
        ax.set_title(f"{name}：{g['n_media']} 根介质 A（{g['n_pieces_in_attachment']} 个碎片），"
                     f"截面 {int(GROUP_BOX[name][1])}×{int(GROUP_BOX[name][2])} nm —— "
                     f"{'导通' if g['as_given'] else '不导通'}（{span_txt}）", fontsize=8.5)
        ax.set_ylabel("Y / nm")
    axes[-1].set_xlabel("X / nm（左右两条粗黑线为带电面；三个面板的纵轴量程不同）")

    # 图例放到整张图下方，避免像旧版那样压住组 1 面板里本就稀疏的碎片
    handles = [Line2D([], [], color=COLORS[2], lw=2.6, label="贯通团簇（同一团簇同时触及左右带电面）"),
               Line2D([], [], color=COLORS[1], lw=1.2, label="仅触及左带电面的团簇"),
               Line2D([], [], color=COLORS[0], lw=1.2, label="仅触及右带电面的团簇"),
               Line2D([], [], color="0.80", lw=1.0, label="两面都不触及的团簇")]
    fig.legend(handles=handles, fontsize=7.5, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    emit(fig, "05_p1_connectivity",
         "三个微构体的接触图在 X–Y 平面的投影（按真实棒—棒团簇着色）")


# ------------------------------------------------------------------ 图4 问题二/三
def fig_p2p3() -> None:
    p2, p3 = load("p2_probabilities.json"), load("p3_threshold.json")
    if not p2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.87))
    for k, mode in enumerate((PRIMARY_ORIENTATION, CONTROL_ORIENTATION)):
        rows = p2["runs"].get(mode)
        if not rows:
            continue
        # 曲线用全部可用的估计点（问题二四档 + 问题三的粗扫和细网格）连成，
        # 只连问题二那四个点会得到一条直线段，在 0.7%–1.0% 之间明显低于真实曲线，
        # 看上去像是细网格点和主曲线矛盾。
        pts = [(r["phi_a"], r["p"]) for r in rows]
        if p3 and p3["modes"].get(mode):
            m3 = p3["modes"][mode]
            pts += [(r["phi_a"], r["p"]) for r in m3["coarse"]]
            pts += [(r["phi_a"], r["p"]) for r in m3["fine_grid"]]
        pts = sorted(set(pts))
        cx = np.array([p for p, _ in pts]) * 100
        cy = np.array([q for _, q in pts])
        lab = ("球面均匀（主口径）" if mode == PRIMARY_ORIENTATION
               else "附件标定 θ~U(0,π)（灵敏度口径）")
        ax.plot(cx, cy, color=COLORS[k], ls=LINESTYLES[k], lw=1.6, alpha=0.9, label=lab)

        # 题目点名的四档单独标出并带 Wilson 区间
        x = np.array([r["phi_a"] for r in rows]) * 100
        y = np.array([r["p"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], color=COLORS[k], marker=MARKERS[k],
                    ms=7, ls="none", capsize=3, zorder=4)
        if p3 and p3["modes"].get(mode):
            ans = p3["modes"][mode]["answer_phi"]
            if ans:
                ax.plot([ans * 100], [0.9], marker="*", ms=15, color=COLORS[k],
                        markeredgecolor="k", markeredgewidth=0.6, zorder=6)
                ax.annotate(f"{ans:.2%}", (ans * 100, 0.9), textcoords="offset points",
                            xytext=(4, -15), color=COLORS[k], fontsize=9)
    ax.axhline(0.9, color=COLORS[5], ls=LINESTYLES[3], lw=1.4)
    ax.text(0.505, 0.915, "P = 90%（问题三的目标线）", fontsize=8, color=COLORS[5])
    ax.set_xlabel("介质 A 体积分数 φ / %")
    ax.set_ylabel("微构体导通概率 P")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    emit(fig, "06_p2_probability_curve",
         "导通概率随介质 A 体积分数的变化")


# ------------------------------------------------------------------ 图5 问题四
def fig_p4() -> None:
    """(N_A,N_B) 平面上的概率曲面、可行边界与等成本线。

    改动三处：
    1. 曲面改用**单色蓝渐变**。旧版用 viridis（紫—绿—黄多色），
       读者无法从色相判断哪端概率更高，灰度打印也不单调；
    2. 只画一条等成本线（最优成本）。旧版另画一条"成本高 15%"的白虚线，
       与前者几乎平行且同色，除了让人误以为是可行边界之外没有信息；
    3. 注记框移出数据区，星号缩小。
    """
    p4 = load("p4_cost_optimum.json")
    if not p4:
        return
    grid = p4["grid"]
    na = np.array([g["n_a"] for g in grid], float)
    nb = np.array([g["n_b"] for g in grid], float)
    pp = np.array([g["p"] for g in grid], float)
    ua, ub = np.unique(na), np.unique(nb)
    P = np.full((len(ub), len(ua)), np.nan)
    for a, b, v in zip(na, nb, pp):
        P[np.searchsorted(ub, b), np.searchsorted(ua, a)] = v

    fig, ax = plt.subplots(figsize=(7.2, 3.87))
    im = ax.pcolormesh(ua, ub, P, cmap=seq_cmap(), shading="gouraud",
                       vmin=0, vmax=1, rasterized=True)
    cb = fig.colorbar(im, ax=ax, label="导通概率 $P$", pad=0.02)
    cb.outline.set_visible(False)

    # 等高线自带的 clabel 会沿线旋转排版，中文竖着转 60° 基本没法读；
    # 改为在曲线一端放一个水平标注。
    cs = ax.contour(ua, ub, P, levels=[0.9], colors=[COLORS[1]], linewidths=1.8)
    seg = cs.allsegs[0][0] if cs.allsegs and cs.allsegs[0] else None
    if seg is not None and len(seg) > 2:
        hx, hy = seg[0]
        ax.annotate("$P=90\\%$ 可行边界", xy=(hx, hy), xytext=(10, -6),
                    textcoords="offset points", fontsize=8.5, color=COLORS[1],
                    ha="left", va="top")

    best = p4["best"]
    ca, cb_ = p4["cost_per_rod_yuan"], p4["cost_per_sphere_yuan"]
    xs = np.linspace(ua.min() - 20, ua.max() + 30, 120)
    ax.plot(xs, (best["cost_yuan"] - xs * ca) / cb_, color="w", ls="--", lw=1.4,
            label=f"等成本线 {best['cost_yuan']:.2f} 元")

    for v in p4.get("verified", []):
        ax.plot([v["n_a"]], [v["n_b"]], marker="o", ms=4.5, mfc="none",
                mec="w", mew=1.1, zorder=5)
    ax.plot([best["n_a"]], [best["n_b"]], marker="*", ms=13, color=COLORS[3],
            mec="w", mew=0.9, zorder=6, label="推荐配比 $(608,0)$")

    ax.legend(loc="lower left", fontsize=8, framealpha=0.92,
              facecolor="w", edgecolor="none")
    ax.set_xlabel("介质 A 根数 $N_A$")
    ax.set_ylabel("介质 B 颗数 $N_B$")
    ax.set_title("等成本线（虚线）与可行边界（实线）在 $N_B=0$ 处相切",
                 fontsize=9.5, color="#52514e")
    ax.set_xlim(ua.min() - 20, ua.max() + 30)
    ax.set_ylim(-140, ub.max() + 80)
    ax.grid(False)
    fig.tight_layout()
    emit(fig, "08_p4_cost_optimum",
         "$(N_A,N_B)$ 平面上的导通概率、可行边界与等成本线")


# ------------------------------------------------------------------ 图6 灵敏度
def fig_sensitivity() -> None:
    s = load("sensitivity.json")
    if not s:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.43))
    ax = axes[0]
    g = s.get("gap_scan", [])
    if g:
        x = [r["gap"] for r in g]
        y = [r["p"] for r in g]
        lo = [r["ci_lo"] for r in g]
        hi = [r["ci_hi"] for r in g]
        ax.errorbar(x, y, yerr=[np.array(y) - lo, np.array(hi) - y], color=COLORS[0],
                    marker="o", capsize=3, lw=1.8)
        ax.axvline(1.8, color=COLORS[1], ls=LINESTYLES[1], lw=1.4)
        ax.annotate("题给值 δ=1.8", xy=(1.8, 0.06), fontsize=8, color=COLORS[1],
                    ha="center")
        ax.set_xlabel("导通判据阈值 δ / nm")
        ax.set_ylabel("导通概率 P")
        # 纵轴按 0–1 全量程画：δ 变动 ±50% 只挪动 0.048，放大纵轴会把
        # "不敏感"画成一条陡线，与结论相反。右图同样用 0–1 横轴，两图可直接对照。
        ax.set_ylim(0, 1.05)
        ax.annotate(f"δ 变动 ±50%，P 仅在 {min(y):.3f}–{max(y):.3f} 之间移动（极差 {max(y)-min(y):.3f}）",
                    xy=(0.5, 0.93), xycoords="axes fraction", ha="center", fontsize=8)
        ax.set_title("对判据阈值的敏感性（φ=0.70% 固定）", fontsize=9)

    ax = axes[1]
    rows = s.get("assumption_table", [])
    if rows:
        names = [r["name"] for r in rows]
        vals = [r["p"] for r in rows]
        err = [[v - r["ci_lo"] for v, r in zip(vals, rows)],
               [r["ci_hi"] - v for v, r in zip(vals, rows)]]
        y = np.arange(len(names))
        ax.barh(y, vals, xerr=err, color=[COLORS[0]] + [COLORS[3]] * (len(names) - 1),
                capsize=3, height=0.6)
        ax.set_yticks(y, names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("导通概率 P")
        ax.set_xlim(0, 1.05)
        ax.set_title("建模口径对结论的影响（同一 φ=0.70%）", fontsize=9)
    fig.tight_layout()
    emit(fig, "11_sensitivity",
         "导通概率对判据阈值与建模口径的敏感性")



# ------------------------------------------------------------------ 图1 技术路线
def fig_roadmap() -> None:
    """黑白技术路线图：突出公共内核、四问调用和校验闭环。"""
    from matplotlib.patches import FancyArrowPatch, Rectangle

    ink = "#111111"
    mid = "#6f6f6f"
    pale = "#f1f1f1"
    white = "#ffffff"

    fig, ax = plt.subplots(figsize=(7.4, 4.28))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 74)
    ax.axis("off")

    def rect(x, y, w, h, *, fc=white, ec=ink, lw=1.0, ls="-"):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc,
                               edgecolor=ec, linewidth=lw, linestyle=ls))

    def arrow(x1, y1, x2, y2, *, dashed=False, ms=9):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=ms,
            linewidth=0.95, color=ink, linestyle="--" if dashed else "-",
            shrinkA=0, shrinkB=0,
        ))

    def input_node(x, w, title, detail=""):
        y, h = 59.0, 11.0
        rect(x, y, w, h, lw=1.05)
        if detail:
            ax.text(x + w / 2, y + 7.2, title, ha="center", va="center",
                    fontsize=8.7, fontweight="bold", color=ink)
            ax.text(x + w / 2, y + 3.3, detail, ha="center", va="center",
                    fontsize=7.3, color=mid, linespacing=1.25)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=8.8, fontweight="bold", color=ink)

    def problem_node(x, number, title, detail):
        y, w, h, head_h = 16.0, 22.0, 13.5, 4.1
        rect(x, y, w, h, lw=1.0)
        rect(x, y + h - head_h, w, head_h, fc=ink, lw=0)
        ax.text(x + w / 2, y + h - head_h / 2,
                f"问题{number}  {title}", ha="center", va="center",
                fontsize=7.8, fontweight="bold", color=white)
        ax.text(x + w / 2, y + (h - head_h) / 2, detail,
                ha="center", va="center", fontsize=7.1, color=ink,
                linespacing=1.35)

    # 1. 输入与建模口径：同排等高，保持阅读方向单一。
    ax.text(1.0, 71.7, "01  输入与建模口径", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=mid)
    input_node(1.0, 21.5, "赛题与附件")
    input_node(28.0, 42.0, "数据审计", "周期盒尺寸 / 碎片口径 / 方向分布")
    input_node(75.5, 23.5, "建模口径", "形成假设 A1-A5")
    arrow(22.5, 64.5, 27.2, 64.5)
    arrow(70.0, 64.5, 74.7, 64.5)

    # 2. 公共判定内核：粗框表示全文唯一复用的算法内核。
    ax.text(1.0, 55.8, "02  公共导通判定内核（四问共用）", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=mid)
    kx, ky, kw, kh = 1.0, 39.0, 98.0, 14.0
    rect(kx, ky, kw, kh, fc=pale, lw=1.55)
    step_x = (4.0, 36.0, 68.0)
    step_w = (25.0, 27.0, 27.0)
    step_text = (
        ("1  边界截断", "越界介质折回周期盒"),
        ("2  距离连边", "表面最短距离 ≤ 1.8 nm"),
        ("3  连通判定", "并查集检验两带电面贯通"),
    )
    for x, w, (title, detail) in zip(step_x, step_w, step_text):
        rect(x, 41.3, w, 8.2, fc=white, lw=0.85)
        ax.text(x + w / 2, 46.9, title, ha="center", va="center",
                fontsize=7.9, fontweight="bold", color=ink)
        ax.text(x + w / 2, 43.6, detail, ha="center", va="center",
                fontsize=6.8, color=mid)
    arrow(29.0, 45.4, 35.0, 45.4, ms=8)
    arrow(63.0, 45.4, 67.0, 45.4, ms=8)
    arrow(87.2, 59.0, 87.2, 53.5)

    # 3. 四问调用：黑色标题条建立清晰层级，正文只保留方法与交付物。
    ax.text(1.0, 35.7, "03  分问题求解", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=mid)
    px = (1.0, 26.3, 51.6, 76.9)
    problem_node(px[0], "一", "确定性判定", "还原附件位姿\n输出三组导通性")
    problem_node(px[1], "二", "概率估计", "蒙特卡洛抽样\n$\\hat P$ + Wilson 区间")
    problem_node(px[2], "三", "阈值反解", "logit 粗定位 + 网格复算\n反解 $P\\geq0.9$ 的 $\\varphi_{min}$")
    problem_node(px[3], "四", "成本优化", "概率约束整数搜索\n$\\min C$ s.t. $P\\geq0.9$")

    # 一条总线把公共内核分发到四问，避免四条放射斜线交叉。
    centers = [x + 11.0 for x in px]
    ax.plot([centers[0], centers[-1]], [34.0, 34.0], color=ink, lw=0.9)
    arrow(50.0, 39.0, 50.0, 34.0, ms=8)
    for cx in centers:
        arrow(cx, 34.0, cx, 30.2, ms=8)

    # 4. 校验闭环：虚线统一表示核验与反馈，不与主计算流混淆。
    ax.text(1.0, 12.7, "04  独立校验与结果审计", ha="left", va="center",
            fontsize=7.4, fontweight="bold", color=mid)
    rect(1.0, 2.3, 98.0, 8.0, fc=white, lw=0.95, ls="--")
    ax.text(50.0, 6.3,
            "几何内核 6 项校验   |   Wilson 区间   |   灵敏度分析   |   数字溯源",
            ha="center", va="center", fontsize=7.4, color=ink)
    for cx in centers:
        arrow(cx, 16.0, cx, 10.7, dashed=True, ms=7)

    fig.tight_layout()
    emit(fig, "01_roadmap", "技术路线：四问共用同一个导通判定内核")


# ------------------------------------------------------------------ 图4 截断示意
def fig_schematic() -> None:
    """边界截断规则与导通判据的示意图（非真实数据，比例已放大）。"""
    from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

    fig = plt.figure(figsize=(9.4, 3.43))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.35], hspace=0.55, wspace=0.22)

    # ---- (a) 边界截断
    ax = fig.add_subplot(gs[:, 0])
    ax.add_patch(Rectangle((-5, -5), 10, 10, fc="#fbfbfb", ec="0.6", lw=1.0, ls="--"))
    ax.plot([-5, -5], [-5, 5], color="k", lw=3.5)
    ax.plot([5, 5], [-5, 5], color="k", lw=3.5)
    ax.text(-5, 5.4, "带电面", ha="center", fontsize=8)
    ax.text(5, 5.4, "带电面", ha="center", fontsize=8)
    ax.plot([0.4, 5.0], [-1.2, 1.6], color=COLORS[0], lw=4, solid_capstyle="round")
    ax.plot([-5.0, -3.1], [1.6, 2.75], color=COLORS[1], lw=4, ls=(0, (4, 2)),
            solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((4.6, 2.5), (-4.4, 2.5), arrowstyle="-|>",
                                 mutation_scale=12, lw=1.2, color=COLORS[1],
                                 linestyle=":", shrinkA=0, shrinkB=0))
    ax.text(0.1, 3.0, "平移一个边长 $L$", ha="center", fontsize=8, color=COLORS[1])
    ax.text(3.1, -1.6, "主段", fontsize=8.5, color=COLORS[0])
    ax.text(-4.6, 3.4, "折回段", fontsize=8.5, color=COLORS[1])
    ax.text(0, -4.4, "两段是彼此独立的导体（假设 A2）", ha="center", fontsize=8)
    ax.set_xlim(-6.6, 6.6); ax.set_ylim(-5.6, 6.0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) 边界截断规则", fontsize=9.5)

    # ---- (b) 棒—棒接触
    ax = fig.add_subplot(gs[0, 1])
    ax.plot([-4.4, -0.35], [0.28, 0.02], color=COLORS[0], lw=7, solid_capstyle="round")
    ax.plot([0.35, 4.4], [-0.02, -0.28], color=COLORS[0], lw=7, solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((-0.35, 0.02), (0.35, -0.02), arrowstyle="<|-|>",
                                 mutation_scale=9, lw=1.2, color=COLORS[1]))
    ax.text(0, 0.95, "表面最短距离 ≤ δ = 1.8 nm，判为导通",
            ha="center", fontsize=8.5, color=COLORS[1])
    ax.text(-4.4, -1.15, "介质 A：球柱体，r = 30 nm，轴长 5000 nm",
            fontsize=8, color=COLORS[0])
    ax.set_xlim(-5, 5); ax.set_ylim(-1.6, 1.5); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) 介质 A 之间的接触", fontsize=9.5)

    # ---- (c) 球做桥
    ax = fig.add_subplot(gs[1, 1])
    ax.plot([-4.4, -1.25], [0.30, 0.10], color=COLORS[0], lw=7, solid_capstyle="round")
    ax.plot([1.25, 4.4], [-0.10, -0.30], color=COLORS[0], lw=7, solid_capstyle="round")
    ax.add_patch(Circle((0.0, 0.0), 1.05, fc=COLORS[3], ec="0.3", lw=0.9, alpha=0.9))
    ax.add_patch(FancyArrowPatch((-1.25, 0.10), (1.25, -0.10), arrowstyle="<|-|>",
                                 mutation_scale=9, lw=1.2, color=COLORS[1]))
    ax.text(0, 1.45, "一颗 B 最多能跨接轴距 $2(R+r+\\delta)=463.6$ nm 的两根 A",
            ha="center", fontsize=8.5, color=COLORS[1])
    ax.text(1.35, -1.25, "介质 B：球，R = 200 nm", fontsize=8, color=COLORS[3])
    ax.set_xlim(-5, 5); ax.set_ylim(-1.9, 2.1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(c) 介质 B 充当棒间的桥", fontsize=9.5)

    emit(fig, "04_schematic", "边界截断规则与导通判据示意")


# ------------------------------------------------------------------ 图9 收敛性
def fig_convergence() -> None:
    s = load("sensitivity.json")
    if not s or not s.get("convergence"):
        return
    conv = s["convergence"]
    T = np.array([c["trials"] for c in conv], float)
    p = np.array([c["p"] for c in conv])
    lo = np.array([c["ci_lo"] for c in conv])
    hi = np.array([c["ci_hi"] for c in conv])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.17))
    ax = axes[0]
    ax.errorbar(T, p, yerr=[p - lo, hi - p], color=COLORS[0], marker="o",
                capsize=3, lw=1.6)
    ax.axhline(p[-1], color=COLORS[5], ls=LINESTYLES[3], lw=1.2)
    ax.set_xscale("log"); ax.set_xlabel("试验次数 $T$"); ax.set_ylabel("导通概率 P")
    ax.set_title("估计值随样本量的稳定过程（φ=0.70%）", fontsize=9)

    ax = axes[1]
    half = (hi - lo) / 2
    ax.loglog(T, half, color=COLORS[0], marker="o", lw=1.6, label="Wilson 区间半宽")
    ref = half[0] * np.sqrt(T[0] / T)
    ax.loglog(T, ref, color=COLORS[1], ls=LINESTYLES[1], lw=1.4,
              label=r"$\propto 1/\sqrt{T}$ 参考线")
    ax.set_xlabel("试验次数 $T$"); ax.set_ylabel("区间半宽")
    ax.legend(fontsize=8)
    ax.set_title("半宽按 1/√T 收窄", fontsize=9)
    fig.tight_layout()
    emit(fig, "12_convergence", "蒙特卡洛估计的收敛性")


# ------------------------------------------------------------------ 图10 团簇结构
def fig_clusters() -> None:
    """最大团簇占比与导通概率。

    两条曲线都是 [0,1] 上的无量纲比例，因此**共用同一根纵轴**。
    旧版用了双纵轴（twinx）：两个刻度的对齐方式是任意的，会凭空制造出
    "两条曲线在某处相交/分离"的假象，是图表最常见的误导性做法。
    共轴之后，"P 已接近 1 而 S_max 仍在 0.2 附近"这一核心事实
    由两条线的真实垂直距离直接读出，不再依赖刻度怎么摆。
    """
    c = load("cluster_stats.json")
    if not c:
        return
    rows = c["rows"]
    x = np.array([r["phi"] for r in rows]) * 100
    smax = np.array([r["s_max_mean"] for r in rows])
    sd = np.array([r["s_max_sd"] for r in rows])
    pp = np.array([r["p_percolate"] for r in rows])

    fig, ax = plt.subplots(figsize=(7.0, 3.43))
    ax.fill_between(x, np.clip(smax - sd, 0, 1), smax + sd,
                    color=COLORS[0], alpha=0.15, lw=0)
    ax.plot(x, pp, color=COLORS[1], ls=LINESTYLES[1], marker="s", ms=4.5,
            lw=1.6, label="导通概率 $P$")
    ax.plot(x, smax, color=COLORS[0], marker="o", ms=4.5, lw=1.6,
            label="最大团簇占全部碎片的比例 $S_{\\max}$")

    ax.set_xlabel("介质 A 体积分数 φ / %")
    ax.set_ylabel("比例（两者同为 [0,1] 无量纲量）")
    ax.set_ylim(0, 1.04)
    ax.set_xlim(x.min() - 0.03, x.max() + 0.09)
    ax.legend(fontsize=8.5, loc="upper left")

    # 只在最右端直接标注两条线的落点，不给每个点标数字
    ax.annotate(f"{pp[-1]:.2f}", (x[-1], pp[-1]), xytext=(6, -2),
                textcoords="offset points", color=COLORS[1], fontsize=8.5,
                va="center")
    ax.annotate(f"{smax[-1]:.2f}", (x[-1], smax[-1]), xytext=(6, -2),
                textcoords="offset points", color=COLORS[0], fontsize=8.5,
                va="center")
    # 用一段竖直标注把"同一体积分数下两者的差距"标出来，替代旧版横穿图面的箭头
    ax.annotate("", xy=(x[-1], pp[-1]), xytext=(x[-1], smax[-1]),
                arrowprops=dict(arrowstyle="<->", lw=1.0, color="#52514e"))
    ax.text(x[-1] - 0.02, (pp[-1] + smax[-1]) / 2,
            "P 已近 1，\n最大团簇仍只占约 %.0f%%" % (smax[-1] * 100),
            fontsize=8.5, ha="right", va="center", color="#52514e")
    fig.tight_layout()
    emit(fig, "07_cluster_structure", "最大团簇占比与导通概率随体积分数的变化")


# --------------------------------------------------- 图9 两种介质的边际效率
def fig_marginal() -> None:
    """每元钱买到多少 ΔP：两种介质的边际效率随 N_A 的变化。

    这是问题四"为什么不掺 B"的机理图。表 13 给的是同样的数，
    但交叉点的位置在表里要靠逐列比大小才看得出来，画成两条线一眼可辨。
    """
    t = load("theory_check.json")
    if not t or "marginal_efficiency" not in t:
        return
    me = t["marginal_efficiency"]
    ks = sorted(me, key=int)
    x = np.array([int(k) for k in ks], float)
    ea = np.array([me[k].get("dP_dNA_per_yuan", np.nan) for k in ks], float)
    eb = np.array([me[k].get("dP_dNB_per_yuan", np.nan) for k in ks], float)

    fig, ax = plt.subplots(figsize=(7.0, 3.34))
    ax.plot(x, ea, color=COLORS[0], marker="o", ms=4.5, lw=1.6, label="加介质 A")
    ax.plot(x, eb, color=COLORS[1], marker="s", ms=4.5, lw=1.6,
            ls=LINESTYLES[1], label="加介质 B")

    # 交叉点：A 的效率首次超过 B 的那一段
    cross = None
    for i in range(1, len(x)):
        if np.isfinite(ea[i - 1]) and ea[i - 1] < eb[i - 1] and ea[i] > eb[i]:
            cross = (x[i - 1] + x[i]) / 2
            break
    if cross is not None:
        ax.axvline(cross, color="#898781", lw=0.9, ls=":")
        ax.annotate(f"效率交叉 $N_A\\approx {cross:.0f}$", xy=(cross, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points",
                    fontsize=8.5, color="#52514e", va="top")

    # 可行域（P≥0.90 所需的 N_A）整段落在交叉点右侧——这才是"用不上 B"的原因
    p3 = load("p3_threshold.json")
    n_star = None
    if p3:
        n_star = p3["modes"].get("isotropic", {}).get("answer_n_a")
    if n_star:
        ax.axvspan(n_star, x.max() + 20, color=COLORS[0], alpha=0.07, lw=0)
        ax.annotate(f"可行域 $N_A\\geq {n_star}$\n（A 占优区段）",
                    xy=(n_star + 6, ax.get_ylim()[1] * 0.72), fontsize=8.5,
                    color="#52514e")

    ax.set_xlabel("介质 A 根数 $N_A$")
    ax.set_ylabel("每元钱换来的 $\\Delta P$")
    ax.set_xlim(x.min() - 15, x.max() + 20)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    emit(fig, "09_marginal_efficiency", "两种介质每元钱的边际效率及其交叉点")


# ------------------------------------------------- 图10 预算线审计的证据链
def fig_budget_audit() -> None:
    """预算线上每个点的 P 与 0.90 的关系——问题四结论的完整证据。

    表 15 有同样的数，但"哪些点被干净地排除、哪一个卡在 0.90 上"
    要对着置信区间逐行心算；画成带误差棒的图，0.90 这条线一穿而过，
    退化点自己跳出来。
    """
    a = load("p4_global_audit.json")
    if not a:
        return
    rows = list(a.get("budget_line_checks", []))
    if not rows:
        return
    rc = load("p4_marginal_recheck.json")
    big = {(r["n_a"], r["n_b"]): r for r in (rc or {}).get("points", [])}

    # p4_global_audit.json 的预算线记录只存了 ci_hi（判据只用上界），
    # 但误差棒必须两侧都有，否则棒子只朝上、把点在视觉上压到区间底部。
    # Wilson 区间由 (p̂, 试验数) 唯一确定，这里按同一公式补出下界。
    def wilson_pair(p_hat: float, n: int) -> tuple[float, float]:
        z = 1.959964
        den = 1.0 + z * z / n
        c = (p_hat + z * z / (2 * n)) / den
        h = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / den
        return c - h, c + h

    n_small = a.get("trials", 2500)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(7.4, 3.52))
    ax.axhline(0.90, color=COLORS[1], lw=1.4, ls="--", zorder=1)
    ax.annotate("$P=0.90$ 约束线", xy=(0, 0.90), xytext=(2, 5),
                textcoords="offset points", ha="left", fontsize=8.5,
                color=COLORS[1])

    for i, r in enumerate(rows):
        key = (r["n_a"], r["n_b"])
        b = big.get(key)
        if b:
            pt, lo, hi = b["p"], b["ci_lo"], b["ci_hi"]
        else:
            pt = r["p"]
            lo, hi = wilson_pair(pt, n_small)
            hi = r.get("ci_hi", hi)          # 上界以文件里存的为准
        undecided = not (hi < 0.90 or lo >= 0.90)
        col = COLORS[1] if undecided else COLORS[0]
        ax.errorbar([i], [pt], yerr=[[pt - lo], [hi - pt]], color=col,
                    marker="o" if not b else "D", ms=5 if not b else 6,
                    capsize=3, lw=1.4, zorder=3)
        if b:
            ax.annotate("$T$=20000", (i, lo), xytext=(0, -14),
                        textcoords="offset points", ha="center",
                        fontsize=7.5, color="#898781")

    ax.set_xticks(x)
    ax.set_xticklabels([f"({r['n_a']},{r['n_b']})" for r in rows],
                       rotation=30, ha="right", fontsize=8)
    ax.set_xlabel("预算线上的整数配比 $(N_A,\\,N_B)$，成本均低于 9.0252 元")
    ax.set_ylabel("导通概率 $P$（误差棒为 Wilson 95% 区间）")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    last = rows[-1]
    ax.annotate("区间横跨 0.90：\n可行性无法判定", xy=(len(rows) - 1, 0.9003),
                xytext=(-20, -52), textcoords="offset points", ha="right",
                fontsize=8.5, color=COLORS[1],
                arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.1))
    ax.set_title("除最右一点外，预算线上所有更便宜的配比都被严格排除",
                 fontsize=9.5, color="#52514e")
    fig.tight_layout()
    emit(fig, "10_budget_audit", "预算线上逐点审计：$P$ 的点估计与 Wilson 区间")


# ------------------------------------------------- 图13 球柱体近似的双侧界
def fig_bracket() -> None:
    """内接/外接球柱体给出的 P 双侧界，以及它如何把问题三的答案夹成区间。"""
    g = load("geometry_bracket.json")
    if not g:
        return
    p3 = g.get("problem3", [])
    if not p3:
        return
    x = np.array([r["phi"] for r in p3]) * 100
    lo = np.array([r["lower"]["p"] for r in p3])
    hi = np.array([r["upper"]["p"] for r in p3])

    fig, ax = plt.subplots(figsize=(7.0, 3.43))
    ax.fill_between(x, lo, hi, color=COLORS[0], alpha=0.18, lw=0,
                    label="真实平端面圆柱的 $P$ 必落在此带内")
    ax.plot(x, hi, color=COLORS[0], marker="o", ms=4.5, lw=1.6,
            label="外界 $P(K^+)$：球柱体（轴长 5000）")
    ax.plot(x, lo, color=COLORS[2], marker="^", ms=4.5, lw=1.6,
            ls=LINESTYLES[2], label="内界 $P(K^-)$：轴长 4940")
    ax.axhline(0.90, color=COLORS[1], lw=1.4, ls="--")
    ax.annotate("$P=0.90$", xy=(x.min(), 0.90), xytext=(2, 5),
                textcoords="offset points", fontsize=8.5, color=COLORS[1])

    lo_ok = [r["phi"] for r in p3 if r["lower_ok"]]
    hi_ok = [r["phi"] for r in p3 if r["upper_ok"]]
    if hi_ok and lo_ok:
        a, b = min(hi_ok) * 100, min(lo_ok) * 100
        ax.axvspan(a, b, color=COLORS[1], alpha=0.10, lw=0)
        ax.annotate(f"答案被夹在 [{a:.2f}%, {b:.2f}%]",
                    xy=((a + b) / 2, 0.97), xycoords=("data", "axes fraction"),
                    ha="center", va="top", fontsize=8.5, color="#52514e")
        ax.plot([b], [0.90], marker="*", ms=13, color=COLORS[1],
                mec="w", mew=0.9, zorder=6)
        ax.annotate(f"可证达标 {b:.2f}%", xy=(b, 0.90), xytext=(6, -16),
                    textcoords="offset points", fontsize=8.5, color=COLORS[1])

    ax.set_xlabel("介质 A 体积分数 φ / %")
    ax.set_ylabel("导通概率 $P$")
    ax.set_xlim(x.min() - 0.005, x.max() + 0.005)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    emit(fig, "13_geometry_bracket", "球柱体近似的严格双侧界与问题三答案的夹逼")


def main() -> int:
    font = setup(font_size=10)
    print(f"中文字体：{font}")
    # 顺序 = 正文中出现的先后。文件名前缀即图号，改动顺序时两者一起改，
    # 避免出现"图 10 排在图 7 前面"这类编号与出现次序不一致的问题。
    for f in (fig_roadmap, fig_orientation, fig_fragments, fig_schematic,
              fig_p1, fig_p2p3, fig_clusters, fig_p4, fig_marginal,
              fig_budget_audit, fig_sensitivity, fig_convergence, fig_bracket):
        try:
            f()
        except Exception as e:  # 单张图失败不该阻断其余图
            print(f"  ✗ {f.__name__}: {type(e).__name__}: {e}")
    lines = ["# 图注清单\n",
             "| 文件 | 图注（正文结论） |", "|---|---|"]
    lines += [f"| `{s}.png` / `.pdf` | {c} |" for s, c in CAPTIONS]
    (FIGDIR / "图注清单.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n共 {len(CAPTIONS)} 张图 -> {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
