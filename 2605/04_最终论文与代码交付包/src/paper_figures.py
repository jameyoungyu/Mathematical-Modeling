#!/usr/bin/env python3
"""生成论文正文图件（PNG 预览 + PDF 矢量），并写出图注清单。

每张图只承载一个结论，图注写结论而不是图名。所有数据图都从 results/*.json 或附件读取，
没有任何一条曲线是手画的。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from setup_cn_plot import COLORS, LINESTYLES, MARKERS, savefig_bundle, setup  # noqa: E402

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

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
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
    emit(fig, "01_orientation_audit",
         "附件的介质方向并非各向同性，而是以带电面法向为极轴的极角均匀分布；"
         "该偏向使同一体积分数下的导通概率显著升高。")


# ------------------------------------------------------------------ 图2 碎片审计
def fig_fragments() -> None:
    audit = load("data_audit.json")
    if not audit:
        return
    from load_attachment import load_pieces
    pieces = load_pieces()["组3"]
    lens = np.linalg.norm(pieces[:, 3:] - pieces[:, :3], axis=1)
    dropped = np.array(audit["groups"]["组3"]["dropped_per_medium_nm"])

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
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
         "附件系统性丢弃了长度小于约 500 nm 的碎片；补回这些碎片后问题一的三个判定结论不变。")


# ------------------------------------------------------------------ 图3 问题一
def fig_p1() -> None:
    res = load("p1_connectivity.json")
    if not res:
        return
    from load_attachment import GROUP_BOX, load_pieces
    from microstructure import EDGE, GAP, R_A, DSU, contact_pairs
    data = load_pieces()

    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2))
    for ax, name in zip(axes, ["组1", "组2", "组3"]):
        pieces = data[name]
        sp, sq = pieces[:, :3], pieces[:, 3:]
        n = len(sp)
        dsu = DSU(n + 2)
        L, R = n, n + 1
        half = EDGE / 2
        lo = np.minimum(sp[:, 0], sq[:, 0]) - R_A
        hi = np.maximum(sp[:, 0], sq[:, 0]) + R_A
        for i in np.nonzero(lo <= -half + GAP)[0]:
            dsu.union(L, int(i))
        for i in np.nonzero(hi >= half - GAP)[0]:
            dsu.union(R, int(i))
        for a, b in contact_pairs(sp, sq, 2 * R_A + GAP):
            dsu.union(int(a), int(b))
        span = dsu.find(L) == dsu.find(R)

        for i in range(n):
            rl, rr = dsu.find(i) == dsu.find(L), dsu.find(i) == dsu.find(R)
            if span and rl and rr:
                c, lw, z = COLORS[2], 2.4, 3
            elif rl:
                c, lw, z = COLORS[1], 1.2, 2
            elif rr:
                c, lw, z = COLORS[0], 1.2, 2
            else:
                c, lw, z = "0.78", 0.7, 1
            ax.plot([sp[i, 0], sq[i, 0]], [sp[i, 1], sq[i, 1]], color=c, lw=lw, zorder=z)
        ax.axvline(-half, color="k", lw=2.2)
        ax.axvline(half, color="k", lw=2.2)
        hy = GROUP_BOX[name][1] / 2
        ax.set_ylim(-hy * 1.05, hy * 1.05)
        ax.set_xlim(-half * 1.04, half * 1.04)
        r = res[name]
        ax.set_title(f"{name}：{r['n_media']} 根介质 A（{r['n_pieces_in_attachment']} 个碎片），"
                     f"截面 {int(GROUP_BOX[name][1])}×{int(GROUP_BOX[name][2])} nm —— "
                     f"{'导通' if r['as_given'] else '不导通'}", fontsize=9)
        ax.set_ylabel("Y / nm")
    axes[-1].set_xlabel("X / nm（左右两条粗黑线为带电面）")
    handles = [Line2D([], [], color=COLORS[2], lw=2.4, label="贯通团簇（同时连到左右带电面）"),
               Line2D([], [], color=COLORS[1], lw=1.2, label="与左带电面连通"),
               Line2D([], [], color=COLORS[0], lw=1.2, label="与右带电面连通"),
               Line2D([], [], color="0.78", lw=1.0, label="孤立团簇")]
    axes[0].legend(handles=handles, fontsize=7, ncol=2, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    emit(fig, "03_p1_connectivity",
         "问题一三个微构体的接触图在 X–Y 平面的投影：组 1 的两端团簇之间存在断口，组 2、"
         "组 3 则各有一个同时连到左右带电面的贯通团簇。")


# ------------------------------------------------------------------ 图4 问题二/三
def fig_p2p3() -> None:
    p2, p3 = load("p2_probabilities.json"), load("p3_threshold.json")
    if not p2:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for k, mode in enumerate(("polar_uniform", "isotropic")):
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
        lab = "附件标定分布 θ~U(0,π)" if mode == "polar_uniform" else "球面均匀（对照）"
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
    emit(fig, "04_p2_probability_curve",
         "导通概率随介质 A 体积分数的上升呈典型渗流跃变；带误差棒的标记为问题二点名的四档"
         "（Wilson 95% 区间），星号为问题三求得的使 P≥90% 的最低体积分数。")


# ------------------------------------------------------------------ 图5 问题四
def fig_p4() -> None:
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

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    im = ax.pcolormesh(ua, ub, P, cmap="viridis", shading="nearest", vmin=0, vmax=1)
    cs = ax.contour(ua, ub, P, levels=[0.9], colors=[COLORS[1]], linewidths=2.2)
    ax.clabel(cs, fmt={0.9: "P=90%"}, fontsize=8)
    fig.colorbar(im, ax=ax, label="导通概率 P")

    best = p4["best"]
    ca, cb = p4["cost_per_rod_yuan"], p4["cost_per_sphere_yuan"]
    xs = np.linspace(ua.min() - 20, ua.max() + 30, 120)
    for c, ls, lab in ((best["cost_yuan"], "-", "最优等成本线"),
                       (best["cost_yuan"] * 1.15, ":", "成本高 15%")):
        ax.plot(xs, (c - xs * ca) / cb, color="w", ls=ls, lw=1.6, label=lab)
    # 已直接验证过的边界点，说明最优不是拟合出来的
    for v in p4.get("verified", []):
        ax.plot([v["n_a"]], [v["n_b"]], marker="o", ms=5, mfc="none",
                mec="w", mew=1.2, zorder=5)
    ax.plot([best["n_a"]], [best["n_b"]], marker="*", ms=20, color=COLORS[3],
            markeredgecolor="k", markeredgewidth=0.8, zorder=6)
    ax.annotate(f"最优 $N_A$={best['n_a']}，$N_B$=0\n"
                f"总成本 {best['cost_yuan']:.3f} 元，$P$={best['p']:.4f}",
                xy=(best["n_a"], best["n_b"]), xycoords="data",
                xytext=(0.03, 0.90), textcoords="axes fraction",
                fontsize=8.5, color="w",
                arrowprops=dict(arrowstyle="->", color="w", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.35", fc="0.2", ec="none", alpha=0.9))
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.85)
    ax.set_xlabel("介质 A 根数 $N_A$")
    ax.set_ylabel("介质 B 颗数 $N_B$")
    ax.set_xlim(ua.min() - 20, ua.max() + 30)
    ax.set_ylim(-160, ub.max() + 120)
    fig.tight_layout()
    emit(fig, "05_p4_cost_optimum",
         "P=90% 的可行边界（红线）与等成本线（白线）在 $N_B=0$ 处相切：给定价格下掺入介质 B "
         "并不划算，最优方案就是纯介质 A。空心圆为经大样本直接验证的边界点。")


# ------------------------------------------------------------------ 图6 灵敏度
def fig_sensitivity() -> None:
    s = load("sensitivity.json")
    if not s:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
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
        ax.set_xlabel("导通判据阈值 δ / nm")
        ax.set_ylabel("导通概率 P")
        ax.set_title("对判据阈值的敏感性（φ 固定）", fontsize=9)

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
        ax.set_title("建模口径对结论的影响", fontsize=9)
    fig.tight_layout()
    emit(fig, "06_sensitivity",
         "导通概率对判据阈值 δ 在 ±50% 范围内不敏感，但对方向分布这一建模口径高度敏感。")


def main() -> int:
    font = setup(font_size=10)
    print(f"中文字体：{font}")
    for f in (fig_orientation, fig_fragments, fig_p1, fig_p2p3, fig_p4, fig_sensitivity):
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
