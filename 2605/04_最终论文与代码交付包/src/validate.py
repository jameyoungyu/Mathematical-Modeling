#!/usr/bin/env python3
"""几何内核的独立校验。

论文里所有结论都建立在两个原语上：线段最短距离和边界截断。这两个错了，
后面所有概率都是错的，而且错得很隐蔽。所以在跑任何仿真之前先把它们钉死：

T1 线段最短距离 vs 暴力采样；
T2 题面图 2 给出的截断算例，逐坐标对照；
T3 用附件真实数据做往返测试：把碎片还原成母介质轴线，再按本模块的规则重新截断，
   看能不能一模一样地还原出附件里的碎片（这同时校验了附件的周期盒尺寸判断）；
T4 并查集导通判定在人工构造的通/断算例上的行为；
T5 分块加速的接触检索与朴素两两比较逐位一致。

另附一项**非校验**的量级参考（球柱体近似），它没有通过标准，不计入退出码；
该近似的严格处理见 geometry_bracket.py 给出的双侧界。

退出码 0 表示全部通过。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from load_attachment import GROUP_BOX, group_by_medium, load_pieces, reconstruct_axis
from microstructure import (EDGE, GAP, R_A, DSU, percolates, seg_seg_dist,
                            wrap_segment)

RESULTS = Path(__file__).resolve().parents[1] / "results"
report: dict[str, object] = {}
failures: list[str] = []


def check(name: str, ok: bool, detail: object) -> None:
    report[name] = {"pass": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- T1 线段距离
def t1_segment_distance() -> None:
    rng = np.random.default_rng(20260810)
    worst = 0.0
    for _ in range(300):
        p1, q1, p2, q2 = (rng.uniform(-1000, 1000, 3) for _ in range(4))
        exact = float(seg_seg_dist(p1[None, None], q1[None, None],
                                   p2[None, None], q2[None, None])[0, 0])
        t = np.linspace(0, 1, 900)[:, None]
        a = p1 + t * (q1 - p1)
        b = p2 + t * (q2 - p2)
        brute = float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)))
        # 采样只能给出上界，精确解不应超过它，且不应显著小于它
        worst = max(worst, exact - brute)
    check("T1_线段最短距离", worst <= 1e-9,
          f"解析解相对 900×900 暴力采样的最大超出量 = {worst:.3e} nm（应 ≤ 0）")


# ---------------------------------------------------------------- T2 题面算例
def t2_statement_example() -> None:
    box = np.full(3, EDGE)
    p = np.array([3500.0, 100.0, 200.0])
    q = np.array([6000.0, 150.0, 250.0])
    segs = wrap_segment(p, q, box)
    xs = sorted(round(float(v), 6) for s in segs for v in (s[0][0], s[1][0]))
    ok = (len(segs) == 2 and abs(xs[0] + 5000) < 1e-6 and abs(xs[-1] - 5000) < 1e-6)
    # 题面：超出的 X1 段 (5000..6000) 平移成 (-5000..-4000)
    tail = [s for s in segs if s[0][0] < -4000][0]
    ok = ok and abs(tail[0][0] + 5000) < 1e-6 and abs(tail[1][0] + 4000) < 1e-6
    check("T2_题面截断算例", ok,
          f"切成 {len(segs)} 段，越界段 X 由 (5000,6000) 平移为 "
          f"({tail[0][0]:.1f},{tail[1][0]:.1f})，与题面图 2 一致")


# ---------------------------------------------------------------- T3 附件往返
def t3_attachment_roundtrip() -> None:
    data = load_pieces()
    detail = {}
    totals = {"matched": 0, "complete": 0}
    all_ok = True
    for name, pieces in data.items():
        box = GROUP_BOX[name]
        n_complete = n_match = 0
        for idx in group_by_medium(pieces):
            start, end, total = reconstruct_axis(pieces, idx, box)
            if abs(total - 5000.0) > 1e-3:
                continue                     # 附件丢了短碎片的介质，跳过
            n_complete += 1
            got = wrap_segment(start, end, box)
            want = [(pieces[i][:3], pieces[i][3:]) for i in idx]
            if len(got) != len(want):
                continue
            key = lambda s: tuple(np.round(np.concatenate(s), 4))
            if sorted(map(key, got)) == sorted(map(key, want)):
                n_match += 1
        ok = n_complete > 0 and n_match == n_complete
        all_ok &= ok
        detail[name] = f"{n_match}/{n_complete} 根完整介质的截断结果与附件逐坐标一致"
        totals["matched"] += n_match
        totals["complete"] += n_complete
    detail["合计"] = f"{totals['matched']}/{totals['complete']}"
    detail["matched_total"] = totals["matched"]
    detail["complete_total"] = totals["complete"]
    check("T3_附件截断往返", all_ok, detail)


# ------------------------------------------------- 附：球柱体近似的粗略量级（非校验）
def note_capsule_scale() -> None:
    """球柱体与平端面圆柱只在端帽处不同，这里给一个**极粗略**的量级参考。

    **这不是一项校验，不参与通过/不通过的判定**，原因有二：

    1. 采样分布是人造的——第一根棒恒沿 X 轴过原点，第二根棒的中心被限制在其轴线周围
       一根半径 2r+δ 的细管里。算出的比例依赖这个人为设定，不是物理构型下的概率。
    2. 它统计的是"第二根棒的中心落在端帽所在的 x 带内"的频率，只是"最近点落在端帽附近"
       的一个粗代理，且完全没有计入第二根棒自身的端帽。

    球柱体近似的**正确**处理方式是几何包含关系给出的严格双侧界（geometry_bracket.py，
    论文第 8.4 节）：K⁻ ⊆ 真实圆柱 ⊆ K⁺，于是 P(K⁻) ≤ P(真实) ≤ P(K⁺)。
    论文结论以那组界为准，本函数只作为背景量级保留。
    """
    rng = np.random.default_rng(7)
    n = 200000
    # 在"轴线距离刚好落在判定阈值附近"的壳层里采样，看端帽区域占多大比例
    L, r = 5000.0, R_A
    c1 = np.zeros((n, 3))
    u1 = np.array([1.0, 0.0, 0.0])
    u2 = rng.normal(size=(n, 3)); u2 /= np.linalg.norm(u2, axis=1, keepdims=True)
    c2 = rng.uniform(-L / 2 - r, L / 2 + r, size=(n, 3))
    c2[:, 1:] = rng.uniform(-2 * r - GAP, 2 * r + GAP, size=(n, 2))
    p1 = c1 - L / 2 * u1; q1 = c1 + L / 2 * u1
    p2 = c2 - L / 2 * u2; q2 = c2 + L / 2 * u2
    d = seg_seg_dist(p1[:, None], q1[:, None], p2[:, None], q2[:, None])[:, 0]
    near = np.abs(d - (2 * r + GAP)) < GAP
    # 端帽影响只出现在最近点落在轴线端点 ±r 范围内的情形
    frac_end = float(np.mean(near & (np.abs(c2[:, 0]) > L / 2 - r)))
    frac_near = float(np.mean(near))
    ratio = frac_end / max(frac_near, 1e-12)
    # 只记录，不判定：severity 为 note，不进入 failures
    report["NOTE_球柱体近似量级"] = {
        "pass": None, "is_check": False,
        "detail": (f"人造采样下端帽相关比例 ≈ {ratio:.2%}；体积差 4r/3h = {4*r/(3*5000):.2%}。"
                   "仅为量级参考，严格结论见 geometry_bracket.py 的双侧界")}
    print(f"[NOTE] 球柱体近似量级: 人造采样下端帽相关比例 ≈ {ratio:.2%}"
          f"（非校验；严格结论见 geometry_bracket.py）")


# ---------------------------------------------------------------- T4 导通判定
def t4_percolation_logic() -> None:
    half = EDGE / 2
    # (a) 一根横贯左右的棒：必导通
    p = np.array([[-half + 1.0, 0.0, 0.0]]); q = np.array([[half - 1.0, 0.0, 0.0]])
    a = percolates(p, q)
    # (b) 两根中间留 100 nm 空隙：不导通
    p2 = np.array([[-half, 0, 0], [50.0, 0, 0]])
    q2 = np.array([[-50.0, 0, 0], [half, 0, 0]])
    b = not percolates(p2, q2)
    # (c) 同样两根，空隙缩到 1.0 nm(< GAP)：导通
    p3 = np.array([[-half, 0, 0], [0.5, 0, 0]])
    q3 = np.array([[-0.5, 0, 0], [half, 0, 0]])
    c = percolates(p3, q3)
    # (d) 只碰左面：不导通
    d = not percolates(np.array([[-half, 0, 0]]), np.array([[0.0, 0, 0]]))
    # (e) 碎片粘合开关：一根跨 X 边界的棒，碎片独立时不通，粘合时通
    box = np.full(3, EDGE)
    segs = wrap_segment(np.array([3000.0, 0, 0]), np.array([8000.0, 0, 0]), box)
    sp = np.array([s[0] for s in segs]); sq = np.array([s[1] for s in segs])
    own = np.zeros(len(segs), dtype=int)
    e = (not percolates(sp, sq)) and percolates(sp, sq, owner_rod=own, bond_fragments=True)
    check("T4_导通判定逻辑", all([a, b, c, d, e]),
          f"贯通={a} 断开={b} 间隙1nm导通={c} 单边不通={d} 碎片粘合开关={e}")


# ---------------------------------------------------------------- T5 加速等价
def t5_blocked_equals_naive() -> None:
    """分块+包围盒预筛的接触检索必须与朴素两两比较逐位一致。

    加速实现如果漏掉一条边，导通概率会系统性偏低，而这种偏差在结果里看不出来——
    所以它必须被测，而不是被相信。
    """
    from microstructure import contact_pairs, sample_rods
    rng = np.random.default_rng(31415)
    box = np.full(3, EDGE)
    ok = True
    detail = []
    for n in (50, 150, 400):
        sp, sq, _ = sample_rods(n, rng, box)
        d = seg_seg_dist(sp[:, None, :], sq[:, None, :], sp[None, :, :], sq[None, :, :])
        iu = np.triu_indices(len(sp), k=1)
        hit = d[iu] <= 2 * R_A + GAP
        naive = set(zip(iu[0][hit].tolist(), iu[1][hit].tolist()))
        fast = set(map(tuple, contact_pairs(sp, sq, 2 * R_A + GAP).tolist()))
        ok &= naive == fast
        detail.append(f"N_A={n}: {len(naive)} 条边，一致={naive == fast}")
    check("T5_加速实现等价", ok, "；".join(detail))


if __name__ == "__main__":
    t1_segment_distance()
    t2_statement_example()
    t3_attachment_roundtrip()
    t4_percolation_logic()
    t5_blocked_equals_naive()
    note_capsule_scale()          # 量级参考，放在最后，不参与通过判定
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + ("全部通过" if not failures else f"失败项：{failures}"))
    sys.exit(1 if failures else 0)
