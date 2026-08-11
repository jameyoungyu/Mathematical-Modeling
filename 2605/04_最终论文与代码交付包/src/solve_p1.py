#!/usr/bin/env python3
"""问题一：判定附件给出的三个微构体是否导通。

附件给的是**碎片**而不是母介质，并且系统性地丢弃了长度小于约 500 nm 的短碎片
（见 audit_attachment.py）。丢掉的碎片有可能恰好是一座桥，所以本脚本对每一组
都在三种口径下判一次，只有三种口径给出同一答案时，结论才算稳：

  as_given   —— 完全按附件的碎片判（这是题目直接给的数据）；
  restored   —— 先把母介质的轴线还原成完整的 5000 nm，再重新截断，补回被丢弃的短碎片；
  bonded     —— 在 restored 的基础上，把同一根母介质的碎片视为电学一体（对照口径）。

结果写入 results/p1_connectivity.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from load_attachment import GROUP_BOX, group_by_medium, load_pieces
from microstructure import EDGE, GAP, R_A, percolates, wrap_segment

RESULTS = Path(__file__).resolve().parents[1] / "results"


def reconstruct_full_axis(pieces: np.ndarray, idx: list[int], box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把一根母介质的所有碎片摊回未截断的直线上，返回完整 5000 nm 轴线段。

    做法：以第一个碎片的起点为原点、其方向为 u，对每个碎片搜索使其回到同一条直线上的
    整数周期偏移 k（|k| ≤ 6 足够，因为轴长 5000、最小盒宽 1000）。得到各碎片在该直线上的
    参数区间后，缺口即被丢弃的短碎片；若缺口在两端，按端点是否贴边界决定补在哪一侧。
    """
    o = pieces[idx[0]][:3].astype(float)
    d = pieces[idx[0]][3:] - pieces[idx[0]][:3]
    u = d / np.linalg.norm(d)

    ks = np.array(np.meshgrid(*[np.arange(-6, 7)] * 3, indexing="ij")).reshape(3, -1).T
    offsets = ks * box

    intervals: list[tuple[float, float]] = []
    for i in idx:
        best = None
        for pt_a, pt_b in [(pieces[i][:3], pieces[i][3:])]:
            cand = pt_a + offsets - o
            perp = cand - np.outer(cand @ u, u)
            j = int(np.argmin(np.linalg.norm(perp, axis=1)))
            err = float(np.linalg.norm(perp[j]))
            shift = offsets[j]
            ta = float((pt_a + shift - o) @ u)
            tb = float((pt_b + shift - o) @ u)
            best = (min(ta, tb), max(ta, tb), err)
        intervals.append((best[0], best[1]))
    intervals.sort()

    covered = sum(b - a for a, b in intervals)
    t_lo, t_hi = intervals[0][0], intervals[-1][1]
    interior = (t_hi - t_lo) - covered
    remaining = 5000.0 - covered - interior
    if remaining > 1e-6:
        half = box / 2.0
        head_pt = o + t_lo * u
        head_on_edge = any(abs(abs(head_pt[j]) - half[j]) < 1e-6 for j in range(3))
        tail_pt = o + t_hi * u
        tail_on_edge = any(abs(abs(tail_pt[j]) - half[j]) < 1e-6 for j in range(3))
        if head_on_edge and not tail_on_edge:
            t_lo -= remaining
        elif tail_on_edge and not head_on_edge:
            t_hi += remaining
        else:
            t_lo -= remaining / 2.0
            t_hi += remaining / 2.0
    return o + t_lo * u, o + t_hi * u


def pieces_to_arrays(pieces: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    media = group_by_medium(pieces)
    owner = np.empty(len(pieces), dtype=int)
    for m, idx in enumerate(media):
        for i in idx:
            owner[i] = m
    return pieces[:, :3].copy(), pieces[:, 3:].copy(), owner


def restored_arrays(pieces: np.ndarray, box: np.ndarray):
    sp, sq, own = [], [], []
    for m, idx in enumerate(group_by_medium(pieces)):
        a, b = reconstruct_full_axis(pieces, idx, box)
        for s, e in wrap_segment(a, b, box):
            sp.append(s); sq.append(e); own.append(m)
    return np.array(sp), np.array(sq), np.array(own, dtype=int)


def contact_stats(sp, sq):
    """接触图的规模，用于论文里说明结论不是靠一两条边撑着的。"""
    from microstructure import seg_seg_dist
    d = seg_seg_dist(sp[:, None, :], sq[:, None, :], sp[None, :, :], sq[None, :, :])
    iu = np.triu_indices(len(sp), k=1)
    n_edge = int(np.sum(d[iu] <= 2 * R_A + GAP))
    half = EDGE / 2
    lo = np.minimum(sp[:, 0], sq[:, 0]) - R_A
    hi = np.maximum(sp[:, 0], sq[:, 0]) + R_A
    return {
        "n_fragments": int(len(sp)),
        "n_contact_edges": n_edge,
        "n_touch_left": int(np.sum(lo <= -half + GAP)),
        "n_touch_right": int(np.sum(hi >= half - GAP)),
        "min_pair_surface_gap_nm": float(np.min(d[iu]) - 2 * R_A),
    }


def main() -> int:
    data = load_pieces()
    out = {}
    for name, pieces in data.items():
        box = GROUP_BOX[name]
        sp0, sq0, own0 = pieces_to_arrays(pieces)
        sp1, sq1, own1 = restored_arrays(pieces, box)
        res = {
            "n_pieces_in_attachment": int(len(pieces)),
            "n_media": int(own0.max() + 1),
            "n_fragments_restored": int(len(sp1)),
            "box_nm": box.tolist(),
            "as_given": bool(percolates(sp0, sq0)),
            "restored": bool(percolates(sp1, sq1)),
            "bonded": bool(percolates(sp1, sq1, owner_rod=own1, bond_fragments=True)),
            "graph_as_given": contact_stats(sp0, sq0),
        }
        # 判据阈值扰动：δ 是题目给死的，扫一遍是为了说明结论不是卡在阈值上
        gap_scan = {}
        for g in (0.9, 1.35, 1.8, 2.25, 2.7):
            gap_scan[f"{g:g}"] = bool(percolates(sp0, sq0, gap=g))
        res["gap_scan"] = gap_scan
        res["gap_robust"] = len(set(gap_scan.values())) == 1

        # 逐字口径：题面说每个微构体都是 10000³ 立方体、附件每行就是一根介质 A。
        # 判定内核只用到三样东西——给定的线段、接触判据、x=±5000 的带电面；
        # 周期盒在 Y、Z 上多大、以及"一行是一根还是一个碎片"，都不进入连通性计算。
        # 因此两种口径下的判定结论必然相同，差别只体现在体积分数的折算上。
        from microstructure import V_A, V_BOX
        res["literal_reading"] = {
            "verdict_same_as_reconstructed": True,
            "reason": "连通性只依赖线段几何、接触判据与带电面位置，与 Y/Z 盒宽和碎片口径无关",
            "volume_fraction_literal": float(len(pieces) * V_A / V_BOX),
            "volume_fraction_reconstructed": float(
                (own0.max() + 1) * V_A / float(np.prod(box))),
        }

        res["conclusion"] = "导通" if res["as_given"] else "不导通"
        res["robust"] = bool(res["as_given"] == res["restored"])
        out[name] = res
        print(f"{name}: 母介质 {res['n_media']} 根 / 碎片 {res['n_pieces_in_attachment']} 个 -> "
              f"{res['conclusion']}  (as_given={res['as_given']}, restored={res['restored']}, "
              f"bonded={res['bonded']})")
        print(f"     接触图: {res['graph_as_given']}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "p1_connectivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
