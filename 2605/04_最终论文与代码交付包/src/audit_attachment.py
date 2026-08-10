#!/usr/bin/env python3
"""附件数据审计。

在对附件做任何建模之前先回答四个问题——每一个的答案都改变了后面的模型：

A1 附件的一行是什么？  不是一根介质 A，而是边界截断之后的一个碎片。
A2 三个微构体的周期盒一样大吗？  组 3 是题面的 10000³；组 1、组 2 的 y、z 在 ±500 处
   回绕，截面只有 1000×1000 nm。
A3 附件完整吗？  不完整：长度小于约 500 nm 的短碎片被系统性丢弃。
A4 介质方向是各向同性的吗？  不是。组 3 的方向服从"以 X 轴为极轴、θ~U(0,π)"的分布，
   明显偏向带电面法向；各向同性被 KS 检验以 p≈7e-19 拒绝。

A4 是全题影响最大的一条：它把 φ=0.50% 的导通概率抬高了约一倍。

结果写入 results/data_audit.json。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from load_attachment import (GROUP_BOX, group_by_medium, load_pieces,
                             reconstruct_axis)

RESULTS = Path(__file__).resolve().parents[1] / "results"


def piece_lengths(pieces: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pieces[:, 3:] - pieces[:, :3], axis=1)


def wrap_evidence(pieces: np.ndarray, box: np.ndarray) -> dict:
    """统计碎片端点精确落在各周期面上的次数。

    这是判断周期盒尺寸的直接证据：碎片在边界处成对出现（前一碎片终点在 +h、
    后一碎片起点在 −h），因此只要数一数端点恰好取到 ±h 的次数，就能读出回绕面在哪。
    """
    half = box / 2.0
    out = {}
    for j, axis in enumerate("XYZ"):
        n = 0
        for row in pieces:
            for pt in (row[:3], row[3:]):
                if abs(abs(pt[j]) - half[j]) < 1e-6:
                    n += 1
        out[axis] = {"half_extent_nm": float(half[j]), "endpoints_on_face": int(n)}
    return out


def audit_group(name: str, pieces: np.ndarray) -> dict:
    box = GROUP_BOX[name]
    media = group_by_medium(pieces)
    lens = piece_lengths(pieces)

    # A3：把每根母介质的碎片接起来，缺多少长度就是被丢弃的碎片
    missing = []
    for idx in media:
        _, _, total = reconstruct_axis(pieces, idx, box)
        if abs(total - 5000.0) > 1e-3:
            missing.append(5000.0 - total)

    # A4：母介质方向的各向异性
    dirs = []
    for idx in media:
        d = pieces[idx[0]][3:] - pieces[idx[0]][:3]
        dirs.append(np.abs(d / np.linalg.norm(d)))
    dirs = np.array(dirs)

    iso = stats.kstest(dirs[:, 0], "uniform")
    polar_cdf = lambda t: np.clip((np.pi - 2 * np.arccos(np.clip(t, 0, 1))) / np.pi, 0, 1)
    pol = stats.kstest(dirs[:, 0], polar_cdf)

    # 只检验 |u_x| 的边缘分布不足以确定整个方向分布：还要看方位角是否均匀、
    # 以及方位角与极角是否独立。方向只到反号意义下确定（本文取 u_x≥0 的一支），
    # 反号会把方位角平移 π，而均匀分布对平移不变，因此下面的检验仍然成立。
    signed = []
    for idx in media:
        d = pieces[idx[0]][3:] - pieces[idx[0]][:3]
        d = d / np.linalg.norm(d)
        signed.append(-d if d[0] < 0 else d)
    signed = np.array(signed)
    azim = np.mod(np.arctan2(signed[:, 2], signed[:, 1]), 2 * np.pi)
    ks_az = stats.kstest(azim / (2 * np.pi), "uniform")
    rho, p_rho = stats.spearmanr(dirs[:, 0], azim)

    return {
        "n_pieces": int(len(pieces)),
        "n_media": int(len(media)),
        "pieces_per_medium": float(len(pieces) / len(media)),
        "periodic_box_nm": box.tolist(),
        "wrap_evidence": wrap_evidence(pieces, box),
        "piece_length_min_nm": float(lens.min()),
        "piece_length_max_nm": float(lens.max()),
        "total_axis_length_nm": float(lens.sum()),
        "volume_fraction": float(len(media) * np.pi * 30.0 ** 2 * 5000.0 / np.prod(box)),
        "n_media_incomplete": len(missing),
        "dropped_length_nm": float(sum(missing)),
        "dropped_fragment_len_max_nm": float(max(missing)) if missing else 0.0,
        # 每根不完整母介质缺的总长；缺 2 个碎片的母介质会给出 >500 nm 的值，
        # 单个被丢弃的碎片本身都短于 500 nm（保留的最短碎片 526 nm）。
        "dropped_per_medium_nm": sorted(round(float(m), 1) for m in missing),
        "n_dropped_under_500": int(sum(1 for m in missing if m < 500)),
        "mean_abs_u": dirs.mean(axis=0).tolist(),
        "ks_isotropic": {"D": float(iso.statistic), "p": float(iso.pvalue)},
        "ks_polar_uniform": {"D": float(pol.statistic), "p": float(pol.pvalue)},
        "ks_azimuth_uniform": {"D": float(ks_az.statistic), "p": float(ks_az.pvalue)},
        "spearman_absux_vs_azimuth": {"rho": float(rho), "p": float(p_rho)},
    }


def main() -> int:
    data = load_pieces()
    out = {
        "reference_means": {
            "isotropic": [0.5, 0.5, 0.5],
            "polar_uniform_about_x": [2 / np.pi, (2 / np.pi) ** 2, (2 / np.pi) ** 2],
        },
        "groups": {name: audit_group(name, p) for name, p in data.items()},
    }
    for name, g in out["groups"].items():
        print(f"{name}: 碎片 {g['n_pieces']} -> 母介质 {g['n_media']} 根 "
              f"({g['pieces_per_medium']:.3f} 碎片/根), 体积分数 {g['volume_fraction']:.4%}")
        print(f"   周期盒 {g['periodic_box_nm']}; 最短碎片 {g['piece_length_min_nm']:.0f} nm; "
              f"丢弃 {g['dropped_length_nm']:.0f} nm（{g['n_media_incomplete']} 根不完整，"
              f"最长丢弃碎片 {g['dropped_fragment_len_max_nm']:.0f} nm）")
        print(f"   E|u| = ({g['mean_abs_u'][0]:.3f},{g['mean_abs_u'][1]:.3f},{g['mean_abs_u'][2]:.3f}); "
              f"KS 各向同性 p={g['ks_isotropic']['p']:.2e}, "
              f"KS 极角均匀 p={g['ks_polar_uniform']['p']:.3f}")
        print(f"   方位角均匀 KS p={g['ks_azimuth_uniform']['p']:.3f}; "
              f"|u_x| 与方位角 Spearman ρ={g['spearman_absux_vs_azimuth']['rho']:+.3f} "
              f"(p={g['spearman_absux_vs_azimuth']['p']:.3f})")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "data_audit.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
