#!/usr/bin/env python3
"""平端圆柱距离判定的解析算例与 K-/K+ 包含关系回归。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microstructure import (EDGE, H_A, R_A, cylinder_cylinder_dist,
                            cylinder_x_extent, percolates, point_cylinder_dist,
                            sample_rods, seg_seg_dist)


def segment(center: np.ndarray, direction: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    u = direction / np.linalg.norm(direction)
    return center - 0.5 * length * u, center + 0.5 * length * u


def shortened(p: np.ndarray, q: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    u = (q - p) / np.linalg.norm(q - p)
    return p + radius * u, q - radius * u


def main() -> int:
    r = 1.0
    p1, q1 = segment(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 10.0)

    cases = [
        (segment(np.array([12.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 10.0), 2.0,
         "同轴端面—端面"),
        (segment(np.array([0.0, 3.0, 0.0]), np.array([1.0, 0.0, 0.0]), 10.0), 1.0,
         "平行侧壁—侧壁"),
        (segment(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 10.0), 0.0,
         "正交相交"),
        (segment(np.array([7.0, 2.0, 0.0]), np.array([0.0, 1.0, 0.0]), 2.0), 1.0,
         "端缘—侧壁"),
    ]
    for (p2, q2), expected, name in cases:
        got = cylinder_cylinder_dist(p1, q1, p2, q2, radius=r)
        if abs(got - expected) > 2e-6:
            raise AssertionError(f"{name}: got {got}, expected {expected}")
        print(f"{name}: {got:.8f}（解析值 {expected:.8f}）")

    pts = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [6.0, 0.0, 0.0],
                    [6.0, 2.0, 0.0]])
    got = point_cylinder_dist(pts, p1, q1, radius=r)
    expected = np.array([0.0, 1.0, 1.0, np.sqrt(2.0)])
    if not np.allclose(got, expected, atol=1e-10):
        raise AssertionError((got, expected))

    lo, hi = cylinder_x_extent(*segment(np.zeros(3), np.array([1.0, 1.0, 0.0]), 10.0),
                               radius=r)
    expected_half = 5.0 / np.sqrt(2.0) + 1.0 / np.sqrt(2.0)
    if not np.allclose([lo, hi], [-expected_half, expected_half], atol=1e-10):
        raise AssertionError((lo, hi, expected_half))

    # 随机验证 K^- ⊆ C ⊆ K^+ 对接触判定的传导方向。
    rng = np.random.default_rng(20260812)
    violations_inner = 0
    violations_outer = 0
    for _ in range(1000):
        u1, u2 = rng.normal(size=(2, 3))
        c1 = rng.uniform(-4.0, 4.0, 3)
        c2 = rng.uniform(-4.0, 4.0, 3)
        a1, b1 = segment(c1, u1, 10.0)
        a2, b2 = segment(c2, u2, 10.0)
        ia1, ib1 = shortened(a1, b1, r)
        ia2, ib2 = shortened(a2, b2, r)
        d_inner_axis = float(seg_seg_dist(ia1, ib1, ia2, ib2))
        d_outer_axis = float(seg_seg_dist(a1, b1, a2, b2))
        d_real = cylinder_cylinder_dist(a1, b1, a2, b2, radius=r)
        gap = 0.2
        if d_inner_axis <= 2 * r + gap and d_real > gap + 2e-6:
            violations_inner += 1
        if d_real <= gap and d_outer_axis > 2 * r + gap + 2e-6:
            violations_outer += 1
    if violations_inner or violations_outer:
        raise AssertionError(f"包含关系违例 inner={violations_inner}, outer={violations_outer}")
    print("随机包含关系：1000/1000 通过（K^- 接触 ⇒ C 接触 ⇒ K^+ 接触）")

    # 在完整接触图层面再次核对传导方向；三种几何使用完全相同的中心与方向随机数。
    box = np.full(3, EDGE)
    for trial in range(12):
        seed = 20260900 + trial
        rng_outer = np.random.default_rng(seed)
        rng_real = np.random.default_rng(seed)
        rng_inner = np.random.default_rng(seed)
        op, oq, oo = sample_rods(240, rng_outer, box, length=H_A)
        rp, rq, ro = sample_rods(240, rng_real, box, length=H_A)
        ip, iq, io = sample_rods(240, rng_inner, box, length=H_A - 2 * R_A)
        outer = percolates(op, oq, owner_rod=oo, rod_geometry="capsule")
        real = percolates(rp, rq, owner_rod=ro, rod_geometry="cylinder")
        inner = percolates(ip, iq, owner_rod=io, rod_geometry="capsule")
        if inner and not real or real and not outer:
            raise AssertionError(f"trial={trial}: inner={inner}, real={real}, outer={outer}")
    print("贯通事件包含关系：12/12 通过（同随机构型逐次核对）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
