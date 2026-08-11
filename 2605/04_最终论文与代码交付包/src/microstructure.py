#!/usr/bin/env python3
"""微构体导电仿真的几何与渗流内核。

本模块只做三件事，不涉及任何题目分支逻辑：

1. **边界截断**：把一根越界的介质按题目给定的规则平移回微构体，得到若干碎片；
2. **接触判定**：两个介质（或介质与带电面）之间的最短表面距离是否不超过 1.8 nm；
3. **导通判定**：并查集把接触关系连成连通块，判断是否存在连接左右带电面的通路。

几何约定
--------
介质 A 建模为**球柱体（spherocylinder / capsule）**：半径 r=30 nm 的圆柱两端各加一个
半球帽。这样两个介质 A 的表面最短距离就精确等于两条轴线段的最短距离减去 2r，
判定可以完全向量化。题目里的介质 A 是平端面直圆柱，与球柱体的差别只出现在
端面附近，且体积仅相差 (4/3)πr³ / (πr²h) = 4r/(3h) = 0.8%。
`validate.py` 里量化了这一近似对导通判定的影响。

介质 B 是半径 200 nm 的正球体，本身就是球，无需近似。
"""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ 物理常数

EDGE = 10000.0          # 微构体边长 (nm)
R_A = 30.0              # 介质 A 底面半径 (nm)
H_A = 5000.0            # 介质 A 高 (nm)
R_B = 200.0             # 介质 B 半径 (nm)
GAP = 1.8               # 导通判定的最大表面间距 (nm)

V_BOX = EDGE ** 3                       # 1.0e12 nm^3
V_A = np.pi * R_A ** 2 * H_A            # 1.41372e7 nm^3
V_B = 4.0 / 3.0 * np.pi * R_B ** 3      # 3.35103e7 nm^3

# 成本：介质 A 1.05 元/μm³，介质 B 0.05 元/μm³；1 μm³ = 1e9 nm³
COST_A = 1.05 * V_A / 1e9               # 元 / 根
COST_B = 0.05 * V_B / 1e9               # 元 / 颗


# ------------------------------------------------------------------ 边界截断

def wrap_segment(p: np.ndarray, q: np.ndarray, box: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """把轴线段 p->q 按边界截断规则折回盒子，返回若干条完全位于盒内的子段。

    做法：把线段参数化为 t∈[0,1]，在每个坐标轴上求出它穿越周期格点边界的所有 t，
    以这些 t 为断点切段；每一小段整体位于同一个周期像里，平移回来即可。
    该过程对轴线与题面平移规则等价，并且天然支持多次平移。柱体径向越界的薄层
    （最厚 R_A）未在这里做实体级精确切分，相关几何误差另由双侧界量化。
    """
    half = box / 2.0
    d = q - p
    ts = [0.0, 1.0]
    for j in range(3):
        if abs(d[j]) < 1e-12:
            continue
        # 穿越平面 x_j = half_j + k*box_j
        k_lo = np.floor((min(p[j], q[j]) - half[j]) / box[j])
        k_hi = np.ceil((max(p[j], q[j]) - half[j]) / box[j])
        for k in range(int(k_lo), int(k_hi) + 1):
            plane = half[j] + k * box[j]
            t = (plane - p[j]) / d[j]
            if 1e-12 < t < 1 - 1e-12:
                ts.append(float(t))
    ts = sorted(set(ts))

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for t0, t1 in zip(ts[:-1], ts[1:]):
        if t1 - t0 < 1e-12:
            continue
        a = p + t0 * d
        b = p + t1 * d
        mid = 0.5 * (a + b)
        shift = np.round(mid / box) * box          # 该子段所在周期像的偏移
        seg_a, seg_b = a - shift, b - shift
        # 数值上把端点压回盒内，避免 ±1e-12 的溢出
        np.clip(seg_a, -half, half, out=seg_a)
        np.clip(seg_b, -half, half, out=seg_b)
        out.append((seg_a, seg_b))
    return out


# ------------------------------------------------------------------ 距离计算

def seg_seg_dist(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """成对线段最短距离，全向量化。

    p1/q1 形状 (n,1,3)，p2/q2 形状 (1,m,3) 时返回 (n,m)。
    算法是 Ericson《Real-Time Collision Detection》的 ClosestPtSegmentSegment，
    分支用 np.where 展开；退化（零长）线段按点处理。
    """
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2

    a = np.sum(d1 * d1, axis=-1)
    e = np.sum(d2 * d2, axis=-1)
    f = np.sum(d2 * r, axis=-1)
    c = np.sum(d1 * r, axis=-1)
    b = np.sum(d1 * d2, axis=-1)

    eps = 1e-12
    a_safe = np.maximum(a, eps)
    e_safe = np.maximum(e, eps)

    denom = a * e - b * b
    s = np.where(denom > eps, (b * f - c * e) / np.where(denom > eps, denom, 1.0), 0.0)
    s = np.clip(s, 0.0, 1.0)

    t = (b * s + f) / e_safe

    # t 越界时钳位并回代求 s
    s_lo = np.clip(-c / a_safe, 0.0, 1.0)
    s_hi = np.clip((b - c) / a_safe, 0.0, 1.0)
    s = np.where(t < 0.0, s_lo, np.where(t > 1.0, s_hi, s))
    t = np.clip(t, 0.0, 1.0)

    # 退化线段
    s = np.where(a <= eps, 0.0, s)
    t = np.where(e <= eps, 0.0, np.where(a <= eps, np.clip(f / e_safe, 0.0, 1.0), t))

    diff = r + s[..., None] * d1 - t[..., None] * d2
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))


def contact_pairs(p: np.ndarray, q: np.ndarray, thresh: float, block: int = 256) -> np.ndarray:
    """返回所有轴线距离 ≤ thresh 的线段对下标 (k,2)。

    分块 + 轴对齐包围盒预筛。朴素写法要一次性开出 M² 的十几个临时数组，
    M≈1300 时光是分配和读写就占了九成时间；分块后临时数组常驻缓存，
    实测在 φ=1% 规模上快 6 倍以上，结果与朴素写法逐位相同（validate.py T6）。
    """
    m = len(p)
    lo = np.minimum(p, q) - thresh
    hi = np.maximum(p, q)
    out: list[np.ndarray] = []
    for s in range(0, m, block):
        e = min(s + block, m)
        # 只与下标更大的比较，避免重复
        cand = np.nonzero(np.all(lo[None, s:e, :] <= hi[s:, None, :], axis=-1)
                          & np.all(lo[s:, None, :] <= hi[None, s:e, :], axis=-1))
        rows = cand[0] + s          # 全局下标 j
        cols = cand[1] + s          # 全局下标 i
        keep = rows > cols
        rows, cols = rows[keep], cols[keep]
        if rows.size == 0:
            continue
        d = seg_seg_dist(p[cols], q[cols], p[rows], q[rows])
        hit = d <= thresh
        if np.any(hit):
            out.append(np.stack([cols[hit], rows[hit]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2), dtype=int)


def point_seg_dist(pts: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """点到线段的距离。pts (n,1,3)，p/q (1,m,3) -> (n,m)。"""
    d = q - p
    w = pts - p
    dd = np.maximum(np.sum(d * d, axis=-1), 1e-12)
    t = np.clip(np.sum(w * d, axis=-1) / dd, 0.0, 1.0)
    diff = w - t[..., None] * d
    return np.sqrt(np.maximum(np.sum(diff * diff, axis=-1), 0.0))


# ------------------------------------------------------------------ 并查集

class DSU:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        p = self.parent
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[ry] = rx


# ------------------------------------------------------------------ 生成配置

# 全文的方向分布口径集中在这两个常量上，改口径只需改这里。
#
# 主口径取**球面均匀**：题面对问题二至四只说"方向随机"，未指定分布，
# 球面均匀是这句话的最少假设读法。
# 对照口径取附件标定的极角均匀分布：附件组 3 的 354 根母介质以
# KS 检验 p=6.7e-19 拒绝球面均匀、p=0.73 不拒绝极角均匀（第 2.4 节）。
# 这一条只作为灵敏度分析出现——它刻画的是附件的生成过程，
# 而题面并未说问题二至四必须沿用同一过程。
PRIMARY_ORIENTATION = "isotropic"
CONTROL_ORIENTATION = "polar_uniform"


def sample_orientations(n: int, rng: np.random.Generator,
                        mode: str = PRIMARY_ORIENTATION) -> np.ndarray:
    """生成 n 个单位方向向量。

    mode="isotropic"     ：球面均匀，即 |u_x| ~ U(0,1)。这是"任意方向、不考虑重力"
                           最自然的物理读法。
    mode="polar_uniform" ：θ ~ U(0,π)、φ ~ U(0,2π)，以 **X 轴（带电面法向）** 为极轴，
                           u = (cosθ, sinθ cosφ, sinθ sinφ)。这是附件数据实际服从的分布：
                           对组 3 的 354 根母介质做 KS 检验，各向同性被拒绝（D=0.243,
                           p=6.7e-19），本分布不被拒绝（D=0.036, p=0.73）。
                           它在两极堆积，使介质明显偏向带电面法向，从而显著提高导通概率。
    """
    if mode == "isotropic":
        u = rng.normal(size=(n, 3))
        return u / np.linalg.norm(u, axis=1, keepdims=True)
    if mode == "polar_uniform":
        theta = rng.uniform(0.0, np.pi, n)
        phi = rng.uniform(0.0, 2 * np.pi, n)
        s = np.sin(theta)
        return np.stack([np.cos(theta), s * np.cos(phi), s * np.sin(phi)], axis=1)
    raise ValueError(f"未知方向分布：{mode}")


def sample_rods(n: int, rng: np.random.Generator, box: np.ndarray,
                length: float = H_A,
                orientation: str = PRIMARY_ORIENTATION) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """随机生成 n 根介质 A 并做边界截断。

    中心在盒内均匀；方向分布见 ``sample_orientations``。
    返回碎片起点、终点，以及每个碎片所属的母介质编号。
    """
    centers = (rng.random((n, 3)) - 0.5) * box
    u = sample_orientations(n, rng, orientation)
    p = centers - 0.5 * length * u
    q = centers + 0.5 * length * u

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    owner: list[int] = []
    for i in range(n):
        for a, b in wrap_segment(p[i], q[i], box):
            starts.append(a)
            ends.append(b)
            owner.append(i)
    if not starts:
        empty = np.zeros((0, 3))
        return empty, empty, np.zeros(0, dtype=int)
    return np.array(starts), np.array(ends), np.array(owner, dtype=int)


def sample_spheres(n: int, rng: np.random.Generator, box: np.ndarray,
                   mode: str = "wrap") -> tuple[np.ndarray, np.ndarray]:
    """随机生成 n 颗介质 B。

    mode="wrap"   ：中心在盒内均匀，越界部分按截断规则折回，碎片以其所在周期像的
                    球心表示（碎片是球与盒的交，用整球近似，只在距壁 200 nm 内有偏差）；
    mode="inside" ：把球心限制在 [-(L/2-R), L/2-R]，保证整颗球都在盒内，不产生碎片。
    两种取法在灵敏度分析里对比。
    """
    if mode == "inside":
        c = (rng.random((n, 3)) - 0.5) * (box - 2 * R_B)
        return c, np.arange(n)

    c = (rng.random((n, 3)) - 0.5) * box
    half = box / 2.0
    centers: list[np.ndarray] = []
    owner: list[int] = []
    for i in range(n):
        # 每个越界方向都产生一个镜像碎片
        offs = [np.zeros(3)]
        for j in range(3):
            if c[i, j] + R_B > half[j]:
                offs = offs + [o + np.eye(3)[j] * -box[j] for o in offs]
            elif c[i, j] - R_B < -half[j]:
                offs = offs + [o + np.eye(3)[j] * box[j] for o in offs]
        for o in offs:
            centers.append(c[i] + o)
            owner.append(i)
    return np.array(centers), np.array(owner, dtype=int)


# ------------------------------------------------------------------ 导通判定

def percolates(seg_p: np.ndarray, seg_q: np.ndarray,
               sph_c: np.ndarray | None = None,
               owner_rod: np.ndarray | None = None,
               owner_sph: np.ndarray | None = None,
               bond_fragments: bool = False,
               edge: float = EDGE) -> bool:
    """判断微构体是否导通（左右带电面之间存在导电通路）。

    bond_fragments=False（默认）：边界截断产生的每个碎片是独立导体；
    bond_fragments=True         ：同一根母介质的所有碎片电学上视为一体（灵敏度对照）。
    """
    half = edge / 2.0
    n_rod = len(seg_p)
    n_sph = 0 if sph_c is None else len(sph_c)
    n = n_rod + n_sph
    if n == 0:
        return False

    dsu = DSU(n + 2)
    LEFT, RIGHT = n, n + 1

    # --- 介质与带电面
    if n_rod:
        lo = np.minimum(seg_p[:, 0], seg_q[:, 0]) - R_A
        hi = np.maximum(seg_p[:, 0], seg_q[:, 0]) + R_A
        for i in np.nonzero(lo <= -half + GAP)[0]:
            dsu.union(LEFT, int(i))
        for i in np.nonzero(hi >= half - GAP)[0]:
            dsu.union(RIGHT, int(i))
    if n_sph:
        for i in np.nonzero(sph_c[:, 0] - R_B <= -half + GAP)[0]:
            dsu.union(LEFT, n_rod + int(i))
        for i in np.nonzero(sph_c[:, 0] + R_B >= half - GAP)[0]:
            dsu.union(RIGHT, n_rod + int(i))

    # --- 棒-棒
    if n_rod > 1:
        for a, b in contact_pairs(seg_p, seg_q, 2 * R_A + GAP):
            dsu.union(int(a), int(b))

    # --- 球-球 与 球-棒（介质 B 数量可达上万，用 KD 树做邻域检索）
    if n_sph:
        from scipy.spatial import cKDTree
        tree = cKDTree(sph_c)
        if n_sph > 1:
            for a, b in tree.query_pairs(2 * R_B + GAP, output_type="ndarray"):
                dsu.union(n_rod + int(a), n_rod + int(b))
        if n_rod:
            # 沿每根碎片轴线按 STEP 采样，查半径 = 接触半径 + 半个采样步长，
            # 保证不漏掉任何真实接触；再用精确的点—线段距离过滤候选。
            reach = R_A + R_B + GAP
            step = 200.0
            samples, tags = [], []
            for i in range(n_rod):
                a, b = seg_p[i], seg_q[i]
                L = float(np.linalg.norm(b - a))
                k = max(2, int(np.ceil(L / step)) + 1)
                ts = np.linspace(0.0, 1.0, k)[:, None]
                samples.append(a + ts * (b - a))
                tags.append(np.full(k, i, dtype=int))
            pts = np.concatenate(samples)
            tags = np.concatenate(tags)
            radius = reach + step / 2.0 + 1e-9
            for pt_idx, sph_list in enumerate(tree.query_ball_point(pts, radius)):
                if not sph_list:
                    continue
                rod = int(tags[pt_idx])
                cand = np.asarray(sph_list, dtype=int)
                d = point_seg_dist(sph_c[cand][:, None, :],
                                   seg_p[rod][None, None, :], seg_q[rod][None, None, :])[:, 0]
                for s in cand[d <= reach]:
                    dsu.union(n_rod + int(s), rod)

    # --- 灵敏度对照：同母介质的碎片粘合
    if bond_fragments:
        for owner, base, cnt in ((owner_rod, 0, n_rod), (owner_sph, n_rod, n_sph)):
            if owner is None or cnt == 0:
                continue
            first: dict[int, int] = {}
            for idx in range(cnt):
                o = int(owner[idx])
                if o in first:
                    dsu.union(first[o], base + idx)
                else:
                    first[o] = base + idx

    return dsu.find(LEFT) == dsu.find(RIGHT)


def n_rods_for_fraction(phi: float) -> int:
    """体积分数 -> 介质 A 根数（题目要求四舍五入）。"""
    return int(round(phi * V_BOX / V_A))
