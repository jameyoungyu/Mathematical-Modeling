#!/usr/bin/env python3
"""微构体导电仿真的几何与渗流内核。

本模块只做三件事，不涉及任何题目分支逻辑：

1. **边界截断**：把一根越界的介质按题目给定的规则平移回微构体，得到若干碎片；
2. **接触判定**：两个介质（或介质与带电面）之间的最短表面距离是否不超过 1.8 nm；
3. **导通判定**：并查集把接触关系连成连通块，判断是否存在连接左右带电面的通路。

几何约定
--------
介质 A 支持两种口径。``rod_geometry="capsule"`` 保留既有球柱体结果，轴线距离判定可完全
向量化；``rod_geometry="cylinder"`` 按题面的平端面直圆柱计算，圆柱—圆柱采用支撑映射
Gilbert/GJK 距离，圆柱—球与圆柱—带电面使用解析距离。平端模式先用内外球柱体筛掉绝大多数
显然接触/不接触的介质对，只在端缘临界窄带调用 GJK，因此可用于问题四的真实几何终核。

介质 B 是半径 200 nm 的正球体，本身就是球，无需近似。
"""

from __future__ import annotations

from itertools import combinations

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


def point_cylinder_dist(pts: np.ndarray, p: np.ndarray, q: np.ndarray,
                        radius: float = R_A) -> np.ndarray:
    """点到平端实心圆柱的欧氏距离。

    ``pts`` 为 ``(..., 3)``；``p,q`` 可广播到相同前缀形状。圆柱轴线是 ``p→q``，
    半径为 ``radius``。点在圆柱内部时返回 0。该公式同时覆盖侧壁、端面和端缘三类
    最近点，比点—轴线段距离减半径的球柱体公式少了端帽处的系统性高估。
    """
    axis = q - p
    length = np.linalg.norm(axis, axis=-1)
    safe = np.maximum(length, 1e-12)
    u = axis / safe[..., None]
    c = 0.5 * (p + q)
    w = pts - c
    axial = np.abs(np.sum(w * u, axis=-1)) - 0.5 * length
    radial_vec = w - np.sum(w * u, axis=-1)[..., None] * u
    radial = np.linalg.norm(radial_vec, axis=-1) - radius
    return np.hypot(np.maximum(axial, 0.0), np.maximum(radial, 0.0))


def cylinder_support(p: np.ndarray, q: np.ndarray, direction: np.ndarray,
                     radius: float = R_A) -> np.ndarray:
    """平端实心圆柱在 ``direction`` 方向的支撑点。"""
    axis = q - p
    length = float(np.linalg.norm(axis))
    if length <= 1e-12:
        # 极短碎片退化为球，避免方向未定义；正常数据不会走到这里。
        n = float(np.linalg.norm(direction))
        return 0.5 * (p + q) if n <= 1e-15 else 0.5 * (p + q) + radius * direction / n
    u = axis / length
    c = 0.5 * (p + q)
    du = float(np.dot(direction, u))
    perp = direction - du * u
    pn = float(np.linalg.norm(perp))
    radial = np.zeros(3) if pn <= 1e-15 else radius * perp / pn
    return c + (0.5 * length if du >= 0.0 else -0.5 * length) * u + radial


def _closest_origin_on_hull(points: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """求有限点集凸包中离原点最近的点，并只保留其活动单纯形。

    三维 Carathéodory 定理保证最多只需 4 个点。枚举至多 4 点的子集后解带
    ``sum(lambda)=1`` 的最小二乘；负权重的候选丢弃。点数很小（GJK 通常 2–4 个），
    这种写法比手写四面体分支更容易核验，也避开退化共面情形。
    """
    best_v: np.ndarray | None = None
    best_pts: list[np.ndarray] | None = None
    best_norm2 = float("inf")
    n = len(points)
    for size in range(1, min(4, n) + 1):
        for idx in combinations(range(n), size):
            sub = np.asarray([points[i] for i in idx], dtype=float)
            if size == 1:
                lam = np.ones(1)
            else:
                gram = sub @ sub.T
                one = np.ones(size)
                # 直接解 KKT 系统；增广约束在点集跨过原点、G 奇异时仍能给出零距离解。
                kkt = np.block([[2.0 * gram, one[:, None]],
                                [one[None, :], np.zeros((1, 1))]])
                rhs = np.r_[np.zeros(size), 1.0]
                lam = np.linalg.lstsq(kkt, rhs, rcond=1e-13)[0][:size]
            if np.min(lam) < -1e-10:
                continue
            lam = np.maximum(lam, 0.0)
            lam /= np.sum(lam)
            v = lam @ sub
            norm2 = float(v @ v)
            if norm2 < best_norm2:
                active = [points[i] for i, w in zip(idx, lam) if w > 1e-9]
                best_v, best_pts, best_norm2 = v, active or [points[idx[0]]], norm2
    if best_v is None or best_pts is None:
        norms = [float(p @ p) for p in points]
        i = int(np.argmin(norms))
        return points[i], [points[i]]
    return best_v, best_pts


def cylinder_cylinder_dist(p1: np.ndarray, q1: np.ndarray,
                           p2: np.ndarray, q2: np.ndarray,
                           radius: float = R_A,
                           max_iter: int = 64) -> float:
    """两个平端实心圆柱的欧氏距离（Gilbert/GJK 支撑映射算法）。

    返回 0 表示相交。调用方先用外接球柱体做候选预筛，因此本函数只处理端帽形状
    可能改变结论的少量临界对。停止条件使用 Frank–Wolfe 对偶间隙；坐标量级为 nm，
    最终误差远小于 1.8 nm 导通阈值。
    """
    c1, c2 = 0.5 * (p1 + q1), 0.5 * (p2 + q2)

    def support(direction: np.ndarray) -> np.ndarray:
        return (cylinder_support(p1, q1, direction, radius)
                - cylinder_support(p2, q2, -direction, radius))

    direction = c2 - c1
    if float(direction @ direction) <= 1e-18:
        direction = np.array([1.0, 0.0, 0.0])
    simplex = [support(direction)]
    v = simplex[0]
    for _ in range(max_iter):
        norm2 = float(v @ v)
        if norm2 <= 1e-18:
            return 0.0
        w = support(-v)
        dual_gap = norm2 - float(v @ w)
        if dual_gap <= 1e-11 * max(1.0, norm2):
            return float(np.sqrt(norm2))
        if any(float((w - s) @ (w - s)) <= 1e-20 for s in simplex):
            return float(np.sqrt(norm2))
        v, simplex = _closest_origin_on_hull(simplex + [w])
    return float(np.linalg.norm(v))


def cylinder_x_extent(p: np.ndarray, q: np.ndarray,
                      radius: float = R_A) -> tuple[float, float]:
    """平端圆柱在 X 轴上的精确投影区间。"""
    axis = q - p
    length = float(np.linalg.norm(axis))
    if length <= 1e-12:
        x = float(0.5 * (p[0] + q[0]))
        return x - radius, x + radius
    ux = float(axis[0] / length)
    half_width = 0.5 * length * abs(ux) + radius * np.sqrt(max(0.0, 1.0 - ux * ux))
    cx = float(0.5 * (p[0] + q[0]))
    return cx - half_width, cx + half_width


def _inner_capsule_axis(p: np.ndarray, q: np.ndarray,
                        radius: float = R_A) -> tuple[np.ndarray, np.ndarray] | None:
    """返回平端圆柱的内接球柱体轴线；碎片短于直径时不使用该快捷判据。"""
    d = q - p
    length = float(np.linalg.norm(d))
    if length <= 2.0 * radius + 1e-12:
        return None
    u = d / length
    return p + radius * u, q - radius * u


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
# KS 检验 p=6.7e-19 拒绝球面均匀、p=0.73 不拒绝极角均匀（第 3.4 节）。
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
    mode="discard"：球心仍在整个盒内均匀生成，但丢弃任何与边界相交的球。
                    它是题设“球体截断后平移”几何的严格内界，专用于
                    可行性确认；不作为点估计口径。
    两种取法在灵敏度分析里对比。
    """
    if mode == "inside":
        c = (rng.random((n, 3)) - 0.5) * (box - 2 * R_B)
        return c, np.arange(n)

    c = (rng.random((n, 3)) - 0.5) * box
    half = box / 2.0
    if mode == "discard":
        keep = np.all(np.abs(c) + R_B <= half + 1e-12, axis=1)
        return c[keep], np.nonzero(keep)[0].astype(int)
    if mode != "wrap":
        raise ValueError(f"未知球体边界口径：{mode}")
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
               rod_geometry: str = "capsule",
               gap: float = GAP,
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

    if rod_geometry not in {"capsule", "cylinder"}:
        raise ValueError(f"未知介质 A 几何：{rod_geometry}")

    # --- 介质与带电面
    if n_rod:
        if rod_geometry == "capsule":
            lo = np.minimum(seg_p[:, 0], seg_q[:, 0]) - R_A
            hi = np.maximum(seg_p[:, 0], seg_q[:, 0]) + R_A
        else:
            ext = np.asarray([cylinder_x_extent(a, b) for a, b in zip(seg_p, seg_q)])
            lo, hi = ext[:, 0], ext[:, 1]
        for i in np.nonzero(lo <= -half + gap)[0]:
            dsu.union(LEFT, int(i))
        for i in np.nonzero(hi >= half - gap)[0]:
            dsu.union(RIGHT, int(i))
    if n_sph:
        for i in np.nonzero(sph_c[:, 0] - R_B <= -half + gap)[0]:
            dsu.union(LEFT, n_rod + int(i))
        for i in np.nonzero(sph_c[:, 0] + R_B >= half - gap)[0]:
            dsu.union(RIGHT, n_rod + int(i))

    # --- 棒-棒
    if n_rod > 1:
        for a, b in contact_pairs(seg_p, seg_q, 2 * R_A + gap):
            hit = rod_geometry == "capsule"
            if not hit:
                ia = _inner_capsule_axis(seg_p[a], seg_q[a])
                ib = _inner_capsule_axis(seg_p[b], seg_q[b])
                # 绝大多数接触也被 K^- 捕获，只对端缘窄带调用较贵的 GJK。
                hit = (ia is not None and ib is not None
                       and float(seg_seg_dist(ia[0], ia[1], ib[0], ib[1])) <= 2 * R_A + gap)
            if not hit and rod_geometry == "cylinder":
                hit = cylinder_cylinder_dist(
                    seg_p[a], seg_q[a], seg_p[b], seg_q[b]) <= gap + 1e-8
            if hit:
                dsu.union(int(a), int(b))

    # --- 球-球 与 球-棒（介质 B 数量可达上万，用 KD 树做邻域检索）
    if n_sph:
        from scipy.spatial import cKDTree
        tree = cKDTree(sph_c)
        if n_sph > 1:
            for a, b in tree.query_pairs(2 * R_B + gap, output_type="ndarray"):
                dsu.union(n_rod + int(a), n_rod + int(b))
        if n_rod:
            # 沿每根碎片轴线按 STEP 采样，查半径 = 接触半径 + 半个采样步长，
            # 保证不漏掉任何真实接触；再用精确的点—线段距离过滤候选。
            reach = R_A + R_B + gap
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
            # 同一棒—球对常被相邻多个轴线采样点重复命中。先编码去重，再一次性做精确
            # 距离过滤；混填边界上可把这部分耗时降低一个数量级。
            # 用两棵 KD 树的稀疏距离矩阵一次性取出所有候选。旧实现对
            # ``query_ball_point`` 返回的每个采样点在 Python 层循环，混填
            # 边界上一次试验要扫几万个 list，全局审计时这一步占
            # 了主要时间。稀疏矩阵返回的 (row,col) 与原候选集合完全
            # 相同；后面仍按“棒编号×球数+球编号”去重，并用精确距离
            # 过滤，因此不改变任何接触判定。
            sample_tree = cKDTree(pts)
            near = sample_tree.sparse_distance_matrix(
                tree, radius, output_type="coo_matrix")
            if near.nnz:
                code = np.unique(tags[near.row].astype(np.int64) * n_sph
                                 + near.col.astype(np.int64))
                rods = (code // n_sph).astype(int)
                spheres = (code % n_sph).astype(int)
                if rod_geometry == "capsule":
                    d = point_seg_dist(sph_c[spheres], seg_p[rods], seg_q[rods])
                    hit = d <= reach
                else:
                    d = point_cylinder_dist(sph_c[spheres], seg_p[rods], seg_q[rods])
                    hit = d <= R_B + gap
                for rod, sphere in zip(rods[hit], spheres[hit]):
                    dsu.union(n_rod + int(sphere), int(rod))

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
