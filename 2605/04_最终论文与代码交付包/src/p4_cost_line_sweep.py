#!/usr/bin/env python3
"""问题四：在一条整数成本线上同时估计所有混填配比。

成本之比恰为 ``c_A/c_B = 567/64``，因此可以用整数刻度
``Q=567*N_A+64*N_B`` 无浮点误差地表示预算。对每个随机情景，
本脚本一次生成最大数量的 A、B，一次建立接触图；每条边在成本线
的 A 列上都只在一个连续区间内激活，故用“时间线段树 + 可回滚
并查集”一次回答整条成本线的导通性。

``sphere_mode=wrap`` 把球缺扩大为完整球，是题设截断球的外界，
用于排除低成本点；``sphere_mode=discard`` 丢弃越界球，是内界，
用于确认候选点。若同一配比在内界达到预设 SAA 判据，且所有更便宜
配比在外界未达到该判据，则得到本文中心线截断模型下的鲁棒 SAA
全局最优；有限样本结论不等同于真概率的确定性全局定理。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import beta

from microstructure import (COST_A, COST_B, EDGE, GAP, R_A, R_B,
                            _inner_capsule_axis, contact_pairs,
                            cylinder_cylinder_dist, cylinder_x_extent,
                            point_cylinder_dist, point_seg_dist, sample_rods,
                            sample_spheres, seg_seg_dist)
from simulate import BOX, N_RANDOM_STREAMS, wilson

RESULTS = Path(__file__).resolve().parents[1] / "results"
TARGET = 0.90
COST_UNIT = COST_B / 64.0
assert abs(COST_A / COST_UNIT - 567.0) < 1e-10


class RollbackDSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.stack: list[tuple[int, int, int] | None] = []

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a == b:
            self.stack.append(None)
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.stack.append((b, a, self.size[a]))
        self.parent[b] = a
        self.size[a] += self.size[b]

    def snapshot(self) -> int:
        return len(self.stack)

    def rollback(self, snap: int) -> None:
        while len(self.stack) > snap:
            item = self.stack.pop()
            if item is None:
                continue
            b, a, old_size = item
            self.parent[b] = b
            self.size[a] = old_size


def cp_one_sided(hits: int, trials: int, alpha: float) -> tuple[float, float]:
    lo = 0.0 if hits == 0 else float(beta.ppf(alpha, hits, trials - hits + 1))
    hi = 1.0 if hits == trials else float(beta.ppf(1 - alpha, hits + 1, trials - hits))
    return lo, hi


def _material_interval(kind: str, owner: int, a_values: np.ndarray,
                       b_values: np.ndarray) -> tuple[int, int] | None:
    if kind == "rod":
        idx = int(np.searchsorted(a_values, owner + 1, side="left"))
        return None if idx >= len(a_values) else (idx, len(a_values) - 1)
    active = np.nonzero(b_values > owner)[0]
    return None if active.size == 0 else (0, int(active[-1]))


def _intersect(x: tuple[int, int] | None,
               y: tuple[int, int] | None) -> tuple[int, int] | None:
    if x is None or y is None:
        return None
    lo, hi = max(x[0], y[0]), min(x[1], y[1])
    return None if lo > hi else (lo, hi)


def graph_edges(sample_max_a: int, max_b: int, rng: np.random.Generator,
                rod_geometry: str, sphere_mode: str,
                active_max_a: int | None = None):
    # 始终按固定 sample_max_a 生成随机棒，保证跨预算、跨分区的
    # 前缀几何一致；建图时只保留当前分区可能用到的母介质，
    # 避免为“低 A 高 B”分区构建不可能同时激活的 630×5500 交叉图。
    p, q, owner_r = sample_rods(sample_max_a, rng, BOX)
    if active_max_a is not None and active_max_a < sample_max_a:
        keep = owner_r < active_max_a
        p, q, owner_r = p[keep], q[keep], owner_r[keep]
    sc, owner_s = sample_spheres(max_b, rng, BOX, mode=sphere_mode)
    n_r, n_s = len(p), len(sc)
    left, right = n_r + n_s, n_r + n_s + 1
    edges: list[tuple[int, int]] = []

    half = EDGE / 2.0
    if rod_geometry == "cylinder":
        ext = np.asarray([cylinder_x_extent(a, b) for a, b in zip(p, q)])
        lo, hi = ext[:, 0], ext[:, 1]
    else:
        lo = np.minimum(p[:, 0], q[:, 0]) - R_A
        hi = np.maximum(p[:, 0], q[:, 0]) + R_A
    edges.extend((left, int(i)) for i in np.nonzero(lo <= -half + GAP)[0])
    edges.extend((right, int(i)) for i in np.nonzero(hi >= half - GAP)[0])
    edges.extend((left, n_r + int(i)) for i in np.nonzero(sc[:, 0] - R_B <= -half + GAP)[0])
    edges.extend((right, n_r + int(i)) for i in np.nonzero(sc[:, 0] + R_B >= half - GAP)[0])

    for a, b in contact_pairs(p, q, 2 * R_A + GAP):
        hit = rod_geometry == "capsule"
        if not hit:
            ia, ib = _inner_capsule_axis(p[a], q[a]), _inner_capsule_axis(p[b], q[b])
            hit = (ia is not None and ib is not None
                   and float(seg_seg_dist(ia[0], ia[1], ib[0], ib[1])) <= 2 * R_A + GAP)
        if not hit:
            hit = cylinder_cylinder_dist(p[a], q[a], p[b], q[b]) <= GAP + 1e-8
        if hit:
            edges.append((int(a), int(b)))

    if n_s:
        tree = cKDTree(sc)
        if n_s > 1:
            edges.extend((n_r + int(a), n_r + int(b))
                         for a, b in tree.query_pairs(2 * R_B + GAP, output_type="ndarray"))
        if n_r:
            reach, step = R_A + R_B + GAP, 200.0
            samples, tags = [], []
            for i, (a, b) in enumerate(zip(p, q)):
                length = float(np.linalg.norm(b - a))
                k = max(2, int(np.ceil(length / step)) + 1)
                t = np.linspace(0.0, 1.0, k)[:, None]
                samples.append(a + t * (b - a))
                tags.append(np.full(k, i, dtype=int))
            pts, tags = np.concatenate(samples), np.concatenate(tags)
            near = cKDTree(pts).sparse_distance_matrix(
                tree, reach + step / 2.0 + 1e-9, output_type="coo_matrix")
            if near.nnz:
                code = np.unique(tags[near.row].astype(np.int64) * n_s
                                 + near.col.astype(np.int64))
                rods, spheres = (code // n_s).astype(int), (code % n_s).astype(int)
                if rod_geometry == "cylinder":
                    dist = point_cylinder_dist(sc[spheres], p[rods], q[rods])
                    keep = dist <= R_B + GAP
                else:
                    dist = point_seg_dist(sc[spheres], p[rods], q[rods])
                    keep = dist <= reach
                edges.extend((int(r), n_r + int(s)) for r, s in zip(rods[keep], spheres[keep]))
    kinds = ["rod"] * n_r + ["sphere"] * n_s
    owners = np.r_[owner_r, owner_s].astype(int)
    return edges, kinds, owners, left, right


def one_scenario(q_tick: int, a_min: int, a_max: int, sample_max_a: int,
                 rng: np.random.Generator, rod_geometry: str,
                 sphere_mode: str) -> np.ndarray:
    a_values = np.arange(a_min, a_max + 1, dtype=int)
    b_values = np.maximum(-1, (q_tick - 567 * a_values) // 64).astype(int)
    max_b = int(max(0, b_values[0]))
    edges, kinds, owners, left, right = graph_edges(
        sample_max_a, max_b, rng, rod_geometry, sphere_mode,
        active_max_a=a_max)
    n_states = len(a_values)
    always = (0, n_states - 1)
    intervals: list[tuple[int, int] | None] = [
        _material_interval(k, int(o), a_values, b_values)
        for k, o in zip(kinds, owners)
    ] + [always, always]

    seg: list[list[tuple[int, int]]] = [[] for _ in range(4 * n_states)]

    def add(node: int, lo: int, hi: int, ql: int, qr: int,
            edge: tuple[int, int]) -> None:
        if ql <= lo and hi <= qr:
            seg[node].append(edge)
            return
        mid = (lo + hi) // 2
        if ql <= mid:
            add(node * 2, lo, mid, ql, qr, edge)
        if qr > mid:
            add(node * 2 + 1, mid + 1, hi, ql, qr, edge)

    for u, v in edges:
        active = _intersect(intervals[u], intervals[v])
        if active is not None:
            add(1, 0, n_states - 1, active[0], active[1], (u, v))

    dsu = RollbackDSU(len(kinds) + 2)
    out = np.zeros(n_states, dtype=np.int64)

    def visit(node: int, lo: int, hi: int) -> None:
        snap = dsu.snapshot()
        for u, v in seg[node]:
            dsu.union(u, v)
        if lo == hi:
            out[lo] = int(dsu.find(left) == dsu.find(right))
        else:
            mid = (lo + hi) // 2
            visit(node * 2, lo, mid)
            visit(node * 2 + 1, mid + 1, hi)
        dsu.rollback(snap)

    visit(1, 0, n_states - 1)
    return out


def _run_chunk(args) -> np.ndarray:
    (q_tick, a_min, a_max, sample_max_a, seed, start, trials,
     rod_geometry, sphere_mode) = args
    hits = np.zeros(a_max - a_min + 1, dtype=np.int64)
    for scenario_id in range(start, start + trials):
        # 每个情景由（总种子，情景编号）独立派生。这样改变预算、
        # 球体内/外界或 worker 数时，同编号情景的 A、B 前缀仍完全
        # 相同，可以逐情景验证 K^- <= K <= K^+ 与成本单调性。
        rng = np.random.default_rng(np.random.SeedSequence([seed, scenario_id]))
        hits += one_scenario(q_tick, a_min, a_max, sample_max_a, rng,
                             rod_geometry, sphere_mode)
    return hits


def sweep(q_tick: int, a_min: int, a_max: int, trials: int, seed: int,
          workers: int, rod_geometry: str, sphere_mode: str,
          sample_max_a: int = 630) -> dict:
    chunks = min(N_RANDOM_STREAMS, trials)
    base = trials // chunks
    sizes = [base + (i < trials - base * chunks) for i in range(chunks)]
    starts = np.cumsum([0] + sizes[:-1]).astype(int).tolist()
    jobs = [(q_tick, a_min, a_max, sample_max_a, seed, start, int(k),
             rod_geometry, sphere_mode)
            for start, k in zip(starts, sizes)]
    with Pool(min(workers, chunks)) as pool:
        hits = sum(pool.map(_run_chunk, jobs), start=np.zeros(a_max - a_min + 1, dtype=np.int64))
    rows = []
    for i, h in enumerate(hits):
        a = a_min + i
        b = max(-1, (q_tick - 567 * a) // 64)
        lo_cp, hi_cp = cp_one_sided(int(h), trials, 0.005)
        lo99, hi99 = wilson(int(h), trials, z=2.5758293035489004)
        rows.append({"n_a": a, "n_b": int(b), "cost_tick": int(567*a+64*b),
                     "cost_yuan": float((567*a+64*b)*COST_UNIT),
                     "hits": int(h), "trials": trials, "p": float(h/trials),
                     "wilson99_lo": lo99, "wilson99_hi": hi99,
                     "cp995_one_sided_lo": lo_cp, "cp995_one_sided_hi": hi_cp,
                     "confirmed": bool(lo_cp >= TARGET),
                     "excluded": bool(hi_cp < TARGET)})
    return {"q_tick": q_tick, "budget_yuan": q_tick*COST_UNIT,
            "a_min": a_min, "a_max": a_max, "trials": trials, "seed": seed,
            "sample_max_a": sample_max_a,
            "rod_geometry": rod_geometry, "sphere_mode": sphere_mode,
            "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--q-tick", type=int)
    g.add_argument("--cost", type=float)
    ap.add_argument("--a-min", type=int, default=560)
    ap.add_argument("--a-max", type=int, default=624)
    ap.add_argument("--sample-max-a", type=int, default=630,
                    help="固定的情景生成上限；跨预算比较时不得改变")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--rod-geometry", choices=("capsule", "cylinder"), default="cylinder")
    ap.add_argument("--sphere-mode", choices=("wrap", "discard"), default="wrap")
    ap.add_argument("--output")
    args = ap.parse_args()
    q_tick = args.q_tick if args.q_tick is not None else math.floor(args.cost/COST_UNIT + 1e-9)
    a_max = min(args.a_max, q_tick // 567)
    t0 = time.time()
    state = sweep(q_tick, args.a_min, a_max, args.trials, args.seed,
                  args.workers, args.rod_geometry, args.sphere_mode,
                  max(args.sample_max_a, a_max))
    state["seconds"] = round(time.time() - t0, 1)
    dst = Path(args.output) if args.output else RESULTS / f"p4_line_{q_tick}_{args.rod_geometry}_{args.sphere_mode}.json"
    dst.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    best = max(state["rows"], key=lambda r: r["p"])
    confirmed = [r for r in state["rows"] if r["confirmed"]]
    print(f"Q={q_tick}, C<={state['budget_yuan']:.4f}, 最高观测点 "
          f"({best['n_a']},{best['n_b']}): P={best['p']:.4f}")
    if confirmed:
        c = min(confirmed, key=lambda r: r["cost_tick"])
        print(f"已确认点：({c['n_a']},{c['n_b']}) "
              f"CP99.5%下界={c['cp995_one_sided_lo']:.4f}")
    print(f"耗时 {state['seconds']:.1f}s，结果 {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
