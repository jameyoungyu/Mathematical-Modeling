#!/usr/bin/env python3
"""问题四：用共同随机情景一次扫描所有纯 A 配比。

对每个随机情景只生成一次 ``max_a`` 根介质 A，建立平端圆柱
的完整接触图，然后按母介质编号从小到大激活节点。由于加入介质
不会删除已有通路，每个情景只需记录首次导通的根数，即可同时
得到 ``N_A=0..max_a`` 的导通频率。这比对每个根数独立重建几何图
快两个数量级，且每个边际分布仍是题设的独立均匀放置。

输出 ``results/p4_pure_a_sweep.json``。该文件的 Clopper--Pearson 单侧
下界用于确认纯 A 可行候选；全局最优性还需由
``p4_global_certificate.py`` 排除所有更便宜的混填点。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import beta

from microstructure import (DSU, EDGE, GAP, R_A, _inner_capsule_axis,
                            contact_pairs, cylinder_cylinder_dist,
                            cylinder_x_extent, sample_rods, seg_seg_dist)
from simulate import BOX, Config, N_RANDOM_STREAMS, wilson

RESULTS = Path(__file__).resolve().parents[1] / "results"
DST = RESULTS / "p4_pure_a_sweep.json"
TARGET = 0.90


def cp_one_sided(hits: int, trials: int, alpha: float) -> tuple[float, float]:
    """Clopper--Pearson 单侧 ``1-alpha`` 下、上界。"""
    lo = 0.0 if hits == 0 else float(beta.ppf(alpha, hits, trials - hits + 1))
    hi = 1.0 if hits == trials else float(beta.ppf(1.0 - alpha, hits + 1, trials - hits))
    return lo, hi


def critical_count(max_a: int, rng: np.random.Generator,
                   orientation: str) -> int:
    """返回当前情景首次导通的母介质 A 根数。"""
    p, q, owner = sample_rods(max_a, rng, BOX, orientation=orientation)
    n_frag = len(p)
    left, right = n_frag, n_frag + 1

    events: list[list[tuple[int, int]]] = [[] for _ in range(max_a)]
    lo_hi = np.asarray([cylinder_x_extent(a, b) for a, b in zip(p, q)])
    half = EDGE / 2.0
    for i in np.nonzero(lo_hi[:, 0] <= -half + GAP)[0]:
        events[int(owner[i])].append((left, int(i)))
    for i in np.nonzero(lo_hi[:, 1] >= half - GAP)[0]:
        events[int(owner[i])].append((right, int(i)))

    # 与 microstructure.percolates 的平端圆柱判定完全同构：
    # K+ 列候选 -> K- 快速确认 -> 窄带内调用 GJK。
    for a, b in contact_pairs(p, q, 2 * R_A + GAP):
        ia = _inner_capsule_axis(p[a], q[a])
        ib = _inner_capsule_axis(p[b], q[b])
        hit = (ia is not None and ib is not None
               and float(seg_seg_dist(ia[0], ia[1], ib[0], ib[1]))
               <= 2 * R_A + GAP)
        if not hit:
            hit = cylinder_cylinder_dist(p[a], q[a], p[b], q[b]) <= GAP + 1e-8
        if hit:
            activate_at = max(int(owner[a]), int(owner[b]))
            events[activate_at].append((int(a), int(b)))

    dsu = DSU(n_frag + 2)
    for owner_idx, bucket in enumerate(events):
        for a, b in bucket:
            dsu.union(a, b)
        if dsu.find(left) == dsu.find(right):
            return owner_idx + 1
    return max_a + 1


def _run_chunk(args: tuple[dict, np.random.SeedSequence, int, int]) -> np.ndarray:
    cfg_dict, seed_state, trials, max_a = args
    cfg = Config(**cfg_dict)
    rng = np.random.default_rng(seed_state)
    hist = np.zeros(max_a + 2, dtype=np.int64)
    for _ in range(trials):
        hist[critical_count(max_a, rng, cfg.orientation)] += 1
    return hist


def sweep(max_a: int, trials: int, seed: int, workers: int,
          orientation: str) -> dict:
    chunks = min(N_RANDOM_STREAMS, trials)
    base = trials // chunks
    sizes = [base + (i < trials - base * chunks) for i in range(chunks)]
    seeds = np.random.SeedSequence(seed).spawn(chunks)
    cfg = asdict(Config(n_a=max_a, rod_geometry="cylinder",
                        orientation=orientation))
    jobs = [(cfg, s, int(k), max_a) for s, k in zip(seeds, sizes)]
    if chunks == 1:
        hist = _run_chunk(jobs[0])
    else:
        with Pool(min(workers, chunks)) as pool:
            hist = sum(pool.map(_run_chunk, jobs), start=np.zeros(max_a + 2, dtype=np.int64))

    # critical<=n 当且仅当前 n 根导通。
    hits = np.cumsum(hist)[:max_a + 1]
    alpha_feasible = 0.005  # 与全局排除的 0.005 合计为 1%
    rows = []
    for n_a, h in enumerate(hits):
        lo_cp, hi_cp = cp_one_sided(int(h), trials, alpha_feasible)
        lo99, hi99 = wilson(int(h), trials, z=2.5758293035489004)
        rows.append({
            "n_a": n_a,
            "hits": int(h),
            "trials": trials,
            "p": float(h / trials),
            "wilson99_lo": lo99,
            "wilson99_hi": hi99,
            "cp995_one_sided_lo": lo_cp,
            "cp995_one_sided_hi": hi_cp,
            "confirmed": bool(lo_cp >= TARGET),
        })
    confirmed = [r for r in rows if r["confirmed"]]
    return {
        "model_scope": "finite flat-ended cylinders after centerline truncation",
        "sampling": "common-random prefix sweep; every marginal remains iid uniform",
        "target": TARGET,
        "seed": seed,
        "trials": trials,
        "max_a": max_a,
        "alpha_feasible": alpha_feasible,
        "critical_count_histogram": hist.tolist(),
        "rows": rows,
        "first_confirmed": confirmed[0] if confirmed else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-a", type=int, default=630)
    ap.add_argument("--trials", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--orientation", default="isotropic")
    args = ap.parse_args()
    t0 = time.time()
    state = sweep(args.max_a, args.trials, args.seed, args.workers, args.orientation)
    state["seconds"] = round(time.time() - t0, 1)
    RESULTS.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    r = state["first_confirmed"]
    if r:
        print(f"首个确认可行的纯 A 点：N_A={r['n_a']}  "
              f"P={r['p']:.4f}  CP99.5%单侧下界={r['cp995_one_sided_lo']:.4f}")
    else:
        print(f"N_A<= {args.max_a} 中没有确认可行点")
    print(f"耗时 {state['seconds']:.1f} s，结果写入 {DST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
