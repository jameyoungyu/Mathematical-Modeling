#!/usr/bin/env python3
"""蒙特卡洛导通概率估计（可复现、并行）。

一次"试验"= 随机生成一个配置 -> 边界截断 -> 建接触图 -> 判导通。
导通概率就是伯努利参数 p，用 Wilson 区间给置信区间（样本比例接近 0 或 1 时，
正态近似的 p±1.96·se 会给出越界的区间，Wilson 不会）。

所有随机性来自一个 SeedSequence，并固定分成 8 条随机流；worker 只负责调度，
不参与随机流的生成。因此在 numpy 与算法版本相同的前提下，改变并行度仍可逐位复现。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import numpy as np

from microstructure import (EDGE, H_A, V_A, V_B, V_BOX, COST_A, COST_B,
                            PRIMARY_ORIENTATION, percolates, sample_rods,
                            sample_spheres)

BOX = np.full(3, EDGE)
N_RANDOM_STREAMS = 8


@dataclass(frozen=True)
class Config:
    n_a: int
    n_b: int = 0
    orientation: str = PRIMARY_ORIENTATION
    sphere_mode: str = "wrap"
    bond_fragments: bool = False
    # 介质 A 的轴长。默认 5000 即题给圆柱；geometry_bracket.py 用 5000-2r 得到
    # 内接球柱体，从而给出真实平端面圆柱的严格下界。
    rod_length: float = H_A

    @property
    def phi_a(self) -> float:
        return self.n_a * V_A / V_BOX

    @property
    def phi_b(self) -> float:
        return self.n_b * V_B / V_BOX

    @property
    def cost(self) -> float:
        return self.n_a * COST_A + self.n_b * COST_B


def _run_chunk(args) -> int:
    cfg_dict, seed_state, n_trials = args
    cfg = Config(**cfg_dict)
    rng = np.random.default_rng(seed_state)
    hits = 0
    for _ in range(n_trials):
        sp, sq, own_r = sample_rods(cfg.n_a, rng, BOX, length=cfg.rod_length,
                                    orientation=cfg.orientation)
        sc = own_s = None
        if cfg.n_b:
            sc, own_s = sample_spheres(cfg.n_b, rng, BOX, mode=cfg.sphere_mode)
        hits += bool(percolates(sp, sq, sph_c=sc, owner_rod=own_r, owner_sph=own_s,
                                bond_fragments=cfg.bond_fragments))
    return hits


def wilson(hits: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def estimate_p(cfg: Config, trials: int, seed: int = 20260810,
               workers: int | None = None) -> dict:
    """返回 {p, lo, hi, hits, trials, ...}。"""
    if trials <= 0:
        raise ValueError("trials 必须为正整数")
    workers = max(1, workers or min(os.cpu_count() or 1, N_RANDOM_STREAMS))
    # 随机流数固定，不随 workers 或机器核数变化。这样默认 8 核运行与历史结果完全一致，
    # 1/2/4/8 个 worker 只是以不同并行度调度同一组 jobs。
    n_chunks = min(N_RANDOM_STREAMS, trials)
    base = trials // n_chunks
    sizes = [base + (1 if i < trials - base * n_chunks else 0) for i in range(n_chunks)]
    sizes = [s for s in sizes if s > 0]
    seeds = np.random.SeedSequence(seed).spawn(len(sizes))
    jobs = [(asdict(cfg), s, k) for s, k in zip(seeds, sizes)]

    if len(jobs) == 1:
        hits = _run_chunk(jobs[0])
    else:
        with Pool(min(workers, len(jobs))) as pool:
            hits = sum(pool.map(_run_chunk, jobs))

    lo, hi = wilson(hits, trials)
    return {
        "n_a": cfg.n_a, "n_b": cfg.n_b,
        "phi_a": cfg.phi_a, "phi_b": cfg.phi_b,
        "orientation": cfg.orientation, "sphere_mode": cfg.sphere_mode,
        "bond_fragments": cfg.bond_fragments,
        "trials": trials, "hits": hits,
        "p": hits / trials, "ci_lo": lo, "ci_hi": hi,
        "half_width": (hi - lo) / 2,
        "cost_yuan": cfg.cost,
        "seed": seed,
    }


def n_rods_for_phi(phi: float) -> int:
    return int(round(phi * V_BOX / V_A))


def n_spheres_for_phi(phi: float) -> int:
    return int(round(phi * V_BOX / V_B))
