#!/usr/bin/env python3
"""蒙特卡洛导通概率估计（可复现、并行）。

一次"试验"= 随机生成一个配置 -> 边界截断 -> 建接触图 -> 判导通。
导通概率就是伯努利参数 p，用 Wilson 区间给置信区间（样本比例接近 0 或 1 时，
正态近似的 p±1.96·se 会给出越界的区间，Wilson 不会）。

所有随机性来自一个 SeedSequence，按**固定的 N_CHUNKS 个分块**分叉——注意分块数
与并行度（进程数）是两件事：分块决定随机流怎么切，进程数只决定谁来跑。
早先的实现把两者绑在一起（`spawn(min(os.cpu_count(), 8))`），子种子于是依赖机器核数，
换一台 4 核机器重跑，同一配置会得到不同的数（实测 T=240、N_A=354 时
8/4/2/1 进程分别得 17/19/23/28 次导通）。现在分块数写死为 N_CHUNKS，
进程数怎么变都不影响结果，换机器重跑才真正能逐位复现。

N_CHUNKS=8 与生成 results/ 时所用的分块数一致，因此本次修正**不改变任何已发布的数值**。
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

# 随机流的分块数。**必须是常数**：子种子由 SeedSequence(seed).spawn(N_CHUNKS) 决定，
# 一旦让它随 os.cpu_count() 变化，结果就跟机器核数绑定了（见模块文档）。
# 取 4 是因为 results/ 里的全部数值是在 min(cpu_count, 8) = 4 的机器上生成的，
# 写死成 4 才能让已发布的结果在任意机器上原样复现；换成别的值会得到一组同样有效、
# 但与已发布数值不同的估计。
N_CHUNKS = 4


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
    # 分块：固定 N_CHUNKS 块，与进程数无关，保证随机流与机器核数解耦
    base = trials // N_CHUNKS
    sizes = [base + (1 if i < trials - base * N_CHUNKS else 0) for i in range(N_CHUNKS)]
    seeds = np.random.SeedSequence(seed).spawn(N_CHUNKS)
    jobs = [(asdict(cfg), s, k) for s, k in zip(seeds, sizes) if k > 0]

    # 进程数：只影响跑多快，不影响跑出什么
    workers = workers or min(os.cpu_count() or 1, N_CHUNKS)
    workers = max(1, min(workers, len(jobs)))

    if workers == 1 or len(jobs) == 1:
        hits = sum(_run_chunk(j) for j in jobs)
    else:
        with Pool(workers) as pool:
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
