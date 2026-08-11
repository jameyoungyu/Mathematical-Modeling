#!/usr/bin/env python3
"""问题四成本下界的**完整**穷举审计（补上 p4_global_audit.py 的覆盖缺口）。

原审计（`p4_global_audit.py`）的问题，评审复核时查实：

1. 预算线阶段是 `for a in range(lo, hi + 1, 16)`，只取到 $N_A=592$；
   `a=608` 时 `nb_max<0` 被跳过，于是 $N_A\\in(592,608)$ 整段没有任何排除依据。
2. 预算线判据 $P(a,N_B^{\\max}(a))<0.90$ 只能排除**该列**（以及 $N_A\\le a$ 且
   $N_B\\le N_B^{\\max}(a)$ 的点），对 $N_A>a$ 的列无效；16 的步长在相邻两点之间
   留下了薄片。
3. 表 9 里 $(604,0)$（8.9658 元，比宣称的下界 9.0239 元还低）被判为"否"用的是
   点估计 0.8943，而本审计声明的判据是 Wilson **上界** < 0.90——它的上界是 0.9019，
   按论文自己的标准并未被排除。

**正确的判据。** 设待证成本 $C^\\*=c_AN_A^\\*$。对固定的 $N_A=a$，成本低于 $C^\\*$ 等价于
$N_B\\le N_B^{\\max}(a)=\\lceil (C^\\*-c_Aa)/c_B\\rceil-1$。由 $P$ 对 $N_B$ 单调，

    该列存在比 C* 便宜的可行点  <=>  P(a, N_B^max(a)) >= 0.90 。

所以**逐列在预算线上取点，是判定"C* 是否最优"的充要检验**，不是充分条件。
把 $a$ 跑遍 $[0,N_A^\\*)$ 的每一个整数，结论就是严格的。

**怎么跑得动。** 逐列 608 次仿真太贵，故分两阶段：

- 阶段 1（角点，便宜）：取网格 $a_1<a_2$，区间内任意成本低于 $C^\\*$ 的点都被角点
  $(a_2,\\,N_B^{\\max}(a_1))$ 按分量支配，一次仿真排除整条区间。步长越小角点越紧。
  本脚本对 $[0,480]$ 直接复用原审计已算出的步长 24 的结果（那 20 个区间已严格排除），
  对 $(480,608]$ 用步长 4 重算。
- 阶段 2（逐列，严格）：阶段 1 没排除掉的区间，逐个整数 $a$ 在预算线上取点。
- 阶段 3（擦边点加样本）：阶段 2 中 Wilson 上界够到 0.90 的点，换独立种子加到 T_FINE
  复核；另把 $(604,0)$ 单独加到 T_FINE，把表 9 那处判据不一致补上。

结果写入 results/p4_global_audit2.json，边算边写，可断点续跑。
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from microstructure import COST_A, COST_B
from simulate import Config, estimate_p, wilson

RESULTS = Path(__file__).resolve().parents[1] / "results"
DST = RESULTS / "p4_global_audit2.json"
TARGET = 0.90
# 样本量阶梯：先便宜地筛，排不掉再加样本。排除判据用的是**最终那一档**的 Wilson 上界。
T_LADDER = (1000, 6000, 20000)
# 排除判据取 99% Wilson 上界（z=2.576）而非 95%：本审计要做上百次判定，
# 95% 下的族错误率不可忽略；99% 更保守，代价只是多算一点。
Z_EXCLUDE = 2.5758293035489004
SEED_FINE = 20260811     # 擦边点复核用的独立种子
CORNER_STEP = 4          # [480, 560] 上的角点步长（原审计用 24，太松）
LOW_COVERED = 480        # [0, 480] 已由原审计的步长 24 角点严格排除
COL_FROM = 561           # [561, 607] 逐列取预算线点（N_B 小、跑得快、且最贴近最优解）


def nb_max(cost_star: float, n_a: int) -> int:
    """成本严格低于 cost_star 时，该 N_A 允许的最大 N_B。"""
    return max(-1, math.ceil((cost_star - n_a * COST_A) / COST_B) - 1)


def judge(n_a: int, n_b: int, seed: int = 20260810) -> dict:
    """沿样本量阶梯自适应地判定一个点。

    每一档都用 99% Wilson 区间判：上界 < 0.90 -> 排除；下界 >= 0.90 -> 可行性确认；
    否则加样本。**报告用的是最终那一档的区间**，阶梯只决定何时停止花算力。
    这是一个序贯停止规则，故判据本身取得比 95% 更严，见模块开头的说明。
    """
    r = None
    for t in T_LADDER:
        r = estimate_p(Config(n_a=n_a, n_b=n_b), t, seed=seed)
        lo, hi = wilson(r["hits"], t, z=Z_EXCLUDE)
        r["ci99_lo"], r["ci99_hi"] = lo, hi
        if hi < TARGET or lo >= TARGET:
            break
    return {"n_a": n_a, "n_b": n_b, "cost": n_a * COST_A + n_b * COST_B,
            "p": r["p"], "trials": r["trials"], "seed": seed,
            "ci95_lo": r["ci_lo"], "ci95_hi": r["ci_hi"],
            "ci99_lo": r["ci99_lo"], "ci99_hi": r["ci99_hi"],
            "excluded": bool(r["ci99_hi"] < TARGET),
            "feasible_confirmed": bool(r["ci99_lo"] >= TARGET)}


def load_state() -> dict:
    if DST.exists():
        try:
            s = json.loads(DST.read_text(encoding="utf-8"))
            s.setdefault("model_scope", "spherocylinder outer-bound model K+")
            s.setdefault("corners", {}); s.setdefault("columns", {})
            return s
        except Exception:
            pass
    return {"model_scope": "spherocylinder outer-bound model K+",
            "corners": {}, "columns": {}}


def save(state: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def coverage_status(state: dict, n_a_star: int) -> dict:
    """检查审计是否覆盖了待证区域；覆盖不完整时不得生成成本下界。"""
    columns = state["columns"]

    # [COL_FROM, N_A*) 必须逐列完成。
    missing_high = [a for a in range(COL_FROM, n_a_star) if str(a) not in columns]

    # [LOW_COVERED, COL_FROM] 的每段要么被角点排除，要么该段已逐列完成。
    grid = list(range(LOW_COVERED, COL_FROM + 1, CORNER_STEP))
    if grid[-1] != COL_FROM:
        grid.append(COL_FROM)
    missing_ranges = []
    for a1, a2 in zip(grid[:-1], grid[1:]):
        corner = state["corners"].get(f"{a1}-{a2}")
        if corner and corner.get("excluded"):
            continue
        missing = [a for a in range(a1, a2 + 1) if str(a) not in columns]
        if missing:
            missing_ranges.append({"a1": a1, "a2": a2, "missing_columns": missing})

    complete = not missing_high and not missing_ranges
    return {"complete": complete, "missing_high_columns": missing_high,
            "missing_corner_ranges": missing_ranges}


def summarize(state: dict, cost_star: float, n_a_star: int) -> None:
    """由已完成的判定生成摘要；覆盖不完整时只报告进度，不声称下界。"""
    done = list(state["columns"].values())
    open_pts = [c for c in done if not c["excluded"]]
    cheaper_ok = [c for c in open_pts
                  if c.get("feasible_confirmed") and c["cost"] < cost_star]
    undecided = [c for c in open_pts if not c.get("feasible_confirmed")]
    coverage = coverage_status(state, n_a_star)
    bound = min([c["cost"] for c in open_pts], default=cost_star)
    state["summary"] = {
        "columns_done": len(done),
        "corners_done": len(state["corners"]),
        "coverage": coverage,
        "proved_lower_bound_yuan": bound if coverage["complete"] else None,
        "undecided": sorted([{k: c[k] for k in ("n_a", "n_b", "cost", "p",
                                                "ci99_lo", "ci99_hi", "trials")}
                             for c in undecided], key=lambda z: z["cost"]),
        "confirmed_cheaper": sorted([{k: c[k] for k in ("n_a", "n_b", "cost", "p")}
                                     for c in cheaper_ok], key=lambda z: z["cost"]),
    }
    if not coverage["complete"]:
        v = (f"审计尚未完成：已完成 {len(done)} 列、{len(state['corners'])} 个角点；"
             "未覆盖区域仍可能存在更便宜的可行整数点，因此当前结果不得用于主张成本下界或全局最优。")
    elif cheaper_ok:
        v = (f"存在比 C*={cost_star:.4f} 元更便宜且可行性已确认的点，"
             f"最低 {min(c['cost'] for c in cheaper_ok):.4f} 元 —— C* 不是最优。")
    elif undecided:
        v = (f"没有任何更便宜的点被确认可行；{len(undecided)} 个点仍无法判定，"
             f"其中最便宜的为 {bound:.4f} 元。故最低总成本落在 "
             f"[{bound:.4f}, {cost_star:.4f}] 元，成本低于 {bound:.4f} 元的整数点均已严格排除。")
    else:
        v = f"所有成本低于 C*={cost_star:.4f} 元的整数点均已严格排除，C* 为全局最小。"
    state["summary"]["verdict"] = v


def main() -> int:
    best = json.loads((RESULTS / "p4_cost_optimum.json").read_text(encoding="utf-8"))["best"]
    n_a_star, cost_star = best["n_a"], best["cost_yuan"]
    state = load_state()
    state.update({"incumbent": {"n_a": n_a_star, "n_b": best["n_b"], "cost_yuan": cost_star},
                  "target": TARGET, "t_ladder": list(T_LADDER),
                  "z_exclude": Z_EXCLUDE, "seed_fine": SEED_FINE,
                  "low_covered_by_original_audit": LOW_COVERED})
    print(f"待证：C* = {cost_star:.4f} 元（N_A={n_a_star}, N_B={best['n_b']}）")
    print(f"[0,{LOW_COVERED}] 复用原审计步长 24 的角点结论（已严格排除）\n")

    # --- 阶段 1：[COL_FROM, N_A*-1] 逐列取预算线点，自高向低（最贴近最优解，且 N_B 小、跑得快）
    print(f"=== 阶段 1：逐列预算线 N_A∈[{COL_FROM},{n_a_star - 1}]，自高向低 ===")
    for a in range(n_a_star - 1, COL_FROM - 1, -1):
        key = str(a)
        if key not in state["columns"]:
            nb = nb_max(cost_star, a)
            if nb < 0:
                state["columns"][key] = {"n_a": a, "n_b": nb, "cost": a * COST_A,
                                         "note": "该列无更便宜的点", "excluded": True,
                                         "feasible_confirmed": False}
            else:
                t0 = time.time()
                state["columns"][key] = judge(a, nb)
                state["columns"][key]["secs"] = round(time.time() - t0, 1)
            summarize(state, cost_star, n_a_star); save(state)
        c = state["columns"][key]
        tag = ("已排除" if c["excluded"] else
               "★可行(更便宜)" if c.get("feasible_confirmed") else "★无法判定")
        print(f"  列 {a:3d}: 预算线=({a:3d},{c['n_b']:4d}) 成本={c['cost']:.4f} "
              f"P={c.get('p', float('nan')):.4f} T={c.get('trials', 0):5d} "
              f"99%区间=[{c.get('ci99_lo', 0):.4f},{c.get('ci99_hi', 0):.4f}] {tag}", flush=True)

    # --- 阶段 2：[480, COL_FROM] 用步长 4 的角点批量覆盖
    print(f"\n=== 阶段 2：角点，步长 {CORNER_STEP}，覆盖 N_A∈[{LOW_COVERED},{COL_FROM}] ===")
    grid = list(range(LOW_COVERED, COL_FROM + 1, CORNER_STEP))
    if grid[-1] != COL_FROM:
        grid.append(COL_FROM)
    for a1, a2 in zip(grid[:-1], grid[1:]):
        key = f"{a1}-{a2}"
        if key not in state["corners"]:
            nb = nb_max(cost_star, a1)
            t0 = time.time()
            r = judge(a2, nb)
            r.update({"a1": a1, "a2": a2, "secs": round(time.time() - t0, 1)})
            state["corners"][key] = r
            save(state)
        c = state["corners"][key]
        print(f"  N_A∈[{a1:3d},{a2:3d}] 角点=({a2:3d},{c['n_b']:5d}) "
              f"P={c.get('p', float('nan')):.4f} T={c.get('trials', 0):5d} "
              f"99%上界={c.get('ci99_hi', 0):.4f} "
              f"{'已排除' if c['excluded'] else '★未排除→需逐列'}", flush=True)
        if not c["excluded"]:
            for a in range(a1, a2 + 1):
                k2 = str(a)
                if k2 in state["columns"]:
                    continue
                nb2 = nb_max(cost_star, a)
                t0 = time.time()
                state["columns"][k2] = judge(a, nb2)
                state["columns"][k2]["secs"] = round(time.time() - t0, 1)
                summarize(state, cost_star, n_a_star); save(state)
                c2 = state["columns"][k2]
                print(f"    ↳ 列 {a:3d}: ({a:3d},{nb2:4d}) 成本={c2['cost']:.4f} "
                      f"P={c2['p']:.4f} {'已排除' if c2['excluded'] else '★未排除'}", flush=True)

    # --- 阶段 3：把仍未判定的点换独立种子复核，另补 (604,0)
    pend = [(c["n_a"], c["n_b"]) for c in state["columns"].values() if not c["excluded"]]
    pend.append((604, 0))
    state.setdefault("recheck", {})
    print(f"\n=== 阶段 3：独立种子 {SEED_FINE} 复核 {len(pend)} 点 ===")
    for a, nb in pend:
        key = f"{a}-{nb}"
        if key not in state["recheck"]:
            state["recheck"][key] = judge(a, nb, seed=SEED_FINE)
            save(state)
        c = state["recheck"][key]
        print(f"  ({a:3d},{nb:4d}) 成本={c['cost']:.4f} P={c['p']:.4f} T={c['trials']} "
              f"→ {'排除' if c['excluded'] else '可行性已确认' if c['feasible_confirmed'] else '无法判定'}",
              flush=True)

    summarize(state, cost_star, n_a_star); save(state)
    print("\n结论：" + state["summary"]["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
