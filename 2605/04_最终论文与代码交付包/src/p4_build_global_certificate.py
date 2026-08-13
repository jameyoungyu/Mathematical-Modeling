#!/usr/bin/env python3
"""汇总问题四的全局最优证书，并对证据链作硬断言。"""

from __future__ import annotations

import json
from pathlib import Path

from p4_cost_line_sweep import COST_UNIT, RESULTS


def read(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> int:
    search = read("p4_saa_search_0p920_wrap.json")
    low = read("p4_low_a_outer_audit.json")
    inner_saa = read("p4_inner_saa_619_8_discard.json")
    validation = read("p4_validation_619_8_discard.json")
    prev_line = read("p4_line_351484_cylinder_wrap.json")
    first_line = read("p4_line_351485_cylinder_wrap.json")

    rec = search["recommended"]
    if (rec["n_a"], rec["n_b"], rec["cost_tick"]) != (619, 8, 351485):
        raise AssertionError(f"搜索候选异常：{rec}")
    if search["last_infeasible_q"] != 351484:
        raise AssertionError("未将候选前一刻度排除")
    if not low["all_low_a_columns_excluded"]:
        raise AssertionError("A<580 的低成本区域尚未全部排除")

    inner_row = next(r for r in inner_saa["rows"]
                     if r["n_a"] == 619 and r["n_b"] == 8)
    if inner_row["hits"] < 552:
        raise AssertionError("候选在球缺内界的鲁棒 SAA 中未达 552/600")
    val = validation["rows"][0]
    if val["cp995_one_sided_lo"] < 0.90:
        raise AssertionError("独立验证的 99.5% 单侧下界未达 0.90")
    prev_best = max(prev_line["rows"], key=lambda r: r["hits"])
    first_best = max(first_line["rows"], key=lambda r: r["hits"])

    out = {
        "claim": "global minimum of the prespecified robust SAA problem",
        "model_scope": (
            "flat-ended cylinders after centerline truncation; true clipped-sphere geometry "
            "bracketed by discard inner bound and full-sphere wrap outer bound"),
        "cost_integerization": {
            "unit_yuan": COST_UNIT,
            "formula": "Q = 567*N_A + 64*N_B",
        },
        "target_probability": 0.90,
        "robust_saa": {
            "trials": 600,
            "seed": 20260824,
            "safety_target": 0.92,
            "required_hits": 552,
            "last_infeasible_q": 351484,
            "first_feasible_q": 351485,
            "recommended_outer_hits": rec["hits"],
            "recommended_inner_hits": inner_row["hits"],
            "boundary_pair": {
                "last_infeasible_best": prev_best,
                "first_feasible_best": first_best,
            },
            "high_a_search_range": search["a_range"],
            "low_a_outer_audit": low["partitions"],
        },
        "recommended": {
            "n_a": 619,
            "n_b": 8,
            "cost_tick": 351485,
            "cost_yuan": 351485 * COST_UNIT,
        },
        "independent_validation": val,
        "global_optimality": {
            "under_robust_saa_rule": True,
            "deterministic_true_probability_theorem": False,
            "covered_integer_domain": {
                "n_a_0_to_579": "partitioned outer audit with exact-cylinder fallback",
                "n_a_580_to_624": "every cost-line column searched during bisection",
                "n_a_at_least_625": "A-only cost already exceeds the candidate cost",
            },
            "explanation": (
                "Every lower-cost integer pair is covered: A<580 by partitioned outer audits, "
                "A=580..624 by complete cost-line search, and A>=625 by the cost bound. All covered "
                "pairs fail the 92% common-scenario rule in an optimistic geometry; the candidate "
                "passes that rule in the conservative inner geometry and its independent CP99.5% "
                "lower bound exceeds 0.90."),
        },
    }
    (RESULTS / "p4_global_certificate.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("全局证书通过：(619,8), C=%.6f 元" % out["recommended"]["cost_yuan"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
