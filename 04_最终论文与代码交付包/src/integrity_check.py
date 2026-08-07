#!/usr/bin/env python3
"""Deterministic integrity checks for manuscript numbers and artifacts."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenario_model import PhysicalPowerModel, _r2, _rmse


ROOT = SRC.parent
DATA = ROOT / "data" / "full_timeseries_with_flags.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures_paper"
MANUSCRIPT = ROOT / "10_修订后完整论文_终稿.md"
OUT = ROOT / "integrity"


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return bool(abs(a - b) <= tol)


def main() -> None:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object, expected: object) -> None:
        checks.append({"check": name, "passed": bool(passed), "actual": actual, "expected": expected})

    df = pd.read_csv(DATA)
    check("raw_row_count", len(df) == 10080, len(df), 10080)
    check("outlet_missing_count", int(df["C_out_missing_flag"].sum()) == 50, int(df["C_out_missing_flag"].sum()), 50)
    valid = df["C_out_mgNm3"].dropna()
    check("outlet_valid_count", len(valid) == 10030, len(valid), 10030)
    check("outlet_min", close(float(valid.min()), 48.74), float(valid.min()), 48.74)
    check("outlet_max", close(float(valid.max()), 50.0), float(valid.max()), 50.0)
    check("outlet_cap50_count", int((valid == 50).sum()) == 5491, int((valid == 50).sum()), 5491)
    check("outlet_le10_count", int((valid <= 10).sum()) == 0, int((valid <= 10).sum()), 0)
    check("outlet_le5_count", int((valid <= 5).sum()) == 0, int((valid <= 5).sum()), 0)

    metrics = json.loads((RESULTS / "power_model_metrics.json").read_text())
    train, val, test = (df[df["split"] == s] for s in ("train", "validation", "test"))
    model = PhysicalPowerModel().fit(train, train["P_total_kW"].to_numpy(float))
    vp, tp = model.predict(val), model.predict(test)
    recomputed = {
        "validation_r2": _r2(val["P_total_kW"].to_numpy(float), vp),
        "validation_rmse_kW": _rmse(val["P_total_kW"].to_numpy(float), vp),
        "retrospective_test_r2": _r2(test["P_total_kW"].to_numpy(float), tp),
        "retrospective_test_rmse_kW": _rmse(test["P_total_kW"].to_numpy(float), tp),
        "retrospective_test_bias_kW": float(np.mean(tp - test["P_total_kW"].to_numpy(float))),
    }
    for key, value in recomputed.items():
        check(f"power_metric_{key}", close(value, float(metrics[key]), 1e-9), value, metrics[key])
    guard = float(np.quantile(np.abs(vp-val["P_total_kW"].to_numpy(float)), 0.90))
    check("validation_error_guard_q90", close(guard, float(metrics["validation_abs_error_q90_kW"]), 1e-9), guard, metrics["validation_abs_error_q90_kW"])
    check(
        "reporting_guard_rule_named_consistently",
        metrics.get("reporting_guard_rule") == "point_prediction_plus_validation_absolute_error_q90"
        and "conservative_reporting_rule" not in metrics,
        {k: metrics[k] for k in metrics if "reporting" in k or "guard" in k},
        {"reporting_guard_rule": "point_prediction_plus_validation_absolute_error_q90"},
    )
    check(
        "power_model_is_physical_specification",
        metrics["model"] == "physical_voltage_square_plus_rapping_frequency",
        metrics["model"], "physical_voltage_square_plus_rapping_frequency",
    )
    # the four field coefficients must be near-identical: the fields share a design
    b = np.asarray(metrics["b_field_kW_per_kV2"], float)
    check("field_coefficients_consistent", float(b.std() / b.mean()) < 0.01,
          float(b.std() / b.mean()), "<1% relative spread")

    baselines = pd.read_csv(RESULTS / "power_model_baselines.csv")
    tst_base = baselines[baselines["split"] == "retrospective_test"].set_index("model")
    check(
        "invT_beats_linear_T_on_test",
        float(tst_base.loc["physical_invT", "rmse_kW"]) < float(tst_base.loc["quadratic_ridge_T", "rmse_kW"]),
        tst_base["rmse_kW"].to_dict(), "physical_invT < quadratic_ridge_T",
    )
    check(
        "invT_removes_test_bias",
        abs(float(tst_base.loc["physical_invT", "bias_kW"])) < 1.0
        and abs(float(tst_base.loc["quadratic_ridge_T", "bias_kW"])) > 5.0,
        {"physical_invT": float(tst_base.loc["physical_invT", "bias_kW"]),
         "quadratic_ridge_T": float(tst_base.loc["quadratic_ridge_T", "bias_kW"])},
        "|bias| < 1 kW with 1/T, > 5 kW with linear T",
    )
    check(
        "condition_variables_add_nothing",
        abs(float(tst_base.loc["physical_invT_plus_conditions", "r2"])
            - float(tst_base.loc["physical_invT", "r2"])) < 1e-3,
        {"with": float(tst_base.loc["physical_invT_plus_conditions", "r2"]),
         "without": float(tst_base.loc["physical_invT", "r2"])},
        "delta R2 < 1e-3",
    )
    rolling = pd.read_csv(RESULTS / "power_model_rolling_validation.csv")
    check("rolling_fold_count", rolling["fold"].nunique() == 3, int(rolling["fold"].nunique()), 3)

    cluster = pd.read_csv(RESULTS / "cluster_selection_metrics.csv")
    selected = cluster[cluster["selected"]].iloc[0]
    check("selected_cluster_k", int(selected["k"]) == 4, int(selected["k"]), 4)
    check("silhouette_k4", close(float(selected["silhouette"]), 0.4083689, 1e-6), float(selected["silhouette"]), 0.4083689)

    optimum = pd.read_csv(RESULTS / "optimal_controls_central_scenario.csv")
    check(
        "central_parameter_table_complete",
        len(optimum) == 8
        and set(map(tuple, optimum[["condition_cluster", "limit_mgNm3"]].to_numpy()))
        == {(cluster, limit) for cluster in range(1, 5) for limit in (10.0, 5.0)},
        optimum[["condition_cluster", "limit_mgNm3"]].values.tolist(),
        "4 conditions × 2 targets",
    )
    check(
        "validation_error_columns_named_consistently",
        {"validation_error_adjusted_power_kW", "validation_error_guard_kW"}.issubset(optimum.columns)
        and {"conservative_power_kW", "power_guard_kW"}.isdisjoint(optimum.columns),
        sorted(optimum.columns),
        "new validation-error names present; old conservative names absent",
    )
    check("central_all_feasible", bool((optimum["status"] == "SCENARIO_FEASIBLE").all()), optimum["status"].unique().tolist(), ["SCENARIO_FEASIBLE"])
    check("central_limits_respected", bool((optimum["scenario_peak_total_mgNm3"] <= optimum["limit_mgNm3"] + 1e-9).all()), float((optimum["scenario_peak_total_mgNm3"] - optimum["limit_mgNm3"]).max()), "<=0")
    check(
        "emission_anchor_not_rescaled",
        bool((pd.read_csv(RESULTS / "condition_profiles.csv")["C_out_recorded_median"] == 50.0).all()),
        pd.read_csv(RESULTS / "condition_profiles.csv")["C_out_recorded_median"].tolist(),
        "all 50.0 mg/Nm3 (recorded, unscaled)",
    )
    scenario = json.loads((RESULTS / "scenario_parameters.json").read_text())
    check("no_outlet_rescaling_parameter", "central_outlet_scale" not in scenario,
          sorted(scenario), "no central_outlet_scale key")

    audit = pd.read_csv(RESULTS / "support_boundary_audit.csv")
    check("support_audit_rows", len(audit) == 8, len(audit), 8)
    check(
        "targets_leave_historical_experience",
        bool((audit["ratio_to_q975"] > 1.0).all()),
        float(audit["ratio_to_q975"].min()), ">1 for all eight solutions",
    )
    check(
        "10mg_inside_voltage_envelope",
        bool((audit[audit["limit_mgNm3"] == 10.0]["max_voltage_over_historical_max_pct"] <= 1e-6).all()),
        float(audit[audit["limit_mgNm3"] == 10.0]["max_voltage_over_historical_max_pct"].max()),
        "<=0% over the historical maximum",
    )
    restricted = pd.read_csv(RESULTS / "mahalanobis_constrained_feasibility.csv")
    merged = restricted.merge(
        optimum[["condition_cluster", "limit_mgNm3", "predicted_power_kW"]],
        on=["condition_cluster", "limit_mgNm3"], suffixes=("_shell", "_free"))
    feasible_shell = merged["status_with_mahalanobis_constraint"] == "SCENARIO_FEASIBLE"
    check(
        "mahalanobis_shell_variant_mostly_feasible",
        int(feasible_shell.sum()) == 7,
        int(feasible_shell.sum()), "7 of 8 combinations feasible inside the shell",
    )
    penalty = 100.0 * (merged.loc[feasible_shell, "predicted_power_kW_shell"]
                       / merged.loc[feasible_shell, "predicted_power_kW_free"] - 1.0)
    check(
        "shell_variant_costs_power",
        bool((penalty > 0).all()),
        [round(float(penalty.min()), 2), round(float(penalty.max()), 2)],
        "staying inside the shell is never cheaper than the free optimum",
    )
    headroom = pd.read_csv(RESULTS / "voltage_headroom_requirement.csv")
    check(
        "headroom_10mg_is_zero",
        bool((headroom[headroom["limit_mgNm3"] == 10.0]["min_voltage_headroom_pct"] == 0.0).all()),
        headroom[headroom["limit_mgNm3"] == 10.0]["min_voltage_headroom_pct"].tolist(), "all 0%",
    )
    check(
        "headroom_5mg_within_declared_margin",
        float(headroom[headroom["limit_mgNm3"] == 5.0]["min_voltage_headroom_pct"].max()) <= 5.0,
        float(headroom[headroom["limit_mgNm3"] == 5.0]["min_voltage_headroom_pct"].max()), "<=5%",
    )

    q4 = pd.read_csv(RESULTS / "question4_by_condition.csv")
    p10 = float(np.sum(q4["share"] * q4["power_10_kW"]) / q4["share"].sum())
    p5 = float(np.sum(q4["share"] * q4["power_5_kW"]) / q4["share"].sum())
    increase = 100 * (p5 / p10 - 1)
    summary = json.loads((RESULTS / "question4_summary.json").read_text())
    check("weighted_power_10", close(p10, summary["weighted_power_10_kW"], 1e-9), p10, summary["weighted_power_10_kW"])
    check("weighted_power_5", close(p5, summary["weighted_power_5_kW"], 1e-9), p5, summary["weighted_power_5_kW"])
    check("weighted_increase", close(increase, summary["weighted_increase_pct"], 1e-9), increase, summary["weighted_increase_pct"])
    check(
        "tightening_costs_power_vs_history",
        summary["power_10_vs_historical_pct"] > 0 and summary["weighted_increase_pct"] > 0,
        {"10mg_vs_history_pct": summary["power_10_vs_historical_pct"],
         "increase_pct": summary["weighted_increase_pct"]},
        "both positive: tightening the limit costs power",
    )
    rel = np.abs(q4["analytic_delta_power_kW"] - q4["numeric_delta_power_kW"]) / q4["numeric_delta_power_kW"]
    check(
        "analytic_matches_numeric_delta_power",
        float(rel.max()) < 0.02, float(rel.max()), "<2% relative difference",
    )

    verification = pd.read_csv(RESULTS / "optimization_verification.csv")
    check("verification_rows", len(verification) == 8, len(verification), 8)
    check(
        "multistart_agrees_on_optimum",
        float(verification["multistart_power_spread_kW"].max()) < 1.0,
        float(verification["multistart_power_spread_kW"].max()), "<1 kW across converged starts",
    )
    gaps = verification["random_search_gap_pct"].dropna()
    check(
        "random_search_finds_nothing_better",
        bool((gaps >= -1e-9).all()), float(gaps.min()), ">=0 for every combination",
    )

    sens = pd.read_csv(RESULTS / "question4_sensitivity.csv")
    check("sensitivity_grid_size", len(sens) == 27, len(sens), 27)
    p_ref = pd.read_csv(RESULTS / "p_exponent_reference.csv")
    check("p_reference_rows", len(p_ref) == 5, len(p_ref), 5)
    check(
        "scale_sensitivity_feasible_count",
        int(sens["all_conditions_feasible"].sum()) == 15,
        int(sens["all_conditions_feasible"].sum()), 15,
    )
    numeric = p_ref.dropna(subset=["numeric_increase_pct"])
    gap = (numeric["analytic_increase_pct"] - numeric["numeric_increase_pct"]).abs().max()
    check(
        "analytic_vs_numeric_p_gap",
        float(gap) < 0.5, round(float(gap), 3), "<0.5 percentage points",
    )
    check(
        "p_reference_monotone_decreasing",
        bool((p_ref.sort_values("p_exponent")["analytic_increase_pct"].diff().dropna() < 0).all()),
        p_ref.sort_values("p_exponent")["analytic_increase_pct"].round(3).tolist(),
        "increase falls as p rises (proportional to 2/p)",
    )

    structural = pd.read_csv(RESULTS / "structural_sensitivity.csv")
    structural_vals = structural.loc[structural["all_conditions_feasible"] == True, "weighted_power_increase_pct"]
    check("structural_total_count", len(structural) == 81, len(structural), 81)
    check("structural_all_feasible", len(structural_vals) == 81, len(structural_vals), 81)

    rapping = pd.read_csv(RESULTS / "rapping_power_tradeoff.csv")
    check(
        "rapping_share_reported",
        30.0 < float(rapping.iloc[0]["rapping_share_pct"]) < 45.0,
        float(rapping.iloc[0]["rapping_share_pct"]), "between 30% and 45% at median settings",
    )
    strategy = pd.read_csv(RESULTS / "historical_strategy_correlations.csv")
    check(
        "strategy_windows_reported",
        set(strategy["window"]) == {"full_record", "train", "validation", "retrospective_test"},
        sorted(set(strategy["window"])),
        "four evaluation windows",
    )
    period_rule = strategy[strategy["control"].str.startswith("T")]
    check(
        "rapping_period_rule_stable_across_windows",
        bool((period_rule["r_C_in_gNm3"] < 0).all()),
        float(period_rule["r_C_in_gNm3"].max()), "<0 in every window and field",
    )
    events = pd.read_csv(RESULTS / "anomaly_events.csv")
    check("anomaly_events_documented", len(events) == 2, len(events), 2)

    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    # A final submission may print this verifier itself in Appendix D. Restrict
    # manuscript-content checks to the paper and non-code appendices so string
    # literals inside the printed source do not become false positives.
    review_manuscript = manuscript.split("## 附录D 完整源程序", 1)[0]
    ref_body = review_manuscript.split("## 参考文献", 1)[0]
    cited = set()
    for token in re.findall(r"\[([0-9,-]+)\]", ref_body):
        for part in token.split(","):
            if "-" in part:
                a, b = map(int, part.split("-"))
                cited.update(range(a, b + 1))
            else:
                cited.add(int(part))
    listed = set(map(int, re.findall(r"^\[([0-9]+)\] ", review_manuscript.split("## 参考文献", 1)[1], flags=re.M)))
    check("all_references_cited", cited == set(range(1, 14)), sorted(cited), list(range(1, 14)))
    check("reference_list_complete", listed == set(range(1, 14)), sorted(listed), list(range(1, 14)))
    clean_manuscript = re.sub(r"<!--.*?-->\n?", "", review_manuscript)
    table6_segment = clean_manuscript.split("表11\u3000四工况两档目标", 1)[1].split("表12", 1)[0]
    table6_rows: dict[tuple[int, float], list[float]] = {}
    for line in table6_segment.splitlines():
        if not re.match(r"^\| [1-4] \| (?:10|5) \|", line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        table6_rows[(int(cells[0]), float(cells[1]))] = [float(x) for x in cells[2:]]
    table6_matches = len(table6_rows) == 8
    for key, displayed in table6_rows.items():
        expected_row = optimum[
            (optimum["condition_cluster"] == key[0]) & (optimum["limit_mgNm3"] == key[1])
        ]
        if len(expected_row) != 1:
            table6_matches = False
            continue
        row = expected_row.iloc[0]
        expected = [
            *[round(float(row[f"U{i}_kV"]), 1) for i in range(1, 5)],
            *[round(float(row[f"T{i}_s"]), 0) for i in range(1, 5)],
            round(float(row["scenario_peak_total_mgNm3"]), 3),
            round(float(row["predicted_power_kW"]), 2),
        ]
        table6_matches &= bool(np.allclose(displayed, expected, atol=5e-3))
    check(
        "table6_complete_and_matches_results",
        table6_matches,
        {f"condition_{key[0]}_target_{key[1]:g}": value for key, value in table6_rows.items()},
        "8 rows matching optimal_controls_central_scenario.csv after displayed rounding",
    )
    p_table_segment = clean_manuscript.split("表15\u3000功率增幅对幂次", 1)[1].split("\n\n", 3)[1]
    printed = {}
    for line in p_table_segment.splitlines():
        cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and re.fullmatch(r"[\d.]+", cells[0]):
            printed[float(cells[0])] = cells[2]
    expected = {float(r["p_exponent"]): f"{r['analytic_increase_pct']:.1f}%"
                for _, r in p_ref.iterrows()}
    check(
        "p_table_matches_source_at_1dp",
        printed == expected, printed, expected,
    )

    abstract = clean_manuscript.split("## 摘要", 1)[1].split("**关键词", 1)[0]
    abstract_chars = len(re.sub(r"\s+", "", abstract))
    check("abstract_fills_one_page", 700 <= abstract_chars <= 1300, abstract_chars, "700-1300 chars (one page)")
    check("anonymous_manuscript", "作者：" not in clean_manuscript and "学校：" not in clean_manuscript, ["作者：" in clean_manuscript, "学校：" in clean_manuscript], [False, False])
    check("no_english_abstract", "## Abstract" not in clean_manuscript, "## Abstract" in clean_manuscript, False)
    check("outlet_rescaling_claim_removed", "乘0.10" not in clean_manuscript and "0.1缩放" not in clean_manuscript, ["乘0.10" in clean_manuscript, "0.1缩放" in clean_manuscript], [False, False])
    check("table_caption_count", len(re.findall(r"^表\d+", clean_manuscript, flags=re.M)) == 15, len(re.findall(r"^表\d+", clean_manuscript, flags=re.M)), 15)
    check("manuscript_figure_count", len(re.findall(r"^!\[图", clean_manuscript, flags=re.M)) == 20, len(re.findall(r"^!\[图", clean_manuscript, flags=re.M)), 20)
    check("ai_disclosure_present", "AI工具使用声明" in clean_manuscript, "AI工具使用声明" in clean_manuscript, True)
    check("support_list_exists", (ROOT / "支撑材料文件清单.md").exists(), (ROOT / "支撑材料文件清单.md").exists(), True)
    ai_detail = next(ROOT.parent.glob("*/AI工具使用详情*.md"), None)
    check("ai_detail_exists", ai_detail is not None, str(ai_detail), "AI工具使用详情*.md present in the process directory")

    manifest = pd.read_csv(FIGURES / "figure_manifest.csv")
    check("figure_manifest_count", len(manifest) == 20, len(manifest), 20)
    required_trace = {"source_data", "transformation", "caption", "supported_manuscript_claims", "limitations"}
    check("figure_trace_fields", required_trace.issubset(manifest.columns), sorted(manifest.columns), sorted(required_trace))
    missing_artifacts = []
    for _, row in manifest.iterrows():
        for col in ("png", "pdf", "source_data"):
            artifact = Path(row[col])
            if not artifact.is_absolute():
                artifact = ROOT / artifact
            if not artifact.exists():
                missing_artifacts.append(f"{row['stem']}:{col}")
    check("figure_artifacts_and_sources_exist", not missing_artifacts, missing_artifacts, [])

    OUT.mkdir(parents=True, exist_ok=True)
    failed = [x for x in checks if not x["passed"]]
    report = {"status": "PASS" if not failed else "FAIL", "total_checks": len(checks), "passed": len(checks)-len(failed), "failed": len(failed), "checks": checks}
    (OUT / "deterministic_integrity_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "total_checks", "passed", "failed")}, ensure_ascii=False, indent=2))
    if failed:
        for item in failed:
            print("FAILED", item)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
