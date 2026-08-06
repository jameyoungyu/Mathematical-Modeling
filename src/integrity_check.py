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

from scenario_model import RidgePowerModel, _r2, _rmse


ROOT = SRC.parent
PROJECT_ROOT = ROOT.parent
DATA_PACKAGE = PROJECT_ROOT / "数据处理" / "水泥电除尘器_数据处理四次返修包_交付链P1闭环_排放仍阻断"
if not DATA_PACKAGE.exists():
    DATA_PACKAGE = PROJECT_ROOT.parent / "题目4" / "数据处理" / "水泥电除尘器_数据处理四次返修包_交付链P1闭环_排放仍阻断"
DATA = DATA_PACKAGE / "data" / "processed" / "full_timeseries_with_flags.csv"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures_paper"
MANUSCRIPT = ROOT / "10_修订后完整论文_核验版.md"
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
    model = RidgePowerModel(alpha=float(metrics["alpha_selected_by_blocked_rolling_validation"]))
    model.fit(train, train["P_total_kW"].to_numpy(float))
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
    baselines = pd.read_csv(RESULTS / "power_model_baselines.csv")
    val_base = baselines[baselines["split"] == "validation"].set_index("model")
    check("quadratic_beats_linear_validation", float(val_base.loc["quadratic_ridge", "rmse_kW"]) < float(val_base.loc["linear_ridge", "rmse_kW"]), val_base["rmse_kW"].to_dict(), "quadratic < linear")
    check("linear_beats_mean_validation", float(val_base.loc["linear_ridge", "rmse_kW"]) < float(val_base.loc["mean_predictor", "rmse_kW"]), val_base["rmse_kW"].to_dict(), "linear < mean")
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
    check("central_limits_respected", bool((optimum["scenario_peak_total_mgNm3"] <= optimum["limit_mgNm3"] + 1e-12).all()), float((optimum["scenario_peak_total_mgNm3"] - optimum["limit_mgNm3"]).max()), "<=0")
    check("mahalanobis_respected", bool((optimum["mahalanobis_d2"] <= optimum["support_threshold_d2"] + 1e-12).all()), float((optimum["mahalanobis_d2"]-optimum["support_threshold_d2"]).max()), "<=0")
    thresholds = optimum.groupby("condition_cluster")["support_threshold_d2"].first().round(2).tolist()
    check("empirical_support_thresholds", thresholds == [20.55, 18.72, 16.85, 19.09], thresholds, [20.55, 18.72, 16.85, 19.09])

    q4 = pd.read_csv(RESULTS / "question4_by_condition.csv")
    p10 = float(np.sum(q4["share"] * q4["power_10_kW"]) / q4["share"].sum())
    p5 = float(np.sum(q4["share"] * q4["power_5_kW"]) / q4["share"].sum())
    increase = 100 * (p5 / p10 - 1)
    summary = json.loads((RESULTS / "question4_summary.json").read_text())
    check("weighted_power_10", close(p10, summary["weighted_power_10_kW"], 1e-9), p10, summary["weighted_power_10_kW"])
    check("weighted_power_5", close(p5, summary["weighted_power_5_kW"], 1e-9), p5, summary["weighted_power_5_kW"])
    check("weighted_increase", close(increase, summary["weighted_increase_pct"], 1e-9), increase, summary["weighted_increase_pct"])

    sens = pd.read_csv(RESULTS / "question4_sensitivity.csv")
    vals = sens["weighted_power_increase_pct"].dropna()
    check("sensitivity_feasible_count", int(sens["all_conditions_feasible"].sum()) == 26, int(sens["all_conditions_feasible"].sum()), 26)
    check("sensitivity_min", round(float(vals.min()), 2) == 10.40, round(float(vals.min()), 2), 10.40)
    check("sensitivity_median", round(float(vals.median()), 2) == 13.51, round(float(vals.median()), 2), 13.51)
    check("sensitivity_max", round(float(vals.max()), 2) == 18.45, round(float(vals.max()), 2), 18.45)

    seeds = pd.read_csv(RESULTS / "optimization_seed_stability.csv")
    check("optimization_seed_count", seeds["seed"].nunique() == 5, int(seeds["seed"].nunique()), 5)
    grouped = seeds.groupby(["condition_cluster", "limit_mgNm3"])["predicted_power_kW"]
    max_cv = float((100*grouped.std()/grouped.mean()).max())
    max_range = float((100*(grouped.max()-grouped.min())/grouped.mean()).max())
    check("seed_max_cv", max_cv <= 0.181, max_cv, "<=0.181%")
    check("seed_max_range", max_range <= 0.465, max_range, "<=0.465%")
    check("local_refinement_gain", float(seeds["local_refinement_improvement_pct"].max()) <= 0.987, float(seeds["local_refinement_improvement_pct"].max()), "<=0.987%")

    structural = pd.read_csv(RESULTS / "structural_sensitivity.csv")
    structural_vals = structural.loc[structural["all_conditions_feasible"] == True, "weighted_power_increase_pct"]
    check("structural_total_count", len(structural) == 405, len(structural), 405)
    check("structural_all_feasible", len(structural_vals) == 405, len(structural_vals), 405)
    check("structural_min", round(float(structural_vals.min()), 2) == 11.37, round(float(structural_vals.min()), 2), 11.37)
    check("structural_median", round(float(structural_vals.median()), 2) == 12.99, round(float(structural_vals.median()), 2), 12.99)
    check("structural_max", round(float(structural_vals.max()), 2) == 14.31, round(float(structural_vals.max()), 2), 14.31)

    ablation = pd.read_csv(RESULTS / "field_priority_ablation_summary.csv")
    shares = ablation["front_two_share_of_positive_adjustment"]
    check("ablation_profiles", set(ablation["alpha_profile"]) == {"equal", "weak_front", "central_front"}, sorted(ablation["alpha_profile"].tolist()), ["central_front", "equal", "weak_front"])
    check("ablation_front_share_range", float(shares.min()) >= 0.707 and float(shares.max()) <= 0.738, [float(shares.min()), float(shares.max())], "[0.707,0.738]")
    dynamic = pd.read_csv(RESULTS / "dynamic_control_schedule.csv")
    check("dynamic_schedule_conditions", len(dynamic) == 4, len(dynamic), 4)
    check("dynamic_schedule_provisional", bool(dynamic["implementation_status"].str.startswith("PROVISIONAL").all()), dynamic["implementation_status"].unique().tolist(), "PROVISIONAL*")
    old_columns = {
        "strict_mode_entry_upper_bound_mgNm3",
        "strict_mode_exit_upper_bound_mgNm3",
        "minimum_dwell_min",
    }
    check(
        "old_auto_switch_thresholds_removed",
        old_columns.isdisjoint(dynamic.columns),
        sorted(dynamic.columns),
        "no legacy 4.0/4.5 threshold or dwell columns",
    )
    new_columns = {
        "mode_selection_rule",
        "available_modes",
        "automatic_emission_threshold_switching",
    }
    check(
        "external_mode_selection_schema",
        new_columns.issubset(dynamic.columns)
        and bool((dynamic["mode_selection_rule"] == "EXTERNAL_TARGET_SELECTION").all())
        and bool((dynamic["available_modes"] == "TARGET-10/TARGET-5/FALLBACK").all())
        and not bool(dynamic["automatic_emission_threshold_switching"].astype(bool).any()),
        {c: dynamic[c].unique().tolist() for c in new_columns if c in dynamic.columns},
        "external target selection; automatic threshold switching disabled",
    )

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
    table6_segment = clean_manuscript.split("表6", 1)[1].split("表7", 1)[0]
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
        table6_matches &= bool(np.allclose(displayed, expected, atol=1e-9))
    check(
        "table6_complete_and_matches_results",
        table6_matches,
        {f"condition_{key[0]}_target_{key[1]:g}": value for key, value in table6_rows.items()},
        "8 rows matching optimal_controls_central_scenario.csv after displayed rounding",
    )
    abstract = clean_manuscript.split("## 摘要", 1)[1].split("**关键词", 1)[0]
    abstract_chars = len(re.sub(r"\s+", "", abstract))
    check("abstract_under_500_chars", abstract_chars <= 500, abstract_chars, "<=500")
    check("anonymous_manuscript", "作者：" not in clean_manuscript and "学校：" not in clean_manuscript, ["作者：" in clean_manuscript, "学校：" in clean_manuscript], [False, False])
    check("no_english_abstract", "## Abstract" not in clean_manuscript, "## Abstract" in clean_manuscript, False)
    check("old_point_estimate_removed", "13.52%" not in clean_manuscript, "13.52%" in clean_manuscript, False)
    check("table_caption_count", len(re.findall(r"^表\d+", clean_manuscript, flags=re.M)) == 10, len(re.findall(r"^表\d+", clean_manuscript, flags=re.M)), 10)
    check("manuscript_figure_count", len(re.findall(r"^!\[图", clean_manuscript, flags=re.M)) == 21, len(re.findall(r"^!\[图", clean_manuscript, flags=re.M)), 21)
    check("ai_disclosure_present", "AI工具使用声明" in clean_manuscript, "AI工具使用声明" in clean_manuscript, True)
    check("support_list_exists", (ROOT / "支撑材料文件清单.md").exists(), (ROOT / "支撑材料文件清单.md").exists(), True)
    check("ai_detail_exists", (ROOT / "AI工具使用详情.md").exists(), (ROOT / "AI工具使用详情.md").exists(), True)

    manifest = pd.read_csv(FIGURES / "figure_manifest.csv")
    check("figure_manifest_count", len(manifest) == 21, len(manifest), 21)
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
