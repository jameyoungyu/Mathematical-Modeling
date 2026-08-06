#!/usr/bin/env python3
"""Reproducible scenario model for the cement ESP problem.

The power model and condition profiles are learned from the supplied data.
The emission/rapping layer is an explicitly labelled engineering scenario,
because the recorded outlet concentration is not identifiable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260805
SEEDS = [20260805, 20260806, 20260807, 20260808, 20260809]
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "full_timeseries_with_flags.csv"
CLUSTER_EVAL_FILE = ROOT / "data" / "condition_cluster_evaluation.csv"
OUT_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"

U_COLS = [f"U{i}_kV" for i in range(1, 5)]
T_COLS = [f"T{i}_s" for i in range(1, 5)]
COND_COLS = ["Temp_C", "C_in_gNm3", "Q_Nm3h"]
POWER_FIELDS = U_COLS + T_COLS + COND_COLS


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def voltage_polynomial(u: np.ndarray) -> np.ndarray:
    """Equivalent to PolynomialFeatures(degree=2, include_bias=False) for 4 voltages."""
    cols = [u[:, i] for i in range(4)]
    for i in range(4):
        for j in range(i, 4):
            cols.append(u[:, i] * u[:, j])
    return np.column_stack(cols)


class RidgePowerModel:
    def __init__(self, alpha: float = 1e-3, quadratic: bool = True):
        self.alpha = alpha
        self.quadratic = quadratic
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.beta: np.ndarray | None = None

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        u = frame[U_COLS].to_numpy(float)
        remainder = frame[T_COLS + COND_COLS].to_numpy(float)
        voltage = voltage_polynomial(u) if self.quadratic else u
        return np.column_stack([voltage, remainder])

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> None:
        x = self.design(frame)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        self.std[self.std == 0] = 1.0
        z = (x - self.mean) / self.std
        zz = np.column_stack([np.ones(len(z)), z])
        penalty = np.eye(zz.shape[1]) * math.sqrt(self.alpha)
        penalty[0, 0] = 0.0
        augmented_x = np.vstack([zz, penalty])
        augmented_y = np.concatenate([y, np.zeros(zz.shape[1])])
        self.beta = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.std is None or self.beta is None:
            raise RuntimeError("model not fitted")
        x = self.design(frame)
        z = (x - self.mean) / self.std
        zz = np.column_stack([np.ones(len(z)), z])
        return np.einsum("ij,j->i", zz, self.beta)


def condition_profiles(df: pd.DataFrame) -> pd.DataFrame:
    train = df[df["split"] == "train"].copy()
    overall_load = train["dust_load_kg_h"].median()
    labels = {
        1: "低温低浓度高流量",
        2: "高温中浓度低流量",
        3: "高温高浓度高流量",
        4: "低温高浓度低流量",
    }
    rows: list[dict[str, float | int | str]] = []
    for cluster, group in train.groupby("condition_cluster"):
        row: dict[str, float | int | str] = {
            "condition_cluster": int(cluster),
            "condition_name": labels[int(cluster)],
            "count": int(len(group)),
            "share": float(len(group) / len(train)),
            "Temp_C": float(group["Temp_C"].mean()),
            "C_in_gNm3": float(group["C_in_gNm3"].mean()),
            "Q_Nm3h": float(group["Q_Nm3h"].mean()),
            "dust_load_kg_h": float(group["dust_load_kg_h"].mean()),
            "C_out_recorded_median": float(group["C_out_mgNm3"].dropna().median()),
            "C_out_scaled_anchor": float((0.1 * group["C_out_mgNm3"].dropna()).median()),
            "historical_power_kW": float(group["P_total_kW"].mean()),
        }
        for col in U_COLS + T_COLS:
            row[f"{col}_p05"] = float(group[col].quantile(0.05))
            row[f"{col}_median"] = float(group[col].median())
            row[f"{col}_p95"] = float(group[col].quantile(0.95))
        row["load_ratio_to_train_median"] = float(row["dust_load_kg_h"] / overall_load)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition_cluster").reset_index(drop=True)


def scenario_t_star(
    profile: pd.Series,
    global_t_median: np.ndarray,
    load_exponent: float = -0.18,
) -> np.ndarray:
    load_ratio = float(profile["load_ratio_to_train_median"])
    raw = global_t_median * load_ratio ** load_exponent
    lower = np.array([profile[f"{c}_p05"] for c in T_COLS], float)
    upper = np.array([profile[f"{c}_p95"] for c in T_COLS], float)
    return np.clip(raw, lower, upper)


def scenario_emission(
    profile: pd.Series,
    u: np.ndarray,
    t: np.ndarray,
    t_star: np.ndarray,
    eta: float = 1.0,
    peak_scale: float = 1.2,
    safety_index: float = 0.08,
    outlet_scale: float = 0.10,
    alpha_weights: np.ndarray | None = None,
    peak_exponent: float = 1.35,
    phase_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return conservative base emission, rapping peak excess, and their sum.

    All non-control arguments are scenario inputs,
    not parameters estimated from the supplied outlet-concentration series.
    """
    u_ref = np.array([profile[f"{c}_median"] for c in U_COLS], float)
    t_ref = np.array([profile[f"{c}_median"] for c in T_COLS], float)
    cin = float(profile["C_in_gNm3"])
    anchor = float(profile["C_out_recorded_median"] * outlet_scale)
    k_anchor = math.log(cin * 1000.0 / anchor)

    if alpha_weights is None:
        alpha_weights = np.array([1.15, 1.10, 0.95, 0.80])
    alpha = eta * np.asarray(alpha_weights, float)
    control_delta = np.sum(alpha * ((u / u_ref) ** 2 - 1.0), axis=1)

    load_ratio = float(profile["load_ratio_to_train_median"])
    beta = np.array([0.30, 0.27, 0.20, 0.17]) * load_ratio ** 0.5

    def loss(period: np.ndarray) -> np.ndarray:
        ratio = period / t_star
        return np.sum(beta * (ratio + 1.0 / ratio - 2.0), axis=1)

    ref_loss = float(loss(t_ref.reshape(1, 4))[0])
    k = k_anchor + control_delta - (loss(t) - ref_loss)
    base = cin * 1000.0 * np.exp(-k + safety_index)

    shares = np.array([0.30, 0.30, 0.22, 0.18])
    peak = (
        peak_scale
        * phase_scale
        * load_ratio ** 0.80
        * np.sum(shares * (t / t_star) ** peak_exponent, axis=1)
    )
    return base, peak, base + peak


def support_statistics(train_group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the empirical control-support centre, inverse covariance and d2 cut-off."""
    hist = train_group[U_COLS + T_COLS].to_numpy(float)
    mu = hist.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(hist, rowvar=False))
    delta = hist - mu
    historical_d2 = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
    return mu, inv_cov, float(np.quantile(historical_d2, 0.975))


def filter_supported_controls(
    u: np.ndarray,
    t: np.ndarray,
    train_group: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu, inv_cov, threshold = support_statistics(train_group)
    controls = np.column_stack([u, t])
    delta = controls - mu
    mahal = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
    ordering = (
        (u[:, 0] >= u[:, 2] + 3.0)
        & (u[:, 1] >= u[:, 3] + 3.0)
        & (t[:, 2] >= t[:, 0] + 100.0)
        & (t[:, 3] >= t[:, 1] + 100.0)
    )
    keep = (mahal <= threshold) & ordering
    return u[keep], t[keep], mahal[keep], threshold


def candidate_bank(
    profile: pd.Series,
    model: RidgePowerModel,
    t_star: np.ndarray,
    n: int,
    rng: np.random.Generator,
    train_group: pd.DataFrame,
) -> pd.DataFrame:
    lower_u = np.array([profile[f"{c}_p05"] for c in U_COLS], float)
    upper_u = np.array([profile[f"{c}_p95"] for c in U_COLS], float)
    lower_t = np.array([profile[f"{c}_p05"] for c in T_COLS], float)
    upper_t = np.array([profile[f"{c}_p95"] for c in T_COLS], float)

    u = np.round(rng.uniform(lower_u, upper_u, size=(n, 4)), 1)
    t = np.round(rng.uniform(lower_t, upper_t, size=(n, 4)))

    # Ensure reference and locally useful candidates are always present.
    ref_u = np.array([profile[f"{c}_median"] for c in U_COLS], float)
    special_u = np.vstack([ref_u, lower_u, upper_u, 0.5 * (ref_u + lower_u), 0.5 * (ref_u + upper_u)])
    special_t = np.vstack([
        np.array([profile[f"{c}_median"] for c in T_COLS], float),
        lower_t,
        upper_t,
        t_star,
        np.clip(0.92 * t_star, lower_t, upper_t),
    ])
    u[: len(special_u)] = np.round(special_u, 1)
    t[: len(special_t)] = np.round(special_t)

    u, t, mahal, threshold = filter_supported_controls(u, t, train_group)

    frame = pd.DataFrame(np.column_stack([u, t]), columns=U_COLS + T_COLS)
    for c in COND_COLS:
        frame[c] = float(profile[c])
    frame["predicted_power_kW"] = model.predict(frame)
    frame["mahalanobis_d2"] = mahal
    frame["support_threshold_d2"] = threshold
    return frame


def local_candidate_bank(
    profile: pd.Series,
    model: RidgePowerModel,
    centre: dict[str, float | int | str],
    n: int,
    rng: np.random.Generator,
    train_group: pd.DataFrame,
) -> pd.DataFrame:
    """Generate a clipped Gaussian neighbourhood around a preliminary solution."""
    lower_u = np.array([profile[f"{c}_p05"] for c in U_COLS], float)
    upper_u = np.array([profile[f"{c}_p95"] for c in U_COLS], float)
    lower_t = np.array([profile[f"{c}_p05"] for c in T_COLS], float)
    upper_t = np.array([profile[f"{c}_p95"] for c in T_COLS], float)
    centre_u = np.array([centre[c] for c in U_COLS], float)
    centre_t = np.array([centre[c] for c in T_COLS], float)
    u = np.clip(
        rng.normal(centre_u, np.maximum(0.15, 0.025 * (upper_u - lower_u)), size=(n, 4)),
        lower_u,
        upper_u,
    )
    t = np.clip(
        rng.normal(centre_t, np.maximum(2.0, 0.025 * (upper_t - lower_t)), size=(n, 4)),
        lower_t,
        upper_t,
    )
    u = np.round(u, 1)
    t = np.round(t)
    u[0] = centre_u
    t[0] = centre_t
    u, t, mahal, threshold = filter_supported_controls(u, t, train_group)
    frame = pd.DataFrame(np.column_stack([u, t]), columns=U_COLS + T_COLS)
    for c in COND_COLS:
        frame[c] = float(profile[c])
    frame["predicted_power_kW"] = model.predict(frame)
    frame["mahalanobis_d2"] = mahal
    frame["support_threshold_d2"] = threshold
    return frame


def select_optimum(
    profile: pd.Series,
    bank: pd.DataFrame,
    t_star: np.ndarray,
    limit: float,
    eta: float = 1.0,
    peak_scale: float = 1.2,
    safety_index: float = 0.08,
    outlet_scale: float = 0.10,
    alpha_weights: np.ndarray | None = None,
    peak_exponent: float = 1.35,
    phase_scale: float = 1.0,
    validation_error_guard_kW: float = 0.0,
) -> dict[str, float | int | str]:
    u = bank[U_COLS].to_numpy(float)
    t = bank[T_COLS].to_numpy(float)
    base, peak, total = scenario_emission(
        profile,
        u,
        t,
        t_star,
        eta,
        peak_scale,
        safety_index,
        outlet_scale,
        alpha_weights,
        peak_exponent,
        phase_scale,
    )
    feasible = total <= limit
    if not np.any(feasible):
        return {
            "condition_cluster": int(profile["condition_cluster"]),
            "condition_name": str(profile["condition_name"]),
            "limit_mgNm3": float(limit),
            "status": "INFEASIBLE_IN_SCENARIO_SUPPORT",
        }
    idx_pool = np.flatnonzero(feasible)
    local = bank["predicted_power_kW"].to_numpy(float)[idx_pool]
    idx = int(idx_pool[int(np.argmin(local))])
    row: dict[str, float | int | str] = {
        "condition_cluster": int(profile["condition_cluster"]),
        "condition_name": str(profile["condition_name"]),
        "limit_mgNm3": float(limit),
        "status": "SCENARIO_FEASIBLE",
        "scenario_base_emission_mgNm3": float(base[idx]),
        "scenario_peak_excess_mgNm3": float(peak[idx]),
        "scenario_peak_total_mgNm3": float(total[idx]),
        "predicted_power_kW": float(bank.iloc[idx]["predicted_power_kW"]),
        "validation_error_adjusted_power_kW": float(
            bank.iloc[idx]["predicted_power_kW"] + validation_error_guard_kW
        ),
        "validation_error_guard_kW": float(validation_error_guard_kW),
        "historical_power_kW": float(profile["historical_power_kW"]),
        "power_change_vs_historical_pct": float(
            100.0 * (bank.iloc[idx]["predicted_power_kW"] / profile["historical_power_kW"] - 1.0)
        ),
        "mahalanobis_d2": float(bank.iloc[idx]["mahalanobis_d2"]),
        "support_threshold_d2": float(bank.iloc[idx]["support_threshold_d2"]),
    }
    for col in U_COLS + T_COLS:
        row[col] = float(bank.iloc[idx][col])
    return row


def make_figures(
    df: pd.DataFrame,
    power_model: RidgePowerModel,
    profiles: pd.DataFrame,
    optimum: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti TC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 160,
        "savefig.dpi": 240,
    })

    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]
    train = df[df["split"] == "train"]
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for (cluster, group), color in zip(train.groupby("condition_cluster"), colors):
        sample = group.iloc[::4]
        ax.scatter(sample["C_in_gNm3"], sample["Temp_C"], s=7, alpha=0.35, color=color,
                   label=f"工况{int(cluster)}")
    ax.set_xlabel("入口粉尘浓度 (g/Nm³)")
    ax.set_ylabel("入口温度 (°C)")
    ax.set_title("训练期典型工况划分")
    ax.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_condition_clusters.png")
    plt.close(fig)

    test = df[df["split"] == "test"]
    pred = power_model.predict(test)
    actual = test["P_total_kW"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(actual, pred, s=9, alpha=0.45, color="#0072B2")
    lo = min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#D55E00", linewidth=1.3)
    ax.set_xlabel("实际总功率 (kW)")
    ax.set_ylabel("预测总功率 (kW)")
    ax.set_title("功率模型回顾性测试")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_power_test.png")
    plt.close(fig)

    pivot = optimum.pivot(index="condition_name", columns="limit_mgNm3", values="predicted_power_kW")
    pivot = pivot[[10.0, 5.0]]
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    x = np.arange(len(pivot))
    width = 0.34
    ax.bar(x - width / 2, pivot[10.0], width, label="10 mg/Nm³", color="#56B4E9")
    ax.bar(x + width / 2, pivot[5.0], width, label="5 mg/Nm³", color="#D55E00")
    ax.set_xticks(x, pivot.index, rotation=12)
    ax.set_ylabel("情景最优功率 (kW)")
    ax.set_title("不同排放约束下的情景最优功率")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_optimal_power_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    ratio = np.linspace(0.75, 1.25, 200)
    for profile, color in zip((profiles.iloc[0], profiles.iloc[2]), (colors[0], colors[2])):
        load_ratio = float(profile["load_ratio_to_train_median"])
        peak = 1.2 * load_ratio ** 0.8 * ratio ** 1.35
        ax.plot(100 * (ratio - 1), peak, color=color, linewidth=2,
                label=str(profile["condition_name"]))
    ax.set_xlabel("振打周期相对最优周期的变化 (%)")
    ax.set_ylabel("单次振打峰值增量代理 (mg/Nm³)")
    ax.set_title("振打周期延长会放大单次再飞扬峰值")
    ax.axvline(0, color="#666666", linewidth=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_rapping_peak_scenario.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    vals = sensitivity["weighted_power_increase_pct"].dropna().sort_values().to_numpy()
    ax.hist(vals, bins=12, color="#009E73", alpha=0.85, edgecolor="white")
    ax.axvline(np.median(vals), color="#D55E00", linestyle="--", linewidth=1.5,
               label=f"中位数 {np.median(vals):.1f}%")
    ax.set_xlabel("5 mg/Nm³ 相对 10 mg/Nm³ 的加权功率增幅 (%)")
    ax.set_ylabel("情景组合数")
    ax.set_title("先验参数敏感性分析")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_sensitivity_distribution.png")
    plt.close(fig)


def evaluate_power_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[RidgePowerModel, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Select ridge strength by blocked rolling validation and retain honest baselines."""
    alpha_grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    n = len(train)
    rolling_bounds = ((0, int(0.40*n), int(0.60*n)), (0, int(0.60*n), int(0.80*n)), (0, int(0.80*n), n))
    rolling_rows: list[dict[str, object]] = []
    selected: dict[str, float] = {}
    for model_name, quadratic in (("linear_ridge", False), ("quadratic_ridge", True)):
        alpha_scores: list[tuple[float, float]] = []
        for alpha in alpha_grid:
            fold_rmses = []
            for fold, (start, fit_end, eval_end) in enumerate(rolling_bounds, start=1):
                fit = train.iloc[start:fit_end]
                evaluate = train.iloc[fit_end:eval_end]
                candidate = RidgePowerModel(alpha=alpha, quadratic=quadratic)
                candidate.fit(fit, fit["P_total_kW"].to_numpy(float))
                pred = candidate.predict(evaluate)
                actual = evaluate["P_total_kW"].to_numpy(float)
                rmse = _rmse(actual, pred)
                fold_rmses.append(rmse)
                rolling_rows.append({
                    "model": model_name,
                    "alpha": alpha,
                    "fold": fold,
                    "fit_rows": len(fit),
                    "evaluation_rows": len(evaluate),
                    "r2": _r2(actual, pred),
                    "rmse_kW": rmse,
                    "bias_kW": float(np.mean(pred-actual)),
                })
            alpha_scores.append((float(np.mean(fold_rmses)), alpha))
        selected[model_name] = min(alpha_scores)[1]

    baseline_rows: list[dict[str, object]] = []
    mean_power = float(train["P_total_kW"].mean())
    fitted_models: dict[str, RidgePowerModel] = {}
    for model_name, quadratic in (("linear_ridge", False), ("quadratic_ridge", True)):
        fitted = RidgePowerModel(alpha=selected[model_name], quadratic=quadratic)
        fitted.fit(train, train["P_total_kW"].to_numpy(float))
        fitted_models[model_name] = fitted
    for split_name, frame in (("validation", validation), ("retrospective_test", test)):
        actual = frame["P_total_kW"].to_numpy(float)
        for model_name in ("mean_predictor", "linear_ridge", "quadratic_ridge"):
            pred = np.full(len(frame), mean_power) if model_name == "mean_predictor" else fitted_models[model_name].predict(frame)
            baseline_rows.append({
                "model": model_name,
                "selected_alpha": np.nan if model_name == "mean_predictor" else selected[model_name],
                "split": split_name,
                "r2": _r2(actual, pred),
                "rmse_kW": _rmse(actual, pred),
                "bias_kW": float(np.mean(pred-actual)),
            })
    chosen = fitted_models["quadratic_ridge"]
    val_actual = validation["P_total_kW"].to_numpy(float)
    val_pred = chosen.predict(validation)
    test_actual = test["P_total_kW"].to_numpy(float)
    test_pred = chosen.predict(test)
    guard = float(np.quantile(np.abs(val_pred-val_actual), 0.90))
    metrics: dict[str, object] = {
        "model": "standardized_quadratic_ridge_voltage_plus_period_and_condition",
        "fit_scope": "train_only",
        "alpha_selected_by_blocked_rolling_validation": selected["quadratic_ridge"],
        "validation_r2": _r2(val_actual, val_pred),
        "validation_rmse_kW": _rmse(val_actual, val_pred),
        "validation_bias_kW": float(np.mean(val_pred-val_actual)),
        "validation_abs_error_q90_kW": guard,
        "retrospective_test_r2": _r2(test_actual, test_pred),
        "retrospective_test_rmse_kW": _rmse(test_actual, test_pred),
        "retrospective_test_bias_kW": float(np.mean(test_pred-test_actual)),
        "test_protocol": "retrospective_not_blind_due_to_prior_package_use",
        "reporting_guard_rule": "point_prediction_plus_validation_absolute_error_q90",
        "selected_alphas": selected,
    }
    return chosen, metrics, pd.DataFrame(rolling_rows), pd.DataFrame(baseline_rows)


def aggregate_pair(rows10: list[dict[str, object]], rows5: list[dict[str, object]]) -> dict[str, float | bool]:
    if len(rows10) != 4 or len(rows5) != 4:
        return {"all_conditions_feasible": False, "weighted_power_10_kW": np.nan, "weighted_power_5_kW": np.nan, "weighted_power_increase_pct": np.nan}
    weights = np.array([float(r["share"]) for r in rows10])
    p10 = float(np.average([float(r["predicted_power_kW"]) for r in rows10], weights=weights))
    p5 = float(np.average([float(r["predicted_power_kW"]) for r in rows5], weights=weights))
    return {"all_conditions_feasible": True, "weighted_power_10_kW": p10, "weighted_power_5_kW": p5, "weighted_power_increase_pct": 100.0*(p5/p10-1.0)}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df["condition_cluster"] = df["condition_cluster"].astype(int)
    profiles = condition_profiles(df)
    profiles.to_csv(OUT_DIR / "condition_profiles.csv", index=False, encoding="utf-8-sig")

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    model, metrics, rolling_metrics, baseline_metrics = evaluate_power_models(train, validation, test)
    rolling_metrics.to_csv(OUT_DIR / "power_model_rolling_validation.csv", index=False, encoding="utf-8-sig")
    baseline_metrics.to_csv(OUT_DIR / "power_model_baselines.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "power_model_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_error_guard = float(metrics["validation_abs_error_q90_kW"])

    cluster_eval = pd.read_csv(CLUSTER_EVAL_FILE)
    cluster_eval.to_csv(OUT_DIR / "cluster_selection_metrics.csv", index=False, encoding="utf-8-sig")
    global_t_median = train[T_COLS].median().to_numpy(float)
    t_stars = {int(p["condition_cluster"]): scenario_t_star(p, global_t_median) for _, p in profiles.iterrows()}

    banks: dict[int, pd.DataFrame] = {}
    seed_rows: list[dict[str, object]] = []
    convergence_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        for _, profile in profiles.iterrows():
            cluster = int(profile["condition_cluster"])
            group = train[train["condition_cluster"] == cluster]
            t_star = t_stars[cluster]
            base_bank = candidate_bank(profile, model, t_star, 220_000, rng, group)
            preliminary = {
                limit: select_optimum(
                    profile,
                    base_bank,
                    t_star,
                    limit,
                    validation_error_guard_kW=validation_error_guard,
                )
                for limit in (10.0, 5.0)
            }
            additions = []
            for limit in (10.0, 5.0):
                if "predicted_power_kW" in preliminary[limit]:
                    additions.append(local_candidate_bank(profile, model, preliminary[limit], 30_000, rng, group))
            bank = pd.concat([base_bank] + additions, ignore_index=True).drop_duplicates(U_COLS+T_COLS, keep="first")
            if seed == SEED:
                banks[cluster] = bank
                for limit in (10.0, 5.0):
                    for fraction in (0.25, 0.50, 0.75, 1.00):
                        partial = base_bank.iloc[:max(1000, int(len(base_bank)*fraction))]
                        result = select_optimum(
                            profile,
                            partial,
                            t_star,
                            limit,
                            validation_error_guard_kW=validation_error_guard,
                        )
                        convergence_rows.append({
                            "condition_cluster": cluster,
                            "condition_name": profile["condition_name"],
                            "limit_mgNm3": limit,
                            "candidate_bank_fraction": fraction,
                            "supported_candidates": len(partial),
                            "status": result["status"],
                            "best_power_kW": result.get("predicted_power_kW", np.nan),
                        })
            for limit in (10.0, 5.0):
                final = select_optimum(
                    profile,
                    bank,
                    t_star,
                    limit,
                    validation_error_guard_kW=validation_error_guard,
                )
                final.update({
                    "seed": seed,
                    "base_supported_candidates": len(base_bank),
                    "refined_supported_candidates": len(bank),
                    "preliminary_power_kW": preliminary[limit].get("predicted_power_kW", np.nan),
                    "local_refinement_improvement_pct": 100.0*(float(preliminary[limit].get("predicted_power_kW", np.nan))/float(final.get("predicted_power_kW", np.nan))-1.0),
                })
                seed_rows.append(final)

    seed_stability = pd.DataFrame(seed_rows)
    seed_stability.to_csv(OUT_DIR / "optimization_seed_stability.csv", index=False, encoding="utf-8-sig")
    central = seed_stability[seed_stability["seed"] == SEED].copy()
    optimum = central.drop(columns=["seed", "base_supported_candidates", "refined_supported_candidates", "preliminary_power_kW", "local_refinement_improvement_pct"])
    optimum.to_csv(OUT_DIR / "optimal_controls_central_scenario.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(convergence_rows).to_csv(OUT_DIR / "optimization_convergence.csv", index=False, encoding="utf-8-sig")

    q4_rows: list[dict[str, object]] = []
    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        a = optimum[(optimum["condition_cluster"] == cluster) & (optimum["limit_mgNm3"] == 10.0)].iloc[0]
        b = optimum[(optimum["condition_cluster"] == cluster) & (optimum["limit_mgNm3"] == 5.0)].iloc[0]
        q4_rows.append({
            "condition_cluster": cluster,
            "condition_name": profile["condition_name"],
            "share": profile["share"],
            "power_10_kW": a["predicted_power_kW"],
            "power_5_kW": b["predicted_power_kW"],
            "validation_error_adjusted_power_10_kW": a["validation_error_adjusted_power_kW"],
            "validation_error_adjusted_power_5_kW": b["validation_error_adjusted_power_kW"],
            "increase_pct": 100.0*(b["predicted_power_kW"]/a["predicted_power_kW"]-1.0),
        })
    q4 = pd.DataFrame(q4_rows)
    weighted_p10 = float(np.average(q4["power_10_kW"], weights=q4["share"]))
    weighted_p5 = float(np.average(q4["power_5_kW"], weights=q4["share"]))
    q4_summary = {
        "weighted_power_10_kW": weighted_p10,
        "weighted_power_5_kW": weighted_p5,
        "weighted_increase_pct": 100.0*(weighted_p5/weighted_p10-1.0),
        "validation_error_guard_kW": validation_error_guard,
        "weighted_validation_error_adjusted_power_10_kW": weighted_p10 + validation_error_guard,
        "weighted_validation_error_adjusted_power_5_kW": weighted_p5 + validation_error_guard,
    }
    q4.to_csv(OUT_DIR / "question4_by_condition.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "question4_summary.json").write_text(json.dumps(q4_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    sensitivity_rows: list[dict[str, object]] = []
    for eta in (0.8, 1.0, 1.2):
        for peak_scale in (0.9, 1.2, 1.5):
            for safety in (0.0, 0.08, 0.15):
                r10s: list[dict[str, object]] = []
                r5s: list[dict[str, object]] = []
                for _, profile in profiles.iterrows():
                    cluster = int(profile["condition_cluster"])
                    r10 = select_optimum(profile, banks[cluster], t_stars[cluster], 10.0, eta=eta, peak_scale=peak_scale, safety_index=safety, validation_error_guard_kW=validation_error_guard)
                    r5 = select_optimum(profile, banks[cluster], t_stars[cluster], 5.0, eta=eta, peak_scale=peak_scale, safety_index=safety, validation_error_guard_kW=validation_error_guard)
                    if "predicted_power_kW" in r10 and "predicted_power_kW" in r5:
                        r10["share"] = profile["share"]
                        r5["share"] = profile["share"]
                        r10s.append(r10); r5s.append(r5)
                sensitivity_rows.append({"eta_voltage_effect": eta, "peak_scale": peak_scale, "safety_index": safety, **aggregate_pair(r10s, r5s)})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT_DIR / "question4_sensitivity.csv", index=False, encoding="utf-8-sig")

    alpha_profiles = {
        "equal": np.array([1.00, 1.00, 1.00, 1.00]),
        "weak_front": np.array([1.08, 1.04, 0.98, 0.90]),
        "central_front": np.array([1.15, 1.10, 0.95, 0.80]),
    }
    structural_rows: list[dict[str, object]] = []
    for outlet_scale in (0.08, 0.09, 0.10, 0.11, 0.12):
        for alpha_name, alpha_weights in alpha_profiles.items():
            for peak_exponent in (1.10, 1.35, 1.60):
                for phase_scale in (0.65, 0.80, 1.00):
                    for tstar_exponent in (-0.30, -0.18, -0.10):
                        r10s = []; r5s = []
                        for _, profile in profiles.iterrows():
                            cluster = int(profile["condition_cluster"])
                            t_star = scenario_t_star(profile, global_t_median, tstar_exponent)
                            args = dict(outlet_scale=outlet_scale, alpha_weights=alpha_weights, peak_exponent=peak_exponent, phase_scale=phase_scale, validation_error_guard_kW=validation_error_guard)
                            r10 = select_optimum(profile, banks[cluster], t_star, 10.0, **args)
                            r5 = select_optimum(profile, banks[cluster], t_star, 5.0, **args)
                            if "predicted_power_kW" in r10 and "predicted_power_kW" in r5:
                                r10["share"] = profile["share"]; r5["share"] = profile["share"]
                                r10s.append(r10); r5s.append(r5)
                        structural_rows.append({
                            "outlet_scale": outlet_scale,
                            "alpha_profile": alpha_name,
                            "peak_exponent": peak_exponent,
                            "phase_scale": phase_scale,
                            "tstar_load_exponent": tstar_exponent,
                            **aggregate_pair(r10s, r5s),
                        })
    structural = pd.DataFrame(structural_rows)
    structural.to_csv(OUT_DIR / "structural_sensitivity.csv", index=False, encoding="utf-8-sig")

    ablation_rows: list[dict[str, object]] = []
    ablation_summary_rows: list[dict[str, object]] = []
    for alpha_name, alpha_weights in alpha_profiles.items():
        per_profile = []
        for _, profile in profiles.iterrows():
            cluster = int(profile["condition_cluster"])
            args = dict(alpha_weights=alpha_weights, validation_error_guard_kW=validation_error_guard)
            r10 = select_optimum(profile, banks[cluster], t_stars[cluster], 10.0, **args)
            r5 = select_optimum(profile, banks[cluster], t_stars[cluster], 5.0, **args)
            row = {"alpha_profile": alpha_name, "condition_cluster": cluster, "condition_name": profile["condition_name"], "share": profile["share"]}
            for i, col in enumerate(U_COLS, start=1):
                row[f"delta_U{i}_kV"] = float(r5[col])-float(r10[col])
            per_profile.append(row); ablation_rows.append(row)
        weights = np.array([float(x["share"]) for x in per_profile])
        deltas = np.array([[float(x[f"delta_U{i}_kV"]) for i in range(1,5)] for x in per_profile])
        weighted = np.average(deltas, axis=0, weights=weights)
        positive = np.maximum(weighted, 0)
        ablation_summary_rows.append({
            "alpha_profile": alpha_name,
            **{f"weighted_delta_U{i}_kV": weighted[i-1] for i in range(1,5)},
            "front_two_share_of_positive_adjustment": float(positive[:2].sum()/positive.sum()) if positive.sum() else np.nan,
        })
    pd.DataFrame(ablation_rows).to_csv(OUT_DIR / "field_priority_ablation_by_condition.csv", index=False, encoding="utf-8-sig")
    ablation_summary = pd.DataFrame(ablation_summary_rows)
    ablation_summary.to_csv(OUT_DIR / "field_priority_ablation_summary.csv", index=False, encoding="utf-8-sig")

    historical_rows: list[dict[str, object]] = []
    dynamic_rows: list[dict[str, object]] = []
    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        frame = pd.DataFrame([{**{c: profile[f"{c}_median"] for c in U_COLS+T_COLS}, **{c: profile[c] for c in COND_COLS}}])
        median_power = float(model.predict(frame)[0])
        r10 = optimum[(optimum["condition_cluster"] == cluster) & (optimum["limit_mgNm3"] == 10.0)].iloc[0]
        r5 = optimum[(optimum["condition_cluster"] == cluster) & (optimum["limit_mgNm3"] == 5.0)].iloc[0]
        historical_rows.append({
            "condition_cluster": cluster,
            "condition_name": profile["condition_name"],
            "share": profile["share"],
            "historical_mean_actual_power_kW": profile["historical_power_kW"],
            "model_power_at_historical_median_controls_kW": median_power,
            "scenario_optimal_10_power_kW": r10["predicted_power_kW"],
            "auxiliary_saving_vs_model_median_pct": 100.0*(median_power/float(r10["predicted_power_kW"])-1.0),
        })
        dynamic_rows.append({
            "condition_cluster": cluster,
            "condition_name": profile["condition_name"],
            "mode_selection_rule": "EXTERNAL_TARGET_SELECTION",
            "available_modes": "TARGET-10/TARGET-5/FALLBACK",
            "automatic_emission_threshold_switching": False,
            "provisional_voltage_ramp_kV_per_min": 0.5,
            "provisional_period_ramp_s_per_min": 5.0,
            "phase_scenario_for_online_trial": "0.65/0.80/1.00",
            "implementation_status": "PROVISIONAL_MUST_BE_OVERRIDDEN_BY_SITE_DCS_AND_INTERLOCK_LIMITS",
            **{f"target10_{c}": r10[c] for c in U_COLS+T_COLS},
            **{f"target5_{c}": r5[c] for c in U_COLS+T_COLS},
        })
    pd.DataFrame(historical_rows).to_csv(OUT_DIR / "historical_median_power_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dynamic_rows).to_csv(OUT_DIR / "dynamic_control_schedule.csv", index=False, encoding="utf-8-sig")

    feasible_structural = structural[structural["all_conditions_feasible"] == True]
    scenario_params = {
        "status": "SCENARIO_ASSUMPTIONS_NOT_IDENTIFIED_FROM_C_OUT",
        "central_outlet_scale": 0.10,
        "structural_outlet_scales": [0.08, 0.09, 0.10, 0.11, 0.12],
        "voltage_weight_profiles": {k: v.tolist() for k, v in alpha_profiles.items()},
        "peak_exponents": [1.10, 1.35, 1.60],
        "rapping_phase_scales": [0.65, 0.80, 1.00],
        "tstar_load_exponents": [-0.30, -0.18, -0.10],
        "rapping_loss_coefficients": [0.30, 0.27, 0.20, 0.17],
        "peak_field_shares": [0.30, 0.30, 0.22, 0.18],
        "support_rule": "cluster-specific P05-P95 plus empirical 97.5th percentile Mahalanobis d2 and engineering ordering",
        "base_candidate_count_per_condition_seed": 220000,
        "local_candidate_count_per_preliminary_solution": 30000,
        "random_seeds": SEEDS,
    }
    (OUT_DIR / "scenario_parameters.json").write_text(json.dumps(scenario_params, ensure_ascii=False, indent=2), encoding="utf-8")

    make_figures(df, model, profiles, optimum, sensitivity)
    seed_grouped = seed_stability.groupby(["condition_cluster", "limit_mgNm3"])["predicted_power_kW"]
    summary = {
        "power_model": metrics,
        "cluster_k": 4,
        "central_q4": q4_summary,
        "prior_parameter_sensitivity": {
            "increase_pct_min": float(sensitivity["weighted_power_increase_pct"].min()),
            "increase_pct_median": float(sensitivity["weighted_power_increase_pct"].median()),
            "increase_pct_max": float(sensitivity["weighted_power_increase_pct"].max()),
            "feasible_combinations": int(sensitivity["all_conditions_feasible"].sum()),
            "total_combinations": len(sensitivity),
        },
        "structural_sensitivity": {
            "feasible_combinations": len(feasible_structural),
            "total_combinations": len(structural),
            "increase_pct_min": float(feasible_structural["weighted_power_increase_pct"].min()),
            "increase_pct_median": float(feasible_structural["weighted_power_increase_pct"].median()),
            "increase_pct_max": float(feasible_structural["weighted_power_increase_pct"].max()),
        },
        "seed_stability_max_cv_pct": float((100.0*seed_grouped.std()/seed_grouped.mean()).max()),
        "seed_stability_max_range_pct_of_mean": float((100.0*(seed_grouped.max()-seed_grouped.min())/seed_grouped.mean()).max()),
        "local_refinement_max_improvement_pct": float(seed_stability["local_refinement_improvement_pct"].max()),
        "central_all_conditions_feasible": bool((optimum["status"] == "SCENARIO_FEASIBLE").all()),
    }
    (OUT_DIR / "model_run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
