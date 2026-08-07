#!/usr/bin/env python3
"""Reproducible scenario model for the cement ESP problem.

Two layers are kept strictly separate:

* the **power layer** is identified from the supplied data.  Total power is a
  near-deterministic function of the control settings alone, with the physically
  motivated form ``P = a0 + sum_i b_i U_i^2 + sum_i c_i / T_i`` (field power
  proportional to voltage squared, rapping power proportional to rapping
  frequency ``60/T_i``);
* the **emission layer** is an explicitly labelled engineering scenario,
  because the recorded outlet concentration is capped near 50 mg/Nm^3 and
  carries no learnable signal.

The historical emission anchor is the *recorded* outlet concentration; no
decimal-point rescaling is applied.  The control problem solved here is
therefore the path from the historical ~50 mg/Nm^3 level down to the 10 and
5 mg/Nm^3 targets set by the problem statement.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize


SEED = 20260805
ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "full_timeseries_with_flags.csv"
CLUSTER_EVAL_FILE = ROOT / "data" / "condition_cluster_evaluation.csv"
OUT_DIR = ROOT / "results"

U_COLS = [f"U{i}_kV" for i in range(1, 5)]
T_COLS = [f"T{i}_s" for i in range(1, 5)]
COND_COLS = ["Temp_C", "C_in_gNm3", "Q_Nm3h"]

# ---- central scenario constants (engineering priors, not fitted to C_out) ----
ALPHA_CENTRAL = np.array([1.15, 1.10, 0.95, 0.80])   # per-field removal weights
BETA_CENTRAL = np.array([0.30, 0.27, 0.20, 0.17])    # rapping-period deviation loss
PEAK_SHARES = np.array([0.30, 0.30, 0.22, 0.18])     # per-field share of the peak proxy
P_EXPONENT_CENTRAL = 2.0     # K ~ U^p ; p = 2 is the Deutsch drift-velocity form
PEAK_EXPONENT_CENTRAL = 1.0  # derived: cake mass per rap ~ L*T  =>  peak ~ T^1
PEAK_SCALE_CENTRAL = 1.2
PHASE_SCALE_CENTRAL = 1.0
SAFETY_INDEX_CENTRAL = 0.08
# 5 mg/Nm^3 needs voltage above every field's historical maximum; the margin is
# an explicitly declared extrapolation of the historical envelope, not a fact.
HEADROOM_10 = 0.00
HEADROOM_5 = 0.05

MULTISTART = 48
SWEEP_MULTISTART = 16   # sensitivity grids: fewer starts, cross-checked against the central solve


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


# --------------------------------------------------------------- power models

def voltage_polynomial(u: np.ndarray) -> np.ndarray:
    """Equivalent to PolynomialFeatures(degree=2, include_bias=False) for 4 voltages."""
    cols = [u[:, i] for i in range(4)]
    for i in range(4):
        for j in range(i, 4):
            cols.append(u[:, i] * u[:, j])
    return np.column_stack(cols)


class _ScaledOLS:
    """Least squares with per-column scaling.

    The scaling matters: ``1/T`` is of order 4e-3 while ``U^2`` is of order
    3e3, and an unscaled ``lstsq`` truncates the rapping columns as though they
    were rank-deficient, which makes the rapping term look irrelevant.
    """

    def __init__(self) -> None:
        self.beta: np.ndarray | None = None

    def _fit_design(self, x: np.ndarray, y: np.ndarray) -> None:
        xx = np.column_stack([np.ones(len(x)), x])
        scale = np.abs(xx).max(axis=0)
        scale[scale == 0] = 1.0
        beta, *_ = np.linalg.lstsq(xx / scale, y, rcond=None)
        self.beta = beta / scale

    def _predict_design(self, x: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("model not fitted")
        return self.beta[0] + x @ self.beta[1:]


class PhysicalPowerModel(_ScaledOLS):
    """P = a0 + sum_i b_i U_i^2 + sum_i c_i / T_i  (optionally + condition terms)."""

    name = "physical_voltage_square_plus_rapping_frequency"

    def __init__(self, with_conditions: bool = False) -> None:
        super().__init__()
        self.with_conditions = with_conditions

    def design(self, frame: pd.DataFrame) -> np.ndarray:
        u = frame[U_COLS].to_numpy(float)
        t = frame[T_COLS].to_numpy(float)
        blocks = [u ** 2, 1.0 / t]
        if self.with_conditions:
            blocks.append(frame[COND_COLS].to_numpy(float))
        return np.column_stack(blocks)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "PhysicalPowerModel":
        self._fit_design(self.design(frame), y)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self._predict_design(self.design(frame))

    # -- fast path used inside the optimiser -------------------------------
    @property
    def intercept(self) -> float:
        return float(self.beta[0])

    @property
    def b(self) -> np.ndarray:
        return np.asarray(self.beta[1:5], float)

    @property
    def c(self) -> np.ndarray:
        return np.asarray(self.beta[5:9], float)

    def power(self, u: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Total power for raw control arrays (last axis = 4 fields)."""
        return self.intercept + np.sum(self.b * u ** 2, axis=-1) + np.sum(self.c / t, axis=-1)

    def field_power(self, u: np.ndarray) -> np.ndarray:
        return np.sum(self.b * u ** 2, axis=-1)

    def rapping_power(self, t: np.ndarray) -> np.ndarray:
        return np.sum(self.c / t, axis=-1)


class RidgePowerModel:
    """Standardised ridge baseline with the rapping period entered *linearly*.

    Retained as the misspecified reference point: it is the specification the
    first version of this paper used, and the comparison in
    ``power_model_baselines.csv`` shows what treating ``T`` as a linear term
    costs on the retrospective test window.
    """

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


def evaluate_power_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[PhysicalPowerModel, dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Fit every candidate specification on the training window only."""
    alpha_grid = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
    n = len(train)
    rolling_bounds = ((0, int(0.40 * n), int(0.60 * n)),
                      (0, int(0.60 * n), int(0.80 * n)),
                      (0, int(0.80 * n), n))
    rolling_rows: list[dict[str, object]] = []
    selected: dict[str, float] = {}
    for model_name, quadratic in (("linear_ridge_T", False), ("quadratic_ridge_T", True)):
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
                    "model": model_name, "alpha": alpha, "fold": fold,
                    "fit_rows": len(fit), "evaluation_rows": len(evaluate),
                    "r2": _r2(actual, pred), "rmse_kW": rmse,
                    "bias_kW": float(np.mean(pred - actual)),
                })
            alpha_scores.append((float(np.mean(fold_rmses)), alpha))
        selected[model_name] = min(alpha_scores)[1]

    # the physical specification has no tuning parameter; still report it per fold
    for fold, (start, fit_end, eval_end) in enumerate(rolling_bounds, start=1):
        fit = train.iloc[start:fit_end]
        evaluate = train.iloc[fit_end:eval_end]
        candidate = PhysicalPowerModel().fit(fit, fit["P_total_kW"].to_numpy(float))
        pred = candidate.predict(evaluate)
        actual = evaluate["P_total_kW"].to_numpy(float)
        rolling_rows.append({
            "model": "physical_invT", "alpha": np.nan, "fold": fold,
            "fit_rows": len(fit), "evaluation_rows": len(evaluate),
            "r2": _r2(actual, pred), "rmse_kW": _rmse(actual, pred),
            "bias_kW": float(np.mean(pred - actual)),
        })

    y_train = train["P_total_kW"].to_numpy(float)
    fitted: dict[str, object] = {}
    for model_name, quadratic in (("linear_ridge_T", False), ("quadratic_ridge_T", True)):
        m = RidgePowerModel(alpha=selected[model_name], quadratic=quadratic)
        m.fit(train, y_train)
        fitted[model_name] = m
    fitted["physical_invT"] = PhysicalPowerModel().fit(train, y_train)
    fitted["physical_invT_plus_conditions"] = PhysicalPowerModel(with_conditions=True).fit(train, y_train)

    mean_power = float(y_train.mean())
    baseline_rows: list[dict[str, object]] = []
    for split_name, frame in (("train", train), ("validation", validation), ("retrospective_test", test)):
        actual = frame["P_total_kW"].to_numpy(float)
        for model_name in ("mean_predictor", "linear_ridge_T", "quadratic_ridge_T",
                           "physical_invT", "physical_invT_plus_conditions"):
            pred = (np.full(len(frame), mean_power) if model_name == "mean_predictor"
                    else fitted[model_name].predict(frame))
            baseline_rows.append({
                "model": model_name,
                "selected_alpha": selected.get(model_name, np.nan),
                "split": split_name,
                "r2": _r2(actual, pred),
                "rmse_kW": _rmse(actual, pred),
                "bias_kW": float(np.mean(pred - actual)),
            })

    chosen: PhysicalPowerModel = fitted["physical_invT"]  # type: ignore[assignment]
    val_actual = validation["P_total_kW"].to_numpy(float)
    val_pred = chosen.predict(validation)
    test_actual = test["P_total_kW"].to_numpy(float)
    test_pred = chosen.predict(test)
    guard = float(np.quantile(np.abs(val_pred - val_actual), 0.90))
    metrics: dict[str, object] = {
        "model": chosen.name,
        "formula": "P = a0 + sum_i b_i U_i^2 + sum_i c_i / T_i",
        "fit_scope": "train_only",
        "intercept_kW": chosen.intercept,
        "b_field_kW_per_kV2": chosen.b.tolist(),
        "c_rapping_kW_s": chosen.c.tolist(),
        "train_r2": _r2(y_train, chosen.predict(train)),
        "validation_r2": _r2(val_actual, val_pred),
        "validation_rmse_kW": _rmse(val_actual, val_pred),
        "validation_bias_kW": float(np.mean(val_pred - val_actual)),
        "validation_abs_error_q90_kW": guard,
        "retrospective_test_r2": _r2(test_actual, test_pred),
        "retrospective_test_rmse_kW": _rmse(test_actual, test_pred),
        "retrospective_test_bias_kW": float(np.mean(test_pred - test_actual)),
        "test_protocol": "retrospective_not_blind_due_to_prior_package_use",
        "reporting_guard_rule": "point_prediction_plus_validation_absolute_error_q90",
        "ridge_selected_alphas": selected,
    }
    return chosen, metrics, pd.DataFrame(rolling_rows), pd.DataFrame(baseline_rows)


# ------------------------------------------------------------ condition layer

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
            "historical_power_kW": float(group["P_total_kW"].mean()),
        }
        for col in U_COLS + T_COLS:
            row[f"{col}_min"] = float(group[col].min())
            row[f"{col}_p05"] = float(group[col].quantile(0.05))
            row[f"{col}_median"] = float(group[col].median())
            row[f"{col}_p95"] = float(group[col].quantile(0.95))
            row[f"{col}_max"] = float(group[col].max())
        row["load_ratio_to_train_median"] = float(row["dust_load_kg_h"] / overall_load)
        row["K0_anchor"] = math.log(1000.0 * row["C_in_gNm3"] / row["C_out_recorded_median"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition_cluster").reset_index(drop=True)


def scenario_t_star(profile: pd.Series, global_t_median: np.ndarray,
                    load_exponent: float = -0.18) -> np.ndarray:
    load_ratio = float(profile["load_ratio_to_train_median"])
    raw = global_t_median * load_ratio ** load_exponent
    lower = np.array([profile[f"{c}_min"] for c in T_COLS], float)
    upper = np.array([profile[f"{c}_max"] for c in T_COLS], float)
    return np.clip(raw, lower, upper)


def support_statistics(train_group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Empirical control-support centre, inverse covariance and d^2 cut-off."""
    hist = train_group[U_COLS + T_COLS].to_numpy(float)
    mu = hist.mean(axis=0)
    inv_cov = np.linalg.pinv(np.cov(hist, rowvar=False))
    delta = hist - mu
    historical_d2 = np.einsum("ij,jk,ik->i", delta, inv_cov, delta)
    return mu, inv_cov, float(np.quantile(historical_d2, 0.975))


# ------------------------------------------------------------- emission layer

def scenario_emission(
    profile: pd.Series,
    u: np.ndarray,
    t: np.ndarray,
    t_star: np.ndarray,
    p_exponent: float = P_EXPONENT_CENTRAL,
    peak_scale: float = PEAK_SCALE_CENTRAL,
    safety_index: float = SAFETY_INDEX_CENTRAL,
    alpha_weights: np.ndarray | None = None,
    peak_exponent: float = PEAK_EXPONENT_CENTRAL,
    phase_scale: float = PHASE_SCALE_CENTRAL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return continuous base emission, rapping peak excess, and their sum.

    ``K = K0 * S(U) - [G(T) - G(T_ref)]`` with
    ``S(U) = sum_i alpha_i (U_i/U_ref_i)^p / sum_i alpha_i``.

    At the historical anchor ``U = U_ref`` and ``T = T_ref`` this returns
    ``K = K0 = ln(1000 C_in / C_out_recorded)``, i.e. the model reproduces the
    recorded operating point by construction.  Every non-control argument is a
    scenario input; none is estimated from the recorded outlet series.
    """
    u_ref = np.array([profile[f"{c}_median"] for c in U_COLS], float)
    t_ref = np.array([profile[f"{c}_median"] for c in T_COLS], float)
    cin = float(profile["C_in_gNm3"])
    k_anchor = float(profile["K0_anchor"])

    if alpha_weights is None:
        alpha_weights = ALPHA_CENTRAL
    alpha = np.asarray(alpha_weights, float)
    strength = np.sum(alpha * (u / u_ref) ** p_exponent, axis=-1) / alpha.sum()

    load_ratio = float(profile["load_ratio_to_train_median"])
    beta = BETA_CENTRAL * load_ratio ** 0.5

    def loss(period: np.ndarray) -> np.ndarray:
        ratio = period / t_star
        return np.sum(beta * (ratio + 1.0 / ratio - 2.0), axis=-1)

    ref_loss = float(loss(t_ref))
    k = k_anchor * strength - (loss(t) - ref_loss)
    base = cin * 1000.0 * np.exp(-k + safety_index)

    # Derived form: cake mass per rap ~ L * T, so the per-event peak grows
    # linearly with the period (peak_exponent = 1) while the contribution to
    # the hourly mean, ~ L*T*(1/T) = L, does not depend on the period at all.
    peak = (peak_scale * phase_scale * load_ratio
            * np.sum(PEAK_SHARES * (t / t_star) ** peak_exponent, axis=-1))
    return base, peak, base + peak


def field_priority(profile: pd.Series, model: PhysicalPowerModel,
                   alpha_weights: np.ndarray | None = None) -> np.ndarray:
    """Removal gain per unit of electrical power for each field.

    Differentiating ``K = K0 * sum_i alpha_i (U_i/U_ref_i)^2 / sum(alpha)`` and
    ``P = sum_i b_i U_i^2`` with respect to ``U_i^2`` gives

        dK/dP |_i = K0 * alpha_i / (sum(alpha) * b_i * U_ref_i^2),

    so the field to raise first is the one with the largest
    ``alpha_i / U_ref_i^2`` -- *not* the largest ``alpha_i``.
    """
    alpha = ALPHA_CENTRAL if alpha_weights is None else np.asarray(alpha_weights, float)
    u_ref = np.array([profile[f"{c}_median"] for c in U_COLS], float)
    return float(profile["K0_anchor"]) * alpha / (alpha.sum() * model.b * u_ref ** 2)


# ---------------------------------------------------------------- optimisation

def _bounds(profile: pd.Series, voltage_headroom: float) -> tuple[np.ndarray, np.ndarray]:
    lo = np.concatenate([[profile[f"{c}_min"] for c in U_COLS],
                         [profile[f"{c}_min"] for c in T_COLS]])
    hi = np.concatenate([[profile[f"{c}_max"] * (1.0 + voltage_headroom) for c in U_COLS],
                         [profile[f"{c}_max"] for c in T_COLS]])
    return lo.astype(float), hi.astype(float)


def _ordering(z: np.ndarray) -> np.ndarray:
    return np.array([z[0] - z[2] - 3.0, z[1] - z[3] - 3.0,
                     z[6] - z[4] - 100.0, z[7] - z[5] - 100.0])


def _round_controls(z: np.ndarray) -> np.ndarray:
    return np.concatenate([np.round(z[:4], 1), np.round(z[4:])])


def solve_condition(
    profile: pd.Series,
    model: PhysicalPowerModel,
    t_star: np.ndarray,
    limit: float,
    voltage_headroom: float = 0.0,
    mahalanobis: tuple[np.ndarray, np.ndarray, float] | None = None,
    multistart: int = MULTISTART,
    seed: int = SEED,
    **emission_kwargs,
) -> dict[str, object]:
    """Deterministic multi-start SLSQP solve of one condition/limit pair.

    The feasible set is *not* convex in ``U``: the objective ``sum b_i U_i^2``
    is convex while the emission constraint requires a convex function of
    ``U`` to be large enough, which is a reverse-convex constraint.  A dense
    multi-start is therefore used, and ``verify_with_random_search`` re-checks
    the result against a large random sample of the same feasible set.
    """
    lo, hi = _bounds(profile, voltage_headroom)

    def objective(z: np.ndarray) -> float:
        return float(model.power(z[:4], z[4:]))

    def emission_slack(z: np.ndarray) -> float:
        return float(limit - scenario_emission(profile, z[:4], z[4:], t_star, **emission_kwargs)[2])

    constraints: list[dict[str, object]] = [
        {"type": "ineq", "fun": emission_slack},
        {"type": "ineq", "fun": _ordering},
    ]
    if mahalanobis is not None:
        mu, inv_cov, threshold = mahalanobis
        constraints.append({
            "type": "ineq",
            "fun": lambda z: float(threshold - (z - mu) @ inv_cov @ (z - mu)),
        })

    u_ref = np.array([profile[f"{c}_median"] for c in U_COLS], float)
    t_ref = np.array([profile[f"{c}_median"] for c in T_COLS], float)
    rng = np.random.default_rng(seed)
    starts = [np.concatenate([u_ref, t_ref]), hi.copy(), lo.copy(), 0.5 * (lo + hi),
              np.concatenate([hi[:4], t_star])]
    starts += [lo + (hi - lo) * rng.random(8) for _ in range(multistart)]

    best_z: np.ndarray | None = None
    best_val = np.inf
    converged = 0
    values: list[float] = []
    for z0 in starts:
        result = minimize(objective, np.clip(z0, lo, hi), method="SLSQP",
                          bounds=list(zip(lo, hi)), constraints=constraints,
                          options={"maxiter": 600, "ftol": 1e-10})
        if not result.success:
            continue
        z = np.clip(result.x, lo, hi)
        if emission_slack(z) < -1e-7 or np.any(_ordering(z) < -1e-7):
            continue
        converged += 1
        values.append(float(result.fun))
        if result.fun < best_val:
            best_val, best_z = float(result.fun), z

    if best_z is None:
        return {
            "condition_cluster": int(profile["condition_cluster"]),
            "condition_name": str(profile["condition_name"]),
            "limit_mgNm3": float(limit),
            "voltage_headroom": float(voltage_headroom),
            "status": "INFEASIBLE_IN_DECLARED_SUPPORT",
        }

    # Round to the resolution a DCS can actually hold, then repair feasibility
    # by nudging the most power-efficient field upward.
    z = _round_controls(best_z)
    order = np.argsort(-field_priority(profile, model, emission_kwargs.get("alpha_weights")))
    for _ in range(400):
        if emission_slack(z) >= 0 and np.all(_ordering(z) >= -1e-9):
            break
        for i in order:
            if z[i] + 0.1 <= hi[i] + 1e-9:
                z[i] = round(z[i] + 0.1, 1)
                break
        else:
            break
    if emission_slack(z) < 0:
        z = best_z  # keep the unrounded solution rather than report an infeasible one

    base, peak, total = scenario_emission(profile, z[:4], z[4:], t_star, **emission_kwargs)
    row: dict[str, object] = {
        "condition_cluster": int(profile["condition_cluster"]),
        "condition_name": str(profile["condition_name"]),
        "limit_mgNm3": float(limit),
        "voltage_headroom": float(voltage_headroom),
        "status": "SCENARIO_FEASIBLE",
        "scenario_base_emission_mgNm3": float(base),
        "scenario_peak_excess_mgNm3": float(peak),
        "scenario_peak_total_mgNm3": float(total),
        "predicted_power_kW": float(model.power(z[:4], z[4:])),
        "field_power_kW": float(model.field_power(z[:4])),
        "rapping_power_kW": float(model.rapping_power(z[4:])),
        "historical_power_kW": float(profile["historical_power_kW"]),
        "multistart_converged": converged,
        "multistart_power_spread_kW": float(np.max(values) - np.min(values)) if values else np.nan,
    }
    row["power_change_vs_historical_pct"] = 100.0 * (
        row["predicted_power_kW"] / float(profile["historical_power_kW"]) - 1.0)
    for i, col in enumerate(U_COLS + T_COLS):
        row[col] = float(z[i])
    return row


def verify_with_random_search(
    profile: pd.Series,
    model: PhysicalPowerModel,
    t_star: np.ndarray,
    limit: float,
    voltage_headroom: float,
    n: int = 400_000,
    seed: int = SEED,
    **emission_kwargs,
) -> float:
    """Best feasible power found by uniform random sampling of the same set."""
    lo, hi = _bounds(profile, voltage_headroom)
    rng = np.random.default_rng(seed)
    z = lo + (hi - lo) * rng.random((n, 8))
    u = np.round(z[:, :4], 1)
    t = np.round(z[:, 4:])
    keep = ((u[:, 0] >= u[:, 2] + 3.0) & (u[:, 1] >= u[:, 3] + 3.0)
            & (t[:, 2] >= t[:, 0] + 100.0) & (t[:, 3] >= t[:, 1] + 100.0))
    u, t = u[keep], t[keep]
    if not len(u):
        return float("nan")
    _, _, total = scenario_emission(profile, u, t, t_star, **emission_kwargs)
    feasible = total <= limit
    if not np.any(feasible):
        return float("nan")
    return float(np.min(model.power(u[feasible], t[feasible])))


def aggregate(rows: list[dict[str, object]], key: str = "predicted_power_kW") -> float:
    weights = np.array([float(r["share"]) for r in rows])
    return float(np.average([float(r[key]) for r in rows], weights=weights))


def solve_all(profiles: pd.DataFrame, model: PhysicalPowerModel,
              t_stars: dict[int, np.ndarray], limit: float, headroom: float,
              multistart: int = MULTISTART, **emission_kwargs) -> list[dict[str, object]]:
    rows = []
    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        r = solve_condition(profile, model, t_stars[cluster], limit,
                            voltage_headroom=headroom, multistart=multistart,
                            **emission_kwargs)
        r["share"] = float(profile["share"])
        rows.append(r)
    return rows


# ------------------------------------------------------- data-side diagnostics

def historical_strategy_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Operator response of the control settings to the exogenous conditions.

    Unlike the outlet concentration, this relationship *is* identifiable, and
    it gives Question 3 an empirical baseline.  It is reported per window as
    well as over the whole record, because the two are not the same: the
    rapping-period rule holds throughout, while the voltage response to inlet
    concentration reverses sign between the first five days and the last two.
    """
    windows = [("full_record", df), ("train", df[df["split"] == "train"]),
               ("validation", df[df["split"] == "validation"]),
               ("retrospective_test", df[df["split"] == "test"])]
    rows = []
    for name, frame in windows:
        for control in U_COLS + T_COLS:
            row: dict[str, object] = {"window": name, "rows": int(len(frame)), "control": control}
            for cond in COND_COLS:
                row[f"r_{cond}"] = float(np.corrcoef(frame[cond], frame[control])[0, 1])
            rows.append(row)
    return pd.DataFrame(rows)


def anomaly_events(df: pd.DataFrame) -> pd.DataFrame:
    """The two sustained excursions planted in the attachment."""
    specs = [
        ("入口浓度尖峰", df["C_in_gNm3"] > 60.0),
        ("入口温度尖峰", df["Temp_C"] > 150.0),
    ]
    baseline = {c: float(df[c].median()) for c in COND_COLS + U_COLS + T_COLS + ["P_total_kW"]}
    rows = []
    for name, mask in specs:
        sub = df[mask]
        row: dict[str, object] = {
            "event": name,
            "start": str(sub["timestamp"].min()),
            "end": str(sub["timestamp"].max()),
            "minutes": int(len(sub)),
        }
        for col in COND_COLS + U_COLS + T_COLS + ["P_total_kW"]:
            row[f"{col}_event_mean"] = float(sub[col].mean())
            row[f"{col}_event_max"] = float(sub[col].max())
            row[f"{col}_event_min"] = float(sub[col].min())
            row[f"{col}_baseline_median"] = baseline[col]
        rows.append(row)
    return pd.DataFrame(rows)


def rapping_power_tradeoff(train: pd.DataFrame, model: PhysicalPowerModel) -> pd.DataFrame:
    """How much power the rapping term holds, and what extending it recovers."""
    u_med = train[U_COLS].median().to_numpy(float)
    t_med = train[T_COLS].median().to_numpy(float)
    field = float(model.field_power(u_med))
    base_rap = float(model.rapping_power(t_med))
    total = model.intercept + field + base_rap
    rows = [{
        "scenario": "历史中位周期",
        "T_setting": ";".join(f"{v:.0f}" for v in t_med),
        "field_power_kW": field,
        "rapping_power_kW": base_rap,
        "total_power_kW": total,
        "rapping_share_pct": 100.0 * base_rap / total,
        "power_recovered_kW": 0.0,
        "cake_mass_per_rap_change_pct": 0.0,
    }]
    for label, t_new in (("延长至历史P95", train[T_COLS].quantile(0.95).to_numpy(float)),
                         ("延长至历史最大值", train[T_COLS].max().to_numpy(float))):
        rap = float(model.rapping_power(t_new))
        rows.append({
            "scenario": label,
            "T_setting": ";".join(f"{v:.0f}" for v in t_new),
            "field_power_kW": field,
            "rapping_power_kW": rap,
            "total_power_kW": model.intercept + field + rap,
            "rapping_share_pct": 100.0 * rap / (model.intercept + field + rap),
            "power_recovered_kW": base_rap - rap,
            "cake_mass_per_rap_change_pct": 100.0 * (float(np.mean(t_new / t_med)) - 1.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- driver

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    (OUT_DIR / "power_model_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    guard = float(metrics["validation_abs_error_q90_kW"])

    pd.read_csv(CLUSTER_EVAL_FILE).to_csv(
        OUT_DIR / "cluster_selection_metrics.csv", index=False, encoding="utf-8-sig")

    historical_strategy_correlations(df).to_csv(
        OUT_DIR / "historical_strategy_correlations.csv", index=False, encoding="utf-8-sig")
    anomaly_events(df).to_csv(OUT_DIR / "anomaly_events.csv", index=False, encoding="utf-8-sig")
    rapping_power_tradeoff(train, model).to_csv(
        OUT_DIR / "rapping_power_tradeoff.csv", index=False, encoding="utf-8-sig")

    global_t_median = train[T_COLS].median().to_numpy(float)
    t_stars = {int(p["condition_cluster"]): scenario_t_star(p, global_t_median)
               for _, p in profiles.iterrows()}

    # -- field priority criterion (analytic) --------------------------------
    priority_rows = []
    for _, profile in profiles.iterrows():
        crit = field_priority(profile, model)
        u_ref = np.array([profile[f"{c}_median"] for c in U_COLS], float)
        row = {"condition_cluster": int(profile["condition_cluster"]),
               "condition_name": profile["condition_name"]}
        for i in range(4):
            row[f"U{i+1}_ref_kV"] = float(u_ref[i])
            row[f"alpha_{i+1}"] = float(ALPHA_CENTRAL[i])
            row[f"criterion_{i+1}_per_kW"] = float(crit[i])
        row["rank_order"] = ">".join(f"E{i+1}" for i in np.argsort(-crit))
        priority_rows.append(row)
    pd.DataFrame(priority_rows).to_csv(
        OUT_DIR / "field_priority_criterion.csv", index=False, encoding="utf-8-sig")

    # -- central solutions ---------------------------------------------------
    rows10 = solve_all(profiles, model, t_stars, 10.0, HEADROOM_10)
    rows5 = solve_all(profiles, model, t_stars, 5.0, HEADROOM_5)
    optimum = pd.DataFrame(rows10 + rows5)
    optimum["validation_error_adjusted_power_kW"] = optimum["predicted_power_kW"] + guard
    optimum["validation_error_guard_kW"] = guard
    optimum.to_csv(OUT_DIR / "optimal_controls_central_scenario.csv",
                   index=False, encoding="utf-8-sig")

    # -- support-domain audit (reported, not imposed) ------------------------
    audit_rows = []
    for row in rows10 + rows5:
        cluster = int(row["condition_cluster"])
        group = train[train["condition_cluster"] == cluster]
        mu, inv_cov, threshold = support_statistics(group)
        z = np.array([float(row[c]) for c in U_COLS + T_COLS])
        d2 = float((z - mu) @ inv_cov @ (z - mu))
        u_max = np.array([group[c].max() for c in U_COLS], float)
        audit_rows.append({
            "condition_cluster": cluster,
            "condition_name": row["condition_name"],
            "limit_mgNm3": row["limit_mgNm3"],
            "mahalanobis_d2": d2,
            "empirical_q975": threshold,
            "ratio_to_q975": d2 / threshold,
            "max_voltage_over_historical_max_pct": float(
                100.0 * max((float(row[c]) / m - 1.0) for c, m in zip(U_COLS, u_max))),
            "verdict": "SUPPORTED" if d2 <= threshold else "EXTRAPOLATION",
        })
    pd.DataFrame(audit_rows).to_csv(OUT_DIR / "support_boundary_audit.csv",
                                    index=False, encoding="utf-8-sig")

    # Does the historical support ellipsoid admit any feasible point at all?
    restricted_rows = []
    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        stats = support_statistics(train[train["condition_cluster"] == cluster])
        for limit, headroom in ((10.0, HEADROOM_10), (5.0, HEADROOM_5)):
            r = solve_condition(profile, model, t_stars[cluster], limit,
                                voltage_headroom=headroom, mahalanobis=stats)
            restricted_rows.append({
                "condition_cluster": cluster, "limit_mgNm3": limit,
                "status_with_mahalanobis_constraint": r["status"],
                "predicted_power_kW": r.get("predicted_power_kW", np.nan),
            })
    pd.DataFrame(restricted_rows).to_csv(
        OUT_DIR / "mahalanobis_constrained_feasibility.csv", index=False, encoding="utf-8-sig")

    # -- minimum voltage headroom needed at each limit -----------------------
    headroom_rows = []
    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        for limit in (10.0, 5.0):
            need = np.nan
            for m in np.arange(0.0, 0.31, 0.01):
                if solve_condition(profile, model, t_stars[cluster], limit,
                                   voltage_headroom=float(m), multistart=16
                                   )["status"] == "SCENARIO_FEASIBLE":
                    need = float(m)
                    break
            headroom_rows.append({
                "condition_cluster": cluster,
                "condition_name": profile["condition_name"],
                "limit_mgNm3": limit,
                "min_voltage_headroom_pct": 100.0 * need,
            })
    pd.DataFrame(headroom_rows).to_csv(
        OUT_DIR / "voltage_headroom_requirement.csv", index=False, encoding="utf-8-sig")

    # -- optimiser verification against dense random search ------------------
    verify_rows = []
    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        for limit, headroom, rows in ((10.0, HEADROOM_10, rows10), (5.0, HEADROOM_5, rows5)):
            solved = next(r for r in rows if r["condition_cluster"] == cluster)
            if solved["status"] != "SCENARIO_FEASIBLE":
                continue
            rnd = verify_with_random_search(profile, model, t_stars[cluster], limit, headroom)
            verify_rows.append({
                "condition_cluster": cluster,
                "condition_name": profile["condition_name"],
                "limit_mgNm3": limit,
                "slsqp_power_kW": solved["predicted_power_kW"],
                "random_search_power_kW": rnd,
                "random_search_gap_pct": 100.0 * (rnd / float(solved["predicted_power_kW"]) - 1.0),
                "multistart_converged": solved["multistart_converged"],
                "multistart_power_spread_kW": solved["multistart_power_spread_kW"],
            })
    verification = pd.DataFrame(verify_rows)
    verification.to_csv(OUT_DIR / "optimization_verification.csv", index=False, encoding="utf-8-sig")

    # -- Question 4 ----------------------------------------------------------
    q4_rows = []
    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        a = next(r for r in rows10 if r["condition_cluster"] == cluster)
        b = next(r for r in rows5 if r["condition_cluster"] == cluster)
        k10 = float(profile["K0_anchor"]) * float(
            np.sum(ALPHA_CENTRAL * (np.array([a[c] for c in U_COLS], float)
                                    / np.array([profile[f"{c}_median"] for c in U_COLS], float)) ** 2)
            / ALPHA_CENTRAL.sum())
        delta_k = math.log((10.0 - float(a["scenario_peak_excess_mgNm3"]))
                           / (5.0 - float(a["scenario_peak_excess_mgNm3"])))
        q4_rows.append({
            "condition_cluster": cluster,
            "condition_name": profile["condition_name"],
            "share": float(profile["share"]),
            "power_10_kW": float(a["predicted_power_kW"]),
            "power_5_kW": float(b["predicted_power_kW"]),
            "field_power_10_kW": float(a["field_power_kW"]),
            "K_at_10": k10,
            "delta_K_10_to_5": delta_k,
            "analytic_delta_power_kW": float(a["field_power_kW"]) * delta_k / k10,
            "numeric_delta_power_kW": float(b["predicted_power_kW"]) - float(a["predicted_power_kW"]),
            "increase_pct": 100.0 * (float(b["predicted_power_kW"]) / float(a["predicted_power_kW"]) - 1.0),
        })
    q4 = pd.DataFrame(q4_rows)
    q4.to_csv(OUT_DIR / "question4_by_condition.csv", index=False, encoding="utf-8-sig")
    weighted_p10 = float(np.average(q4["power_10_kW"], weights=q4["share"]))
    weighted_p5 = float(np.average(q4["power_5_kW"], weights=q4["share"]))
    historical_weighted = float(np.average(profiles["historical_power_kW"], weights=profiles["share"]))
    q4_summary = {
        "weighted_power_10_kW": weighted_p10,
        "weighted_power_5_kW": weighted_p5,
        "weighted_increase_pct": 100.0 * (weighted_p5 / weighted_p10 - 1.0),
        "historical_weighted_actual_power_kW": historical_weighted,
        "power_10_vs_historical_pct": 100.0 * (weighted_p10 / historical_weighted - 1.0),
        "daily_energy_10_MWh": weighted_p10 * 24.0 / 1000.0,
        "daily_energy_5_MWh": weighted_p5 * 24.0 / 1000.0,
        "daily_energy_historical_MWh": historical_weighted * 24.0 / 1000.0,
        "validation_error_guard_kW": guard,
    }
    (OUT_DIR / "question4_summary.json").write_text(
        json.dumps(q4_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- p reference table: the single scalar that needs site calibration ----
    # Holding the 10 mg/Nm^3 operating point fixed, scaling every voltage by f
    # gives K = K_10 f^p and P_field = P_field,10 f^2, so halving the limit
    # (delta_K) costs exactly P_field,10 [(1 + delta_K/K_10)^{2/p} - 1]:
    # the penalty is governed by 2/p and nothing else.
    p_rows = []
    for p_exp in (1.0, 1.5, 2.0, 2.5, 3.0):
        analytic = []
        for row in q4_rows:
            ratio = 1.0 + row["delta_K_10_to_5"] / row["K_at_10"]
            analytic.append({"share": row["share"],
                             "delta": row["field_power_10_kW"] * (ratio ** (2.0 / p_exp) - 1.0),
                             "p10": row["power_10_kW"]})
        w = np.array([a["share"] for a in analytic])
        delta = float(np.average([a["delta"] for a in analytic], weights=w))
        p10 = float(np.average([a["p10"] for a in analytic], weights=w))
        # required voltage headroom above the historical maximum, same reasoning
        f = float(np.mean([(1.0 + r["delta_K_10_to_5"] / r["K_at_10"]) ** (1.0 / p_exp)
                           for r in q4_rows]))
        r10 = solve_all(profiles, model, t_stars, 10.0, HEADROOM_10,
                        multistart=SWEEP_MULTISTART, p_exponent=p_exp)
        r5 = solve_all(profiles, model, t_stars, 5.0, HEADROOM_5,
                       multistart=SWEEP_MULTISTART, p_exponent=p_exp)
        feasible = all(r["status"] == "SCENARIO_FEASIBLE" for r in r10 + r5)
        p_rows.append({
            "p_exponent": p_exp,
            "analytic_delta_power_kW": delta,
            "analytic_increase_pct": 100.0 * delta / p10,
            "implied_voltage_scaling_10_to_5": f,
            "feasible_at_5pct_headroom": feasible,
            "numeric_weighted_power_10_kW": aggregate(r10) if feasible else np.nan,
            "numeric_weighted_power_5_kW": aggregate(r5) if feasible else np.nan,
            "numeric_increase_pct": (100.0 * (aggregate(r5) / aggregate(r10) - 1.0)) if feasible else np.nan,
        })
    pd.DataFrame(p_rows).to_csv(OUT_DIR / "p_exponent_reference.csv",
                                index=False, encoding="utf-8-sig")

    # -- scale sensitivity: p x peak scale x safety index (27 combinations) --
    sensitivity_rows = []
    for p_exp in (1.5, 2.0, 2.5):
        for peak_scale in (0.9, 1.2, 1.5):
            for safety in (0.0, 0.08, 0.15):
                kw = dict(p_exponent=p_exp, peak_scale=peak_scale, safety_index=safety)
                r10 = solve_all(profiles, model, t_stars, 10.0, HEADROOM_10,
                                multistart=SWEEP_MULTISTART, **kw)
                r5 = solve_all(profiles, model, t_stars, 5.0, HEADROOM_5,
                               multistart=SWEEP_MULTISTART, **kw)
                ok = all(r["status"] == "SCENARIO_FEASIBLE" for r in r10 + r5)
                sensitivity_rows.append({
                    "p_exponent": p_exp, "peak_scale": peak_scale, "safety_index": safety,
                    "all_conditions_feasible": ok,
                    "weighted_power_10_kW": aggregate(r10) if ok else np.nan,
                    "weighted_power_5_kW": aggregate(r5) if ok else np.nan,
                    "weighted_power_increase_pct": (100.0 * (aggregate(r5) / aggregate(r10) - 1.0)) if ok else np.nan,
                })
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(OUT_DIR / "question4_sensitivity.csv", index=False, encoding="utf-8-sig")

    # -- structural sensitivity: 81 grey-box shape combinations --------------
    alpha_profiles = {
        "equal": np.array([1.00, 1.00, 1.00, 1.00]),
        "weak_front": np.array([1.08, 1.04, 0.98, 0.90]),
        "central_front": ALPHA_CENTRAL,
    }
    structural_rows = []
    for alpha_name, alpha_weights in alpha_profiles.items():
        for peak_exponent in (1.00, 1.30, 1.60):
            for phase_scale in (0.65, 0.80, 1.00):
                for tstar_exponent in (-0.30, -0.18, -0.10):
                    stars = {int(p["condition_cluster"]): scenario_t_star(p, global_t_median, tstar_exponent)
                             for _, p in profiles.iterrows()}
                    kw = dict(alpha_weights=alpha_weights, peak_exponent=peak_exponent,
                              phase_scale=phase_scale)
                    r10 = solve_all(profiles, model, stars, 10.0, HEADROOM_10,
                                    multistart=SWEEP_MULTISTART, **kw)
                    r5 = solve_all(profiles, model, stars, 5.0, HEADROOM_5,
                                   multistart=SWEEP_MULTISTART, **kw)
                    ok = all(r["status"] == "SCENARIO_FEASIBLE" for r in r10 + r5)
                    structural_rows.append({
                        "alpha_profile": alpha_name,
                        "peak_exponent": peak_exponent,
                        "phase_scale": phase_scale,
                        "tstar_load_exponent": tstar_exponent,
                        "all_conditions_feasible": ok,
                        "weighted_power_10_kW": aggregate(r10) if ok else np.nan,
                        "weighted_power_5_kW": aggregate(r5) if ok else np.nan,
                        "weighted_power_increase_pct": (100.0 * (aggregate(r5) / aggregate(r10) - 1.0)) if ok else np.nan,
                    })
    structural = pd.DataFrame(structural_rows)
    structural.to_csv(OUT_DIR / "structural_sensitivity.csv", index=False, encoding="utf-8-sig")

    # -- field weight ablation ----------------------------------------------
    ablation_rows, ablation_summary_rows = [], []
    for alpha_name, alpha_weights in alpha_profiles.items():
        r10 = solve_all(profiles, model, t_stars, 10.0, HEADROOM_10,
                        multistart=SWEEP_MULTISTART, alpha_weights=alpha_weights)
        r5 = solve_all(profiles, model, t_stars, 5.0, HEADROOM_5,
                       multistart=SWEEP_MULTISTART, alpha_weights=alpha_weights)
        per_profile = []
        for a, b in zip(r10, r5):
            row = {"alpha_profile": alpha_name, "condition_cluster": a["condition_cluster"],
                   "condition_name": a["condition_name"], "share": a["share"]}
            for i, col in enumerate(U_COLS, start=1):
                row[f"delta_U{i}_kV"] = float(b[col]) - float(a[col])
            per_profile.append(row)
            ablation_rows.append(row)
        weights = np.array([x["share"] for x in per_profile])
        deltas = np.array([[x[f"delta_U{i}_kV"] for i in range(1, 5)] for x in per_profile])
        weighted = np.average(deltas, axis=0, weights=weights)
        positive = np.maximum(weighted, 0)
        ablation_summary_rows.append({
            "alpha_profile": alpha_name,
            **{f"weighted_delta_U{i}_kV": weighted[i - 1] for i in range(1, 5)},
            "front_two_share_of_positive_adjustment": float(positive[:2].sum() / positive.sum())
            if positive.sum() else np.nan,
        })
    pd.DataFrame(ablation_rows).to_csv(
        OUT_DIR / "field_priority_ablation_by_condition.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ablation_summary_rows).to_csv(
        OUT_DIR / "field_priority_ablation_summary.csv", index=False, encoding="utf-8-sig")

    # -- historical comparison and online schedule ---------------------------
    historical_rows, dynamic_rows = [], []
    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        u_med = np.array([profile[f"{c}_median"] for c in U_COLS], float)
        t_med = np.array([profile[f"{c}_median"] for c in T_COLS], float)
        median_power = float(model.power(u_med, t_med))
        a = next(r for r in rows10 if r["condition_cluster"] == cluster)
        b = next(r for r in rows5 if r["condition_cluster"] == cluster)
        historical_rows.append({
            "condition_cluster": cluster,
            "condition_name": profile["condition_name"],
            "share": float(profile["share"]),
            "historical_mean_actual_power_kW": float(profile["historical_power_kW"]),
            "model_power_at_historical_median_controls_kW": median_power,
            "scenario_optimal_10_power_kW": float(a["predicted_power_kW"]),
            "power_change_vs_model_median_pct": 100.0 * (float(a["predicted_power_kW"]) / median_power - 1.0),
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
            "implementation_status": "PROVISIONAL_REQUIRES_T/R_CAPACITY_CHECK_AND_SITE_INTERLOCK_LIMITS",
            **{f"target10_{c}": a[c] for c in U_COLS + T_COLS},
            **{f"target5_{c}": b[c] for c in U_COLS + T_COLS},
        })
    pd.DataFrame(historical_rows).to_csv(
        OUT_DIR / "historical_median_power_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dynamic_rows).to_csv(
        OUT_DIR / "dynamic_control_schedule.csv", index=False, encoding="utf-8-sig")

    scenario_params = {
        "status": "EMISSION_LAYER_IS_SCENARIO_ONLY_C_OUT_NOT_IDENTIFIABLE",
        "emission_anchor": "recorded outlet median, no decimal rescaling",
        "p_exponent_central": P_EXPONENT_CENTRAL,
        "p_exponent_grid": [1.0, 1.5, 2.0, 2.5, 3.0],
        "peak_exponent_central": PEAK_EXPONENT_CENTRAL,
        "peak_exponent_grid": [1.00, 1.30, 1.60],
        "voltage_weight_profiles": {k: v.tolist() for k, v in alpha_profiles.items()},
        "rapping_phase_scales": [0.65, 0.80, 1.00],
        "tstar_load_exponents": [-0.30, -0.18, -0.10],
        "rapping_loss_coefficients": BETA_CENTRAL.tolist(),
        "peak_field_shares": PEAK_SHARES.tolist(),
        "support_rule": "cluster-specific full historical range plus engineering ordering; "
                        "Mahalanobis d2 reported as extrapolation audit, not imposed",
        "voltage_headroom_10": HEADROOM_10,
        "voltage_headroom_5": HEADROOM_5,
        "optimiser": "multi-start SLSQP (%d starts) verified against 400k random samples" % MULTISTART,
    }
    (OUT_DIR / "scenario_parameters.json").write_text(
        json.dumps(scenario_params, ensure_ascii=False, indent=2), encoding="utf-8")

    feasible_structural = structural[structural["all_conditions_feasible"]]
    feasible_scale = sensitivity[sensitivity["all_conditions_feasible"]]
    summary = {
        "power_model": metrics,
        "cluster_k": 4,
        "central_q4": q4_summary,
        "rapping_share_at_median_pct": float(
            rapping_power_tradeoff(train, model).iloc[0]["rapping_share_pct"]),
        "scale_sensitivity": {
            "feasible_combinations": int(len(feasible_scale)),
            "total_combinations": int(len(sensitivity)),
            "increase_pct_min": float(feasible_scale["weighted_power_increase_pct"].min()),
            "increase_pct_median": float(feasible_scale["weighted_power_increase_pct"].median()),
            "increase_pct_max": float(feasible_scale["weighted_power_increase_pct"].max()),
        },
        "structural_sensitivity": {
            "feasible_combinations": int(len(feasible_structural)),
            "total_combinations": int(len(structural)),
            "increase_pct_min": float(feasible_structural["weighted_power_increase_pct"].min()),
            "increase_pct_median": float(feasible_structural["weighted_power_increase_pct"].median()),
            "increase_pct_max": float(feasible_structural["weighted_power_increase_pct"].max()),
        },
        "optimiser_verification": {
            "max_random_search_gap_pct": float(verification["random_search_gap_pct"].max()),
            "max_multistart_spread_kW": float(verification["multistart_power_spread_kW"].max()),
        },
        "central_all_conditions_feasible": bool((optimum["status"] == "SCENARIO_FEASIBLE").all()),
    }
    (OUT_DIR / "model_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
