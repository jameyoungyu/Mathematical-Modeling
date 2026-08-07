#!/usr/bin/env python3
"""Compute the scenario emission-power trade-off frontier.

The support-domain search is repeated at a range of emission limits, so the
paper can show the whole trade-off curve rather than only the 10 and
5 mg/Nm^3 end points.  Because the historical anchor is ~50 mg/Nm^3, every
point on this curve costs power relative to historical operation; the curve
also records how much voltage headroom above the historical maximum each
limit needs before it becomes reachable at all.

Usage:  python3 src/compute_frontier.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scenario_model import (
    DATA_FILE,
    HEADROOM_5,
    T_COLS,
    condition_profiles,
    evaluate_power_models,
    scenario_t_star,
    solve_condition,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"
LIMITS = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def main() -> None:
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df["condition_cluster"] = df["condition_cluster"].astype(int)
    profiles = condition_profiles(df)

    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()
    model, metrics, _, _ = evaluate_power_models(train, validation, test)
    guard = float(metrics["validation_abs_error_q90_kW"])

    global_t_median = train[T_COLS].median().to_numpy(float)
    t_stars = {int(p["condition_cluster"]): scenario_t_star(p, global_t_median)
               for _, p in profiles.iterrows()}
    historical_weighted = float(np.average(profiles["historical_power_kW"],
                                           weights=profiles["share"]))

    rows: list[dict[str, object]] = []
    weighted_rows: list[dict[str, object]] = []
    for limit in LIMITS:
        # smallest declared headroom that makes every condition reachable
        headroom = 0.0
        while headroom <= HEADROOM_5 + 1e-9:
            trial = [solve_condition(p, model, t_stars[int(p["condition_cluster"])], limit,
                                     voltage_headroom=headroom, multistart=12)
                     for _, p in profiles.iterrows()]
            if all(r["status"] == "SCENARIO_FEASIBLE" for r in trial):
                break
            headroom = round(headroom + 0.01, 2)

        solved = [solve_condition(p, model, t_stars[int(p["condition_cluster"])], limit,
                                  voltage_headroom=headroom)
                  for _, p in profiles.iterrows()]
        for profile_row, r in zip(profiles.itertuples(), solved):
            rows.append({
                "condition_cluster": r["condition_cluster"],
                "condition_name": r["condition_name"],
                "limit_mgNm3": limit,
                "voltage_headroom_pct": 100.0 * headroom,
                "status": r["status"],
                "predicted_power_kW": r.get("predicted_power_kW", np.nan),
                "validation_error_adjusted_power_kW": r.get("predicted_power_kW", np.nan) + guard,
                "share": float(profile_row.share),
                "historical_power_kW": historical_weighted,
            })
        weights = np.array([float(p.share) for p in profiles.itertuples()])
        powers = np.array([float(r.get("predicted_power_kW", np.nan)) for r in solved])
        weighted_rows.append({
            "limit_mgNm3": limit,
            "voltage_headroom_pct": 100.0 * headroom,
            "weighted_power_kW": float(np.average(powers, weights=weights)),
            "weighted_daily_energy_MWh": float(np.average(powers, weights=weights)) * 24.0 / 1000.0,
            "historical_power_kW": historical_weighted,
        })

    pd.DataFrame(rows).to_csv(OUT_DIR / "emission_power_frontier.csv",
                              index=False, encoding="utf-8-sig")
    pd.DataFrame(weighted_rows).to_csv(OUT_DIR / "emission_power_frontier_weighted.csv",
                                       index=False, encoding="utf-8-sig")
    print(pd.DataFrame(weighted_rows).to_string(index=False))


if __name__ == "__main__":
    main()
