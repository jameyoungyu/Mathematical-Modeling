#!/usr/bin/env python3
"""Compute the scenario emission–power trade-off frontier.

For each typical operating condition the support-domain search is repeated at
a range of emission limits, so the paper can show the whole trade-off curve
rather than only the 10 and 5 mg/Nm³ end points. This makes explicit that the
10 mg/Nm³ solution relaxes emission relative to the 0.1-scaled historical
anchor (~5 mg/Nm³) instead of saving energy at constant emission.

Usage:  python3 src/compute_frontier.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_model import (
    DATA_FILE,
    SEED,
    T_COLS,
    U_COLS,
    candidate_bank,
    condition_profiles,
    evaluate_power_models,
    local_candidate_bank,
    scenario_t_star,
    select_optimum,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results"
LIMITS = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
BASE_CANDIDATES = 220_000
LOCAL_CANDIDATES = 30_000


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
    rows: list[dict[str, object]] = []

    for _, profile in profiles.iterrows():
        cluster = int(profile["condition_cluster"])
        group = train[train["condition_cluster"] == cluster]
        t_star = scenario_t_star(profile, global_t_median)
        rng = np.random.default_rng(SEED)

        base_bank = candidate_bank(profile, model, t_star, BASE_CANDIDATES, rng, group)
        # 先在两个端点做局部细化，再把细化样本并入公共候选库
        additions = []
        for limit in (LIMITS[0], LIMITS[-1]):
            prelim = select_optimum(profile, base_bank, t_star, limit,
                                    validation_error_guard_kW=guard)
            if "predicted_power_kW" in prelim:
                additions.append(local_candidate_bank(profile, model, prelim,
                                                      LOCAL_CANDIDATES, rng, group))
        bank = pd.concat([base_bank] + additions, ignore_index=True)
        bank = bank.drop_duplicates(U_COLS + T_COLS, keep="first")

        for limit in LIMITS:
            res = select_optimum(profile, bank, t_star, limit,
                                 validation_error_guard_kW=guard)
            rows.append({
                "condition_cluster": cluster,
                "condition_name": profile["condition_name"],
                "share": float(profile["share"]),
                "limit_mgNm3": limit,
                "status": res["status"],
                "predicted_power_kW": res.get("predicted_power_kW", np.nan),
                "scenario_base_emission_mgNm3": res.get("scenario_base_emission_mgNm3", np.nan),
                "scenario_peak_total_mgNm3": res.get("scenario_peak_total_mgNm3", np.nan),
                "historical_power_kW": float(profile["historical_power_kW"]),
                "historical_anchor_mgNm3": float(profile["C_out_scaled_anchor"]),
            })
            print(f"工况{cluster}  限值{limit:>4.1f}  "
                  f"功率{res.get('predicted_power_kW', float('nan')):8.2f} kW  {res['status']}")

    frontier = pd.DataFrame(rows)
    frontier.to_csv(OUT_DIR / "emission_power_frontier.csv", index=False, encoding="utf-8-sig")

    weighted = (frontier.groupby("limit_mgNm3")
                .apply(lambda g: float(np.average(g["predicted_power_kW"], weights=g["share"])),
                       include_groups=False)
                .rename("weighted_power_kW").reset_index())
    weighted.to_csv(OUT_DIR / "emission_power_frontier_weighted.csv",
                    index=False, encoding="utf-8-sig")
    print("\n工况加权前沿：")
    print(weighted.to_string(index=False))


if __name__ == "__main__":
    main()
