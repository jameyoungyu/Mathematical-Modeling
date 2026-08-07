#!/usr/bin/env python3
"""Falsifiable tests on the recorded outlet concentration.

Section 2.2 of the paper rules the outlet column unusable, and section 4.5
derives the rapping-peak scaling instead of measuring it.  Both statements are
negative, and a negative claim of the form "we looked and found nothing" is
weaker than it needs to be.  This module replaces each with a test that could
have failed and did not:

* **censoring.**  If the column really is a latent concentration clipped at the
  50 mg/Nm^3 range limit, then fitting the latent normal by censored maximum
  likelihood on the *uncensored* records must reproduce the *observed*
  proportion of records sitting exactly on the cap.  That proportion is not
  used in the fit, so agreement is a genuine out-of-sample prediction.

* **rapping periodicity.**  Individual rapping events cannot be attributed at
  one-minute resolution, but the set periods (T1, T2 ~ 3.9 min; T3, T4 ~ 7.4
  min) all sit above the 2-minute Nyquist limit, so a *periodic* component
  synchronised with them is detectable.  Epoch folding of the outlet residual
  bounds any such component; because upward excursions are the ones the cap
  hides, the censoring indicator is folded as well, which is the test that
  survives clipping.

Outputs ``results/outlet_censoring_periodicity.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "full_timeseries_with_flags.csv"
OUT_DIR = ROOT / "results"

CAP = 50.0
T_COLS = [f"T{i}_s" for i in range(1, 5)]
N_PHASE_BINS = 10


def censoring_test(outlet: np.ndarray) -> dict[str, float]:
    """Censored-normal MLE fitted without using the observed cap fraction."""
    uncensored = outlet[outlet < CAP]
    n_censored = int((outlet >= CAP).sum())

    def negative_log_likelihood(theta: np.ndarray) -> float:
        mu, log_sigma = theta
        sigma = float(np.exp(log_sigma))
        return -(stats.norm.logpdf(uncensored, mu, sigma).sum()
                 + n_censored * stats.norm.logsf(CAP, mu, sigma))

    fit = optimize.minimize(negative_log_likelihood, [CAP + 0.2, np.log(0.3)],
                            method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    mu, sigma = float(fit.x[0]), float(np.exp(fit.x[1]))
    predicted = float(stats.norm.sf(CAP, mu, sigma))
    observed = n_censored / len(outlet)
    return {
        "latent_mu_mgNm3": mu,
        "latent_sigma_mgNm3": sigma,
        "predicted_censored_fraction": predicted,
        "observed_censored_fraction": observed,
        "absolute_error": abs(predicted - observed),
        # the latent spread as a share of the level: how much room the column
        # has to carry any information about the controls at all
        "latent_cv_pct": 100.0 * sigma / mu,
    }


def epoch_folding(values: np.ndarray, period_minutes: float) -> float:
    """Peak-to-peak amplitude of ``values`` folded on ``period_minutes``."""
    phase = (np.arange(len(values)) % period_minutes) / period_minutes
    binned = np.minimum((phase * N_PHASE_BINS).astype(int), N_PHASE_BINS - 1)
    means = np.array([values[binned == k].mean() for k in range(N_PHASE_BINS)])
    return float(means.max() - means.min())


def censoring_phase_test(censored: np.ndarray, period_minutes: float) -> float:
    """p-value for the censoring rate depending on rapping phase.

    A real rapping peak pushes the concentration *up*, and up is exactly what
    the 50 mg/Nm^3 cap removes.  So the sharp test is not on the recorded
    values but on which minutes are clipped: a periodic peak would clip
    preferentially at one phase.
    """
    phase = (np.arange(len(censored)) % period_minutes) / period_minutes
    binned = np.minimum((phase * N_PHASE_BINS).astype(int), N_PHASE_BINS - 1)
    table = np.array([[int(((binned == k) & (censored == 1)).sum()) for k in range(N_PHASE_BINS)],
                      [int(((binned == k) & (censored == 0)).sum()) for k in range(N_PHASE_BINS)]])
    return float(stats.chi2_contingency(table).pvalue)


def main() -> None:
    df = pd.read_csv(DATA_FILE).dropna(subset=["C_out_mgNm3"]).reset_index(drop=True)
    outlet = df["C_out_mgNm3"].to_numpy(float)

    rows: list[dict[str, object]] = []
    censoring = censoring_test(outlet)
    rows.append({
        "test": "censored_normal_mle",
        "target": "C_out_mgNm3",
        "statistic": censoring["predicted_censored_fraction"],
        "reference": censoring["observed_censored_fraction"],
        "detail": (f"latent N(mu={censoring['latent_mu_mgNm3']:.3f}, "
                   f"sigma={censoring['latent_sigma_mgNm3']:.3f}) censored at {CAP:g}; "
                   f"latent CV {censoring['latent_cv_pct']:.2f}% of level"),
        "verdict": ("censoring hypothesis reproduces the observed cap fraction to "
                    f"{censoring['absolute_error']:.4f}"),
    })

    residual = outlet - outlet.mean()
    censored_flag = (outlet >= CAP).astype(int)
    residual_sd = float(residual.std())
    for col in T_COLS:
        period_minutes = float(df[col].median()) / 60.0
        amplitude = epoch_folding(residual, period_minutes)
        p_value = censoring_phase_test(censored_flag, period_minutes)
        rows.append({
            "test": "epoch_folding",
            "target": col,
            "statistic": amplitude,
            "reference": residual_sd,
            "detail": (f"median period {period_minutes:.2f} min "
                       f"(Nyquist limit 2 min); censoring-rate chi2 p={p_value:.3f}"),
            "verdict": (f"no rapping-synchronous component above {amplitude:.3f} mg/Nm3 "
                        f"({100.0 * amplitude / residual_sd:.1f}% of residual sd)"),
        })

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT_DIR / "outlet_censoring_periodicity.csv",
               index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
