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
