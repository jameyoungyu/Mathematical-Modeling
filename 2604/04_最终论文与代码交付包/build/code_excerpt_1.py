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
