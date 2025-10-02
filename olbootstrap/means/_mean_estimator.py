from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np


class MeanEstimator:
    """Online mean estimator supporting EWMA, Holt, Brown, and Holt–Winters."""

    def __init__(
        self,
        method: str = 'ewma',
        eta: float = 0.1,
        beta: Optional[float] = None,
        n_series: int = 1,
        init_level: Optional[Union[float, np.ndarray]] = None,
        init_trend: Optional[Union[float, np.ndarray]] = None,
        seasonal_period: Optional[int] = None,
        gamma: Optional[float] = None,
        init_seasonal: Optional[Union[np.ndarray, list]] = None,
        init_mode: Literal['simple', 'seasonal_mean'] = 'simple',
        eps: float = 1e-8,  # numerical safety for multiplicative model
    ):
        """Initialize a MeanEstimator.

        Args:
            method (str): One of 'ewma', 'holt', 'brown', 'holtwinters' (or 'hw').
            eta (float): Smoothing parameter in (0, 1].
            beta (Optional[float]): Trend smoothing (used for 'holt', optional).
            n_series (int): Number of parallel series (1 for scalar).
            init_level (Optional[float|np.ndarray]): Initial level(s).
            init_trend (Optional[float|np.ndarray]): Initial trend(s) (for Holt/HW).
            Seasonal period (Optional[int]): Seasonal period (required for 'holtwinters').
            gamma (Optional[float]): Seasonal smoothing for Holt–Winters.
            init_seasonal (Optional[array-like]): Initial seasonal indices.
            init_mode (Literal['simple','seasonal_mean']): Seasonal init mode.
            eps (float): Small epsilon for numerical safety (HW multiplicative).

        Raises:
            ValueError: If `method` is invalid, `eta` not in (0,1], `n_series` not pos int,
                        or if HW-specific arguments are invalid (m, gamma, init_seasonal).
        """
        method = method.lower()
        if method not in ('ewma', 'holt', 'brown', 'holtwinters', 'hw'):
            raise ValueError(
                "method must be one of 'ewma','holt','brown','holtwinters','hw'"
            )
        if not (0 < eta <= 1.0):
            raise ValueError('eta must be in (0,1].')

        self.method = 'holtwinters' if method == 'hw' else method
        self.eta = float(eta)

        if n_series < 1 or int(n_series) != n_series:
            raise ValueError('n_series must be a positive integer')
        self.n_series = int(n_series)
        shape = (self.n_series,)

        if self.method in ('holt', 'holtwinters'):
            self.beta = (
                float(beta)
                if (beta is not None)
                else (max(1e-6, self.eta / 4.0) if self.method == 'holt' else None)
            )
        else:
            self.beta = None

        if init_level is None:
            self.level = np.zeros(shape, dtype=float)
            self._initialized_level = False
        else:
            arr = np.asarray(init_level, dtype=float)
            self.level = arr.reshape(shape) if arr.shape != shape else arr.copy()
            self._initialized_level = True

        if self.method == 'holt':
            if init_trend is None:
                self.trend = np.zeros(shape, dtype=float)
                self._initialized_trend = False
            else:
                arr = np.asarray(init_trend, dtype=float)
                self.trend = arr.reshape(shape) if arr.shape != shape else arr.copy()
                self._initialized_trend = True
            self._prev_level = self.level.copy()

        elif self.method == 'brown':
            self.s1 = np.zeros(shape, dtype=float)
            self.s2 = np.zeros(shape, dtype=float)
            self.trend = np.zeros(shape, dtype=float)

        elif self.method == 'holtwinters':
            if (
                seasonal_period is None
                or int(seasonal_period) != seasonal_period
                or seasonal_period < 2
            ):
                raise ValueError('m (seasonal period) must be an integer >= 2')
            self.seasonal_period = int(seasonal_period)

            self.gamma = (
                float(gamma) if gamma is not None else max(1e-6, self.eta / 2.0)
            )
            if not (0 < self.gamma <= 1.0):
                raise ValueError('gamma must be in (0,1]')

            if init_trend is None:
                self.trend = np.zeros(shape, dtype=float)
                self._initialized_trend = False
            else:
                arr = np.asarray(init_trend, dtype=float)
                self.trend = arr.reshape(shape) if arr.shape != shape else arr.copy()
                self._initialized_trend = True

            if init_seasonal is not None:
                S = np.asarray(init_seasonal, dtype=float)
                if S.ndim == 1:
                    if S.size != self.seasonal_period:
                        raise ValueError('init_seasonal length must equal m')
                    S = np.tile(S[None, :], (self.n_series, 1))
                elif S.shape != (self.n_series, self.seasonal_period):
                    raise ValueError(
                        'init_seasonal must have shape (m,) or (n_series, m)'
                    )
                self.seasonal_indices = S
                self._initialized_seasonal = True
            else:
                base = 0.0
                self.seasonal_indices = np.full(
                    (self.n_series, self.seasonal_period), base, dtype=float
                )
                self._initialized_seasonal = init_mode == 'simple'

            self.init_mode = init_mode
            self._t_seen = 0
            self._season_buf = np.full(
                (self.seasonal_period, self.n_series), np.nan, dtype=float
            )
            self._eps = float(eps)

        else:  # "ewma"
            self.trend = np.zeros(shape, dtype=float)
            self._initialized_trend = False  # unused

    def _initialize_hw_from_first_season(self):
        """Initialize Holt–Winters seasonal indices from first full season buffer.

        Returns:
            bool: True if initialization succeeded (season buffer complete), else False.
        """
        assert self.method == 'holtwinters'
        if not np.all(np.isfinite(self._season_buf)):
            return False

        y_first = self._season_buf  # shape (m, n_series)
        ybar = np.mean(y_first, axis=0)  # (n_series,)
        S = (y_first - ybar[None, :]).T  # (n_series, m)

        self.level = ybar.copy()
        self.trend = np.zeros_like(self.level)
        self.seasonal_indices = S
        self._initialized_level = True
        self._initialized_trend = True
        self._initialized_seasonal = True
        return True

    def update(self, x: Union[float, np.ndarray]) -> np.ndarray:
        """Ingest one observation (or vector of length `n_series`) and update state.

        Args:
            x (float | np.ndarray): Scalar observation applied to all series, or an
                array-like of length `n_series` providing parallel observations.

        Returns:
            np.ndarray or float: Updated level (scalar if `n_series==1`, else array).

        Raises:
            ValueError: If `x` is neither scalar nor length `n_series`.
        """
        x_arr = np.asarray(x, dtype=float)
        if x_arr.shape == ():
            obs = np.full((self.n_series,), x_arr)
        elif x_arr.size == self.n_series:
            obs = x_arr.reshape((self.n_series,))
        else:
            raise ValueError('x must be scalar or have length n_series')

        a = self.eta

        if self.method == 'ewma':
            if not getattr(self, '_initialized_level', False):
                self.level = obs.copy()
                self._initialized_level = True
            else:
                self.level = a * obs + (1.0 - a) * self.level

        elif self.method == 'holt':
            b = self.beta
            if not getattr(self, '_initialized_level', False):
                self.level = obs.copy()
                self.trend = np.zeros_like(obs)
                self._initialized_level = True
                self._initialized_trend = True
            else:
                prev_level = self.level.copy()
                self.level = a * obs + (1.0 - a) * (self.level + self.trend)
                self.trend = b * (self.level - prev_level) + (1.0 - b) * self.trend

        elif self.method == 'brown':
            if not getattr(self, '_initialized_level', False):
                self.s1 = obs.copy()
                self.s2 = obs.copy()
                self.level = (2.0 * self.s1 - self.s2).copy()
                denom = (1.0 - a) if (1.0 - a) != 0 else np.finfo(float).eps
                self.trend = (a / denom) * (self.s1 - self.s2)
                self._initialized_level = True
            else:
                self.s1 = a * obs + (1.0 - a) * self.s1
                self.s2 = a * self.s1 + (1.0 - a) * self.s2
                self.level = 2.0 * self.s1 - self.s2
                denom = (1.0 - a) if (1.0 - a) != 0 else np.finfo(float).eps
                self.trend = (a / denom) * (self.s1 - self.s2)

        else:  # "holtwinters"
            b = self.beta if self.beta is not None else self.eta / 4.0
            g = self.gamma
            self._t_seen += 1
            idx = (self._t_seen - 1) % self.seasonal_period

            if self.init_mode == 'seasonal_mean' and not self._initialized_seasonal:
                self._season_buf[idx, :] = obs
                if self._t_seen == self.seasonal_period:
                    ok = self._initialize_hw_from_first_season()
                    if not ok:
                        # fallback to simple
                        pass

            if not getattr(self, '_initialized_level', False):
                self.level = obs.copy()
                self._initialized_level = True
            if not getattr(self, '_initialized_trend', False):
                self.trend = np.zeros_like(obs)
                self._initialized_trend = True
            if not getattr(self, '_initialized_seasonal', False):
                pass

            S_prev = self.seasonal_indices[:, idx]  # (n_series,)

            Lt = a * (obs - S_prev) + (1.0 - a) * (self.level + self.trend)
            Bt = b * (Lt - self.level) + (1.0 - b) * self.trend
            St_new = g * (obs - Lt) + (1.0 - g) * S_prev

            self.level = Lt
            self.trend = Bt
            self.seasonal_indices[:, idx] = St_new

        return float(self.level) if self.n_series == 1 else self.level.copy()

    @property
    def estimate(self):
        """Return current level estimate.

        Returns:
            float or np.ndarray: Scalar if `n_series==1`, otherwise array of
            length `n_series`.
        """
        return float(self.level) if self.n_series == 1 else self.level.copy()

    @property
    def current_trend(self):
        """Return current trend estimate.

        Returns:
            float or np.ndarray: Scalar if `n_series==1`, otherwise array of
            length `n_series`.
        """
        return float(self.trend) if self.n_series == 1 else self.trend.copy()

    def forecast(self, m: int = 0):
        """Forecast `m` steps ahead.

        Args:
            m (int): Non-negative integer horizon.

        Returns:
            float or np.ndarray: Forecast value(s). For EWMA returns estimate,
                for Holt/Brown returns level + m*trend, for Holt–Winters adds seasonal
                phase.

        Raises:
            ValueError: If `m` is negative or not integer.
        """
        if m < 0 or int(m) != m:
            raise ValueError('m must be a non-negative integer')

        if self.method in ['ewma']:
            return self.estimate

        if self.method in ('holt', 'brown'):
            val = self.level + m * self.trend
            return float(val) if self.n_series == 1 else val.copy()

        if self._t_seen == 0:
            # no seasonal phase yet; fall back to non-seasonal
            base = self.level + m * self.trend
            return float(base) if self.n_series == 1 else base.copy()

        idx = ((self._t_seen - 1) + m) % self.seasonal_period
        S = self.seasonal_indices[:, idx]  # (n_series,)
        base = self.level + m * self.trend
        val = base + S

        return float(val) if self.n_series == 1 else val.copy()

    def reset(self):
        """Reset internal states to uninitialized/zero.

        Returns:
            None
        """
        shape = (self.n_series,)
        self.level = np.zeros(shape, dtype=float)
        self._initialized_level = False
        self.trend = np.zeros(shape, dtype=float)

        if self.method == 'holt':
            self._initialized_trend = False

        if self.method == 'brown':
            self.s1 = np.zeros(shape, dtype=float)
            self.s2 = np.zeros(shape, dtype=float)

        if self.method == 'holtwinters':
            base = 0.0
            seasonal_period = getattr(self, 'seasonal_period', 2)
            self.seasonal_indices = np.full(
                (self.n_series, seasonal_period), base, dtype=float
            )
            self._initialized_seasonal = (
                getattr(self, 'init_mode', 'simple') == 'simple'
            )
            self._t_seen = 0
            self._season_buf = np.full(
                (seasonal_period, self.n_series), np.nan, dtype=float
            )
