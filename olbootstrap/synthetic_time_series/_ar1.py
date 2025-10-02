from typing import Optional, Sequence

import numpy as np

from ._base_process import BaseProcess


class AR1Process(BaseProcess):
    """AR(1) time-series generator with optional trend, seasonality, and shocks."""

    def __init__(
        self,
        mean: float,
        phi: float,
        *,
        noise_std: float = 1.0,
        noise_dist: str = 'normal',  # "normal" or "student"
        t_df: Optional[float] = None,  # used when noise_dist == "student"
        trend_slope: float = 0.0,
        seasonal_amplitude: float = 0.0,
        seasonal_period: Optional[float] = None,
        seasonal_phase: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        shock_type: str = 'none',
        jump_prob: float = 0.0,
        jump_scale: float = 1.0,
        decay: float = 0.95,
        jump_times: Optional[Sequence[int]] = None,
        jump_sizes: Optional[Sequence[float]] = None,
    ):
        """Initialize the AR(1) process.

        Args:
            mean (float): Baseline constant mean.
            phi (float): AR(1) coefficient.
            noise_std (float, optional): Innovation standard deviation.
                Defaults to 1.0.
            noise_dist (str, optional): Innovation distribution, either 'normal' or
                'student'. Defaults to 'normal'.
            t_df (Optional[float], optional): Degrees of freedom for Student-t
                innovations (required if noise_dist == 'student').
            trend_slope (float, optional): Linear trend slope. Defaults to 0.0.
            seasonal_amplitude (float, optional): Amplitude of seasonal component.
                Defaults to 0.0.
            seasonal_period (Optional[float], optional): Period of seasonality.
                Required if seasonal_amplitude != 0. Defaults to None.
            seasonal_phase (float, optional): Phase shift for seasonal component.
                Defaults to 0.0.
            rng (Optional[np.random.Generator], optional): Random number generator.
                Defaults to np.random.default_rng().
            shock_type (str, optional): One of {'none','permanent','transient'}.
                Defaults to 'none'.
            jump_prob (float, optional): Probability of a random jump at each time
                step. Defaults to 0.0.
            jump_scale (float, optional): Std. dev. of sampled jump sizes.
                Defaults to 1.0.
            decay (float, optional): Decay factor for transient shocks (only used
                when shock_type == 'transient'). Defaults to 0.95.
            jump_times (Optional[Sequence[int]], optional): Deterministic 1-based
                jump times. Defaults to None.
            jump_sizes (Optional[Sequence[float]], optional): Deterministic jump
                sizes aligned with jump_times. Defaults to None.

        Raises:
            ValueError: If `noise_dist` is not 'normal' or 'student', or if
                `noise_dist == 'student'` and `t_df` is not provided or not > 0.
        """
        super().__init__(
            mean=mean,
            trend_slope=trend_slope,
            seasonal_amplitude=seasonal_amplitude,
            seasonal_period=seasonal_period,
            seasonal_phase=seasonal_phase,
            rng=rng,
            shock_type=shock_type,
            jump_prob=jump_prob,
            jump_scale=jump_scale,
            decay=decay,
            jump_times=jump_times,
            jump_sizes=jump_sizes,
        )
        self.phi = float(phi)
        self.noise_std = float(noise_std)

        noise_dist = str(noise_dist).lower()
        if noise_dist not in {'normal', 'student'}:
            raise ValueError("noise_dist must be 'normal' or 'student'")
        self.noise_dist = noise_dist
        self.t_df = None if t_df is None else float(t_df)
        if self.noise_dist == 'student' and (self.t_df is None or self.t_df <= 0):
            raise ValueError("t_df > 0 required when noise_dist='student'")

        self.samples = np.array([])
        self._meta.sigma_seq = None
        self._meta.noise_dist = self.noise_dist
        self._meta.t_df = self.t_df

    def generate_samples(self, number_samples: int) -> np.ndarray:
        """Generate `number_samples` from the AR(1) process.

        Args:
            number_samples (int): Number of observations to generate.

        Returns:
            np.ndarray: Generated time series of length `number_samples`.
        """
        n = int(number_samples)
        if n <= 0:
            self.samples = np.zeros(0, dtype=float)
            _ = self._finalize_meta(0, trend_seq=np.zeros(0), seasonal_seq=np.zeros(0))
            return self.samples

        sigma_seq = np.full(n, float(self.noise_std), dtype=float)
        trend_seq, seasonal_seq = self._init_common(n)

        if self.noise_dist == 'normal':
            base = self.rng.normal(loc=0.0, scale=1.0, size=n)  # unit-variance
            eps = base * sigma_seq  # std = noise_std
        else:
            base_t = self.rng.standard_t(df=self.t_df, size=n)
            if self.t_df > 2:
                std_correction = np.sqrt(self.t_df / (self.t_df - 2.0))
                base_unitvar = base_t / std_correction
                eps = base_unitvar * sigma_seq  # std = noise_std
            else:
                eps = base_t * self.noise_std

        x = np.empty(n, dtype=float)

        # t=1 jump (before)
        self._maybe_apply_jump_t1()
        m0 = self.mean + trend_seq[0] + seasonal_seq[0] + self._shock_component()
        x[0] = m0 + eps[0]
        self._log_shock_state(0)

        for t in range(1, n):
            time1 = t + 1
            self._decay_transient_before_step()
            self._maybe_apply_jump(time1)

            m_t = self.mean + trend_seq[t] + seasonal_seq[t] + self._shock_component()
            # previous mean used for AR centering:
            shock_prev = (
                self._level_jumps[t - 1]
                if self.shock_type == 'permanent'
                else self._transient_seq[t - 1]
                if self.shock_type == 'transient'
                else 0.0
            )
            m_prev = self.mean + trend_seq[t - 1] + seasonal_seq[t - 1] + shock_prev

            x[t] = m_t + self.phi * (x[t - 1] - m_prev) + eps[t]
            self._log_shock_state(t)

        self.samples = x
        self._finalize_meta(
            n,
            trend_seq=trend_seq,
            seasonal_seq=seasonal_seq,
            extra_meta={
                'sigma_seq': sigma_seq,
                'noise_dist': self.noise_dist,
                't_df': self.t_df,
            },
        )
        return self.samples

    @property
    def variance(self) -> Optional[float]:
        """Unconditional variance (if |phi|<1) under constant innovation scale."""
        if (
            self._meta.n_generated == 0
            or getattr(self._meta, 'sigma_seq', None) is None
        ):
            return None
        if abs(self.phi) >= 1.0:
            return None

        if (
            getattr(self, 'noise_dist', 'normal') == 'student'
            and (self.t_df is not None)
            and (self.t_df <= 2)
        ):
            return None  # or float("inf")

        avg_sig2 = float(self.noise_std**2)  # we scaled to this std when df>2
        return avg_sig2 / max(1e-12, 1.0 - self.phi**2)

    def instantaneous_variance(self, t: int) -> float:
        """Instantaneous unconditional variance at time index `t`.

        Args:
            t (int): Zero-based time index (0 <= t < n_generated).

        Returns:
            float: Instantaneous unconditional variance at time t.

        Raises:
            RuntimeError: If no samples have been generated yet.
            IndexError: If `t` is out of the generated range.
        """
        if (
            self._meta.n_generated == 0
            or getattr(self._meta, 'sigma_seq', None) is None
        ):
            raise RuntimeError(
                'No samples generated yet; cannot provide instantaneous variance.'
            )
        if t < 0 or t >= self._meta.n_generated:
            raise IndexError('t out of range')
        if abs(self.phi) >= 1.0:
            return float('inf')

        if (
            getattr(self, 'noise_dist', 'normal') == 'student'
            and (self.t_df is not None)
            and (self.t_df <= 2)
        ):
            return float('inf')  # variance undefined

        sig2 = float(self._meta.sigma_seq[t] ** 2)  # equals noise_std**2
        return sig2 / max(1e-12, 1.0 - self.phi**2)
