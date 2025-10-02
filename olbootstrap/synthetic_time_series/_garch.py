from typing import Optional, Sequence

import numpy as np

from ._base_process import BaseProcess


class GARCH11Process(BaseProcess):
    """GARCH(1,1) time-series generator with optional trend, seasonality, and shocks."""

    def __init__(
        self,
        mean: float,
        omega: float,
        alpha: float,
        beta: float,
        *,
        trend_slope: float = 0.0,
        seasonal_amplitude: float = 0.0,
        seasonal_period: Optional[float] = None,
        seasonal_phase: float = 0.0,
        noise_std: float = 1.0,
        noise_dist: str = 'normal',  # "normal" or "student"
        t_df: Optional[float] = None,  # used when noise_dist == "student"
        rng: Optional[np.random.Generator] = None,
        shock_type: str = 'none',
        jump_prob: float = 0.0,
        jump_scale: float = 1.0,
        decay: float = 0.95,
        jump_times: Optional[Sequence[int]] = None,
        jump_sizes: Optional[Sequence[float]] = None,
    ):
        """Initialize the GARCH11Process.

        Args:
            mean (float): Baseline constant mean.
            omega (float): GARCH omega (>0).
            alpha (float): GARCH alpha (>=0).
            beta (float): GARCH beta (>=0, with alpha+beta<1).
            trend_slope (float, optional): Linear trend slope. Defaults to 0.0.
            seasonal_amplitude (float, optional): Seasonal amplitude. Defaults to 0.0.
            seasonal_period (Optional[float], optional): Seasonal period. Defaults to None.
            seasonal_phase (float, optional): Seasonal phase. Defaults to 0.0.
            noise_std (float, optional): Base innovation scale. Defaults to 1.0.
            noise_dist (str, optional): 'normal' or 'student'. Defaults to 'normal'.
            t_df (Optional[float], optional): Degrees of freedom for Student-t innovations.
            rng (Optional[np.random.Generator], optional): Random generator. Defaults to None.
            shock_type (str, optional): One of {'none','permanent','transient'}. Defaults to 'none'.
            jump_prob (float, optional): Per-step jump probability. Defaults to 0.0.
            jump_scale (float, optional): Std. dev. for sampled jumps. Defaults to 1.0.
            decay (float, optional): Decay for transient shocks. Defaults to 0.95.
            jump_times (Optional[Sequence[int]], optional): Deterministic 1-based jump times.
            jump_sizes (Optional[Sequence[float]], optional): Deterministic jump sizes.

        Raises:
            ValueError: If omega<=0 or alpha<0 or beta<0 or alpha+beta>=1.
            ValueError: If noise_dist is invalid or t_df is invalid for Student-t.
        """
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
            raise ValueError(
                'Require ω>0, α>=0, β>=0, α+β<1 for covariance-stationary GARCH(1,1).'
            )

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

        self.omega = float(omega)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.noise_std = float(noise_std)

        nd = str(noise_dist).lower()
        if nd not in {'normal', 'student'}:
            raise ValueError("noise_dist must be 'normal' or 'student'")
        self.noise_dist = nd
        self.t_df = None if t_df is None else float(t_df)
        if self.noise_dist == 'student' and (self.t_df is None or self.t_df <= 0):
            raise ValueError("t_df > 0 required when noise_dist='student'")

        self.samples = np.array([])
        self._meta.sigma_seq = None  # will store σ_t

    def generate_samples(self, number_samples: int) -> np.ndarray:
        """Generate `number_samples` from the GARCH(1,1) process.

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

        trend_seq, seasonal_seq = self._init_common(n)

        if self.noise_dist == 'normal':
            z = self.rng.normal(loc=0.0, scale=1.0, size=n)
        else:
            z_raw = self.rng.standard_t(df=self.t_df, size=n)
            if self.t_df > 2:
                z = z_raw / np.sqrt(self.t_df / (self.t_df - 2.0))
            else:
                z = z_raw

        eps = z * self.noise_std
        sigma2 = np.empty(n, dtype=float)
        x = np.empty(n, dtype=float)
        self._maybe_apply_jump_t1()
        uncond_var = self.omega / max(1e-12, (1.0 - self.alpha - self.beta))
        sigma2[0] = uncond_var

        m0 = self.mean + trend_seq[0] + seasonal_seq[0] + self._shock_component()
        x[0] = m0 + np.sqrt(sigma2[0]) * eps[0]
        self._log_shock_state(0)

        for t in range(1, n):
            time1 = t + 1

            self._decay_transient_before_step()
            self._maybe_apply_jump(time1)
            m_t = self.mean + trend_seq[t] + seasonal_seq[t] + self._shock_component()
            shock_prev = (
                self._level_jumps[t - 1]
                if self.shock_type == 'permanent'
                else self._transient_seq[t - 1]
                if self.shock_type == 'transient'
                else 0.0
            )
            m_prev = self.mean + trend_seq[t - 1] + seasonal_seq[t - 1] + shock_prev
            resid_prev = x[t - 1] - m_prev

            sigma2[t] = (
                self.omega + self.alpha * (resid_prev**2) + self.beta * sigma2[t - 1]
            )
            x[t] = m_t + np.sqrt(sigma2[t]) * eps[t]

            self._log_shock_state(t)

        self.samples = x
        self._finalize_meta(
            n,
            trend_seq=trend_seq,
            seasonal_seq=seasonal_seq,
            extra_meta={
                'sigma_seq': np.sqrt(sigma2),
                'noise_dist': self.noise_dist,
                't_df': self.t_df,
            },
        )
        return self.samples

    @property
    def variance(self) -> Optional[float]:
        """Unconditional variance of the covariance-stationary GARCH(1,1).

        Returns:
            Optional[float]: Unconditional variance omega/(1-alpha-beta) if defined;
            otherwise None (e.g., Student-t df<=2 or non-stationary parameters).
        """
        if (
            getattr(self, 'noise_dist', 'normal') == 'student'
            and self.t_df is not None
            and self.t_df <= 2
        ):
            return None
        return self.omega / max(1e-12, (1.0 - self.alpha - self.beta))

    def instantaneous_variance(self, t: int) -> float:
        """Instantaneous conditional variance at time index `t`.

        Args:
            t (int): Zero-based time index (0 <= t < n_generated).

        Returns:
            float: Instantaneous variance (sigma_t^2) at time t.

        Raises:
            RuntimeError: If no samples have been generated yet.
            IndexError: If `t` is out of range.
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
        # If df<=2, variance is formally undefined; match AR1 behavior and return inf
        if (
            getattr(self, 'noise_dist', 'normal') == 'student'
            and self.t_df is not None
            and self.t_df <= 2
        ):
            return float('inf')
        return float(self._meta.sigma_seq[t] ** 2)
