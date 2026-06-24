from typing import Optional, Sequence

import numpy as np

from ._base_process import BaseProcess


class GARCH11Process(BaseProcess):
    """GARCH(1,1) time-series generator with optional nonlinear sine dynamics.

    The default process is

        X_t = m_t + sigma_t z_t,

    with

        sigma_t^2 = omega * noise_std^2
                    + alpha * eps_{t-1}^2
                    + beta * sigma_{t-1}^2.

    If `nonlinear_sin=True`, the centered process follows

        Y_t = a sin(Y_{t-1}) + sigma_t z_t,

    and the observed process is

        X_t = m_t + Y_t.

    This gives a nonlinear and conditionally heteroskedastic DGP.
    """

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
        noise_dist: str = 'normal',
        t_df: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        shock_type: str = 'none',
        jump_prob: float = 0.0,
        jump_scale: float = 1.0,
        decay: float = 0.95,
        jump_times: Optional[Sequence[int]] = None,
        jump_sizes: Optional[Sequence[float]] = None,
        nonlinear_sin: bool = True,
        nonlinear_coef: float = 1.0,
    ):
        """Initialize the GARCH(1,1) process.

        Args:
            mean (float): Baseline constant mean.
            omega (float): GARCH intercept before scaling by `noise_std**2`.
            alpha (float): GARCH ARCH coefficient. Must satisfy alpha >= 0.
            beta (float): GARCH persistence coefficient. Must satisfy beta >= 0
                and alpha + beta < 1.
            trend_slope (float, optional): Linear trend slope. Defaults to 0.0.
            seasonal_amplitude (float, optional): Seasonal amplitude.
                Defaults to 0.0.
            seasonal_period (Optional[float], optional): Seasonal period.
                Required if seasonal_amplitude != 0. Defaults to None.
            seasonal_phase (float, optional): Seasonal phase. Defaults to 0.0.
            noise_std (float, optional): Overall innovation scale. Defaults to 1.0.
            noise_dist (str, optional): Innovation distribution, either 'normal'
                or 'student'. Defaults to 'normal'.
            t_df (Optional[float], optional): Degrees of freedom for Student-t
                innovations. Required if noise_dist == 'student'.
            rng (Optional[np.random.Generator], optional): Random number generator.
                Defaults to np.random.default_rng().
            shock_type (str, optional): One of {'none', 'permanent', 'transient'}.
                Defaults to 'none'.
            jump_prob (float, optional): Probability of a random jump at each time
                step. Defaults to 0.0.
            jump_scale (float, optional): Std. dev. of sampled jump sizes.
                Defaults to 1.0.
            decay (float, optional): Decay factor for transient shocks. Only used
                when shock_type == 'transient'. Defaults to 0.95.
            jump_times (Optional[Sequence[int]], optional): Deterministic 1-based
                jump times. Defaults to None.
            jump_sizes (Optional[Sequence[float]], optional): Deterministic jump
                sizes aligned with jump_times. Defaults to None.
            nonlinear_sin (bool, optional): If True, include nonlinear sine
                dynamics in the centered process. Defaults to False.
            nonlinear_coef (float, optional): Coefficient multiplying the sine
                term. Defaults to 1.0.

        Raises:
            ValueError: If GARCH parameters are invalid, if `noise_std <= 0`,
                if `noise_dist` is invalid, or if Student-t degrees of freedom
                are invalid.
        """
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
            raise ValueError(
                'Require omega > 0, alpha >= 0, beta >= 0, '
                'and alpha + beta < 1 for covariance-stationary GARCH(1,1).'
            )
        if noise_std <= 0:
            raise ValueError('noise_std must be positive.')

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

        noise_dist = str(noise_dist).lower()
        if noise_dist not in {'normal', 'student'}:
            raise ValueError("noise_dist must be 'normal' or 'student'")
        self.noise_dist = noise_dist

        self.t_df = None if t_df is None else float(t_df)
        if self.noise_dist == 'student' and (self.t_df is None or self.t_df <= 0):
            raise ValueError("t_df > 0 required when noise_dist='student'")

        self.nonlinear_sin = bool(nonlinear_sin)
        self.nonlinear_coef = float(nonlinear_coef)

        self.samples = np.array([])
        self._meta.sigma_seq = None
        self._meta.noise_dist = self.noise_dist
        self._meta.t_df = self.t_df
        self._meta.garch_order = (1, 1)
        self._meta.omega = self.omega
        self._meta.alpha = self.alpha
        self._meta.beta = self.beta
        self._meta.noise_std = self.noise_std
        self._meta.nonlinear_sin = self.nonlinear_sin
        self._meta.nonlinear_coef = self.nonlinear_coef

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
            _ = self._finalize_meta(
                0,
                trend_seq=np.zeros(0),
                seasonal_seq=np.zeros(0),
            )
            return self.samples

        trend_seq, seasonal_seq = self._init_common(n)
        z = self._generate_unit_variance_noise(n)

        sigma2 = np.empty(n, dtype=float)
        y = np.empty(n, dtype=float)
        x = np.empty(n, dtype=float)
        mean_seq = np.empty(n, dtype=float)

        scaled_omega = self.omega * self.noise_std**2
        uncond_var = scaled_omega / max(1e-12, 1.0 - self.alpha - self.beta)

        for t in range(n):
            time1 = t + 1

            if t == 0:
                self._maybe_apply_jump_t1()
                sigma2[t] = uncond_var
                nonlinear_mean = 0.0
            else:
                self._decay_transient_before_step()
                self._maybe_apply_jump(time1)

                resid_prev = y[t - 1]
                sigma2[t] = (
                    scaled_omega
                    + self.alpha * resid_prev**2
                    + self.beta * sigma2[t - 1]
                )

                if self.nonlinear_sin:
                    nonlinear_mean = self.nonlinear_coef * np.sin(y[t - 1])
                else:
                    nonlinear_mean = 0.0

            mean_seq[t] = (
                self.mean + trend_seq[t] + seasonal_seq[t] + self._shock_component()
            )

            y[t] = nonlinear_mean + np.sqrt(max(0.0, sigma2[t])) * z[t]
            x[t] = mean_seq[t] + y[t]

            self._log_shock_state(t)

        self.samples = x
        self._finalize_meta(
            n,
            trend_seq=trend_seq,
            seasonal_seq=seasonal_seq,
            extra_meta={
                'sigma_seq': np.sqrt(np.maximum(0.0, sigma2)),
                'noise_dist': self.noise_dist,
                't_df': self.t_df,
                'garch_order': (1, 1),
                'omega': self.omega,
                'alpha': self.alpha,
                'beta': self.beta,
                'noise_std': self.noise_std,
                'nonlinear_sin': self.nonlinear_sin,
                'nonlinear_coef': self.nonlinear_coef,
            },
        )
        return self.samples

    def _generate_unit_variance_noise(self, n: int) -> np.ndarray:
        """Generate noise with unit variance whenever variance is finite.

        Args:
            n (int): Number of noise draws.

        Returns:
            np.ndarray: Noise sequence of length `n`.
        """
        if self.noise_dist == 'normal':
            return self.rng.normal(loc=0.0, scale=1.0, size=n)

        z_raw = self.rng.standard_t(df=self.t_df, size=n)

        if self.t_df > 2:
            return z_raw / np.sqrt(self.t_df / (self.t_df - 2.0))

        return z_raw

    @property
    def variance(self) -> Optional[float]:
        """Unconditional variance of the covariance-stationary GARCH(1,1).

        Returns:
            Optional[float]: Approximate unconditional variance if defined.
                Returns None for Student-t innovations with df <= 2.
        """
        if (
            getattr(self, 'noise_dist', 'normal') == 'student'
            and self.t_df is not None
            and self.t_df <= 2
        ):
            return None

        scaled_omega = self.omega * self.noise_std**2
        return scaled_omega / max(1e-12, 1.0 - self.alpha - self.beta)

    def instantaneous_variance(self, t: int) -> float:
        """Instantaneous conditional variance at time index `t`.

        Args:
            t (int): Zero-based time index.

        Returns:
            float: Instantaneous conditional variance sigma_t^2 at time t.

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

        if (
            getattr(self, 'noise_dist', 'normal') == 'student'
            and self.t_df is not None
            and self.t_df <= 2
        ):
            return float('inf')

        return float(self._meta.sigma_seq[t] ** 2)
