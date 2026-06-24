from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.special import lambertw

from olbootstrap.experiments._ess import effective_sample_size

from ._base_bootstrap import MeanEstimator


class OnlineGaussianMixtureAsympCSSmoothedBootstrap:
    """Gaussian-mixture martingale AsympCS baseline centered on smoother."""

    def __init__(
        self,
        smoothing_method: str = None,
        eta: float = None,
        smoothing_beta: float = None,
        gamma: Optional[float] = None,
        seasonal_period: int = None,
        forecast_s: int = 0,  # unused but kept for signature compatibility
        rng: np.random.Generator = None,
        use_variance_smoothing: bool = False,
        alpha: float = 0.05,
        var_warmup: int = 0,
        t0: int = 0,
        transform: Optional[str] = None,  # unused but kept for compatibility
        transform_power: float = 1.0 / 3.0,  # unused but kept for compatibility
        **_ignored,
    ):
        """Initialize the Gaussian-mixture asymptotic CS baseline.

        Args:
            smoothing_method (str, optional): Smoother name. Defaults to "ewma".
            eta (float, optional): Primary smoothing parameter. Defaults to 0.1.
            smoothing_beta (float, optional): Trend smoothing parameter.
            gamma (Optional[float], optional): Seasonal smoothing parameter.
            seasonal_period (int, optional): Seasonal period for seasonal smoothers.
            forecast_s (int, optional): Forecast horizon kept for API compatibility.
            rng (np.random.Generator, optional): Random number generator.
            use_variance_smoothing (bool, optional): Whether to smooth residual
                variance estimates. Defaults to False.
            alpha (float, optional): Nominal significance level. Defaults to 0.05.
            var_warmup (int, optional): Variance warmup length. Defaults to 0.
            t0 (int, optional): Burn-in time. Defaults to 0.
            transform (Optional[str], optional): Unused compatibility argument.
            transform_power (float, optional): Unused compatibility argument.
            **_ignored: Additional compatibility keyword arguments.
        """
        self._rng = rng or np.random.default_rng()
        self.alpha = float(alpha)
        self.t0 = int(t0 or 0)
        self._var_warmup = int(var_warmup or 0)
        self.use_variance_smoothing = bool(use_variance_smoothing)

        if smoothing_method is None:
            smoothing_method = 'ewma'
        if eta is None:
            eta = 0.1

        self._smoothing_method = str(smoothing_method)
        self._eta = float(eta)
        self._beta = None if smoothing_beta is None else float(smoothing_beta)
        self._gamma_smoother = None if gamma is None else float(gamma)
        self.seasonal_period = seasonal_period

        self._mean_est = MeanEstimator(
            method=self._smoothing_method,
            eta=self._eta,
            beta=self._beta,
            seasonal_period=self.seasonal_period,
            gamma=self._gamma_smoother,
        )

        nu_eff = effective_sample_size(
            smoothing_method=self._smoothing_method,
            eta=self._eta,
            beta=self._beta,
            gamma=self._gamma_smoother,
            seasonal_period=self.seasonal_period,
        )
        self._nu_eff = float(max(1.0, nu_eff))
        t_star = max(1, int(self.t0) + int(self._var_warmup))
        self._rho_mixture = 1.0
        self.set_rho_opt(t_star)

        self._t = 0
        self._res_mean = 0.0
        self._res_M2 = 0.0
        self._sigma2_res = 0.0
        self._last_mu = None

        self._B = None
        self._bootstrap_averages = None

        self._sigma_star = np.nan
        self._q_active = np.nan
        self._last_mu_point = np.nan

    def set_rho_opt(self, t_star: int) -> float:
        """Set the Waudby-Smith mixture parameter using Lambert W tuning.

        Args:
            t_star (int): Effective target time used in the tuning formula.

        Returns:
            float: Tuned mixture parameter.
        """
        t_star = int(max(1, t_star))
        a = float(self.alpha)
        z = -(a**2) * np.exp(-1.0)
        w = lambertw(z, k=-1)
        rho2 = float(-w.real - 1.0) / float(t_star)
        rho2 = max(rho2, 1e-12)
        self._rho_mixture = float(np.sqrt(rho2))
        return self._rho_mixture

    @property
    def mu_point(self) -> float:
        """Return the latest smoothed point estimate.

        Returns:
            float: Most recent smoothed mean estimate.
        """
        return float(self._last_mu_point)

    @property
    def bootstrap_averages(self) -> np.ndarray:
        """Return the current bootstrap replicate averages.

        Returns:
            np.ndarray: Bootstrap replicate values with shape `(B, 1)`.
        """
        return self._bootstrap_averages

    @property
    def sigma_star(self) -> float:
        """Return the current scale estimate.

        Returns:
            float: Current scale estimate, or NaN before calibration is active.
        """
        return float(self._sigma_star)

    @property
    def q_active(self) -> float:
        """Return the current active critical value.

        Returns:
            float: Current active studentized critical value, or NaN before it is
                available.
        """
        return float(self._q_active)

    def __call__(
        self, new_samples: np.ndarray, number_bootstrap_samples: Optional[int] = None
    ):
        """Process new samples and update Gaussian-mixture bootstrap state.

        Args:
            new_samples (np.ndarray): Incoming observations.
            number_bootstrap_samples (Optional[int], optional): Number of
                bootstrap replicates to initialize or reset.

        Returns:
            None
        """
        x_arr = np.asarray(new_samples, dtype=float).reshape(-1)

        if number_bootstrap_samples is not None:
            self._B = int(number_bootstrap_samples)
        if self._B is None:
            self._B = 200

        if self._bootstrap_averages is None:
            self._bootstrap_averages = np.zeros((self._B, 1), dtype=float)

        burn_eff = int(self.t0) + int(self._var_warmup)

        for x in x_arr:
            self._t += 1
            t = self._t

            mu_center = (
                float(self._last_mu) if (self._last_mu is not None) else float(x)
            )
            r = float(x - mu_center)

            delta = r - self._res_mean
            self._res_mean += delta / t
            delta2 = r - self._res_mean
            self._res_M2 += delta * delta2
            s2_raw = max(0.0, float(self._res_M2 / t))

            if self.use_variance_smoothing:
                w = 1.0 - 1.0 / self._nu_eff
                self._sigma2_res = max(0.0, w * self._sigma2_res + (1.0 - w) * s2_raw)
            else:
                self._sigma2_res = s2_raw

            mu_hat = float(self._mean_est.update(x))
            self._last_mu = mu_hat
            self._last_mu_point = mu_hat

            se = (
                float(np.sqrt(self._sigma2_res / self._nu_eff))
                if self._nu_eff > 0
                else 0.0
            )
            rho = float(self._rho_mixture)
            t_eff = float(self._nu_eff)

            v = t_eff * self._sigma2_res * (rho**2) + 1.0
            log_arg = np.sqrt(v) / float(self.alpha)
            log_arg = max(log_arg, 1.0 + 1e-12)

            half_width = float(
                np.sqrt((2.0 * v) / (t_eff * t_eff * rho * rho) * np.log(log_arg))
            )

            if se > 0.0 and np.isfinite(se):
                reps = mu_hat + se * self._rng.standard_normal(self._B)
            else:
                reps = np.full(self._B, mu_hat, dtype=float)

            self._bootstrap_averages = reps.reshape(self._B, 1)

            # expose q_active/sigma_star only after burn-in+warmup
            if t <= burn_eff or not np.isfinite(half_width) or half_width <= 0.0:
                self._sigma_star = np.nan
                self._q_active = np.nan
            else:
                self._sigma_star = se if (se > 0.0 and np.isfinite(se)) else 1.0
                self._q_active = (
                    half_width / self._sigma_star if self._sigma_star > 0 else np.nan
                )
