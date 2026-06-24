import math
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from olbootstrap.experiments._ess import effective_sample_size

from ._base_bootstrap import BaseARBootstrap


def _transform_multiplier(
    z: np.ndarray,
    nu_eff: float,
    *,
    kind: str = 'student',
    power: float = 1.0 / 3.0,
) -> np.ndarray:
    """Transform standard normal samples z into multipliers.

    Args:
        z (np.ndarray): Standard-normal samples.
        nu_eff (float): Effective sample size controlling tail heaviness.
        kind (str, optional): 'student' for Student-t inverse-CDF transform or
        'gauss' for identity. Defaults to 'student'.
        power (float, optional): Exponent used to map nu_eff to degrees of freedom.
        Defaults to 1/3.

    Returns:
        np.ndarray: Transformed multipliers with the same shape as z.
    """
    if kind == 'gauss':
        return z
    p = float(power)
    nu = max(1.0, 2.0 + (float(nu_eff) ** p))
    u = norm.cdf(z)
    v = student_t.ppf(u, df=nu)
    return v


def generate_recursive_weight_old(
    i: int,
    V_i: np.ndarray | float,
    alpha: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray | float:
    """Legacy recursive update of multipliers V_i.

    Args:
        i (int): Current time index (used to compute the recursion coefficient).
        V_i (np.ndarray | float): Current multiplier(s).
        alpha (float): Decay exponent used in r_i = 1 - i^{-alpha}.
        rng (np.random.Generator | None): Optional random generator.

    Returns:
        np.ndarray | float: Updated multiplier(s) V_{i+1} with the same shape as V_i.
    """
    rng = rng or np.random.default_rng()
    V_i = np.asarray(V_i, dtype=float)
    r = 1.0 - float(i) ** (-float(alpha))
    z = rng.normal(size=V_i.shape)
    return 1.0 + r * (V_i - 1.0) + math.sqrt(max(0.0, 1.0 - r * r)) * z


class OnlineARBootstrap(BaseARBootstrap):
    """Online AR-multiplier bootstrap for smoothed mean estimation."""

    def _update_latents_and_multipliers(
        self, B: int, nu_eff: float, rho: float
    ) -> np.ndarray:
        """Update latent AR(1) variables and return transformed multipliers.

        Args:
            B (int): Number of bootstrap replicates.
            nu_eff (float): Effective sample size for the latent transform.
            rho (float): Latent AR(1) correlation parameter.

        Returns:
            np.ndarray: Transformed multipliers of shape (B, 1).
        """
        if self._previous_latent is None or self._previous_latent.shape != (B, 1):
            self._previous_latent = self._rng.normal(loc=0.0, scale=1.0, size=(B, 1))
        eps = self._rng.normal(loc=0.0, scale=1.0, size=(B, 1))
        Z_new = rho * self._previous_latent + math.sqrt(1.0 - rho * rho) * eps
        self._previous_latent = Z_new
        return _transform_multiplier(
            Z_new.ravel(),
            nu_eff,
            kind=self._transform_kind,
            power=self._transform_power,
        ).reshape(B, 1)

    def _step_new_bootstrap(
        self,
        x_t: float,
        mu_center: float,
        B: int,
        *,
        nu_eff: float,
        rho: float,
    ) -> np.ndarray:
        """Compute bootstrap updates for the new-path multiplier bootstrap.

        Args:
            x_t (float): Observed value at time t.
            mu_center (float): Centering mean (usually the lagged estimate).
            B (int): Number of bootstrap replicates.
            nu_eff (float): Effective sample size for the latent transform.
            rho (float): Latent AR(1) correlation parameter.

        Returns:
            np.ndarray: Per-bootstrap deviations γ*_t (shape (B,)), i.e., the
            bootstrap increments to add to the current mean estimate.
        """
        V_t = self._update_latents_and_multipliers(B, nu_eff=nu_eff, rho=rho)
        X_star_t = V_t * (x_t - mu_center)

        self._offset_estimator.update(X_star_t)
        gamma_vec = np.asarray(
            self._instantaneous_from_estimator(self._offset_estimator), dtype=float
        ).reshape(B)

        return gamma_vec

    def __call__(
        self, new_samples: np.ndarray, number_bootstrap_samples: Optional[int] = None
    ):
        """Process incoming samples online and update bootstrap state in-place.

        Args:
            new_samples (np.ndarray): Incoming observations, shape (n_batch,) or
            (n_batch, n_series).
            number_bootstrap_samples (Optional[int]): If provided and bootstrap is not
            yet initialized, use this value to initialize B.
        """
        if new_samples.ndim == 1:
            new_samples = new_samples.reshape(-1, 1)

        if self._bootstrap_averages is None:
            B = int(number_bootstrap_samples)
            self._bootstrap_averages = np.zeros((B, 1))
            self._previous_latent = self._rng.normal(loc=0.0, scale=1.0, size=(B, 1))
            self._average_samples = 0.0
            self._ensure_estimators(B)

            self._B1 = max(1, B // 5)
            self._B2 = B - self._B1
            self._sigma2_star = 0.0
            self._m_running = np.zeros(B, dtype=float)

        B = self._bootstrap_averages.shape[0]

        nu_eff = effective_sample_size(
            smoothing_method=self._smoothing_method,
            eta=self.eta,
            beta=self._beta,
            gamma=self._gamma,
            seasonal_period=self.seasonal_period,
        )
        nu_eff = max(1.0, float(nu_eff))
        rho = 1.0 - (nu_eff**self._rho_power)

        for _, sample in enumerate(new_samples):
            self._index_time = 0 if (self._index_time is None) else self._index_time
            self._index_time += 1
            t = self._index_time
            x_t = float(sample.ravel()[0])

            if t <= int(self.t0):
                mu_center = (
                    float(self._last_mu_point)
                    if (self._last_mu_point is not None)
                    else 0.0
                )

                delta_star_t = self._step_new_bootstrap(
                    x_t=x_t, mu_center=mu_center, B=B, nu_eff=nu_eff, rho=rho
                )

                self._data_estimator.update(x_t)
                mu_point_now = float(
                    self._instantaneous_from_estimator(self._data_estimator)
                )
                self._last_mu_point = mu_point_now
                self._bootstrap_averages = (mu_point_now + delta_star_t).reshape(B, 1)
                self._delta_star = delta_star_t.copy()
                self._sigma_star = float('nan')  # calibration not active yet

                continue

            mu_center = (
                float(self._last_mu_point) if (self._last_mu_point is not None) else 0.0
            )

            delta_star_t = self._step_new_bootstrap(
                x_t=x_t, mu_center=mu_center, B=B, nu_eff=nu_eff, rho=rho
            )

            v_hat = float(np.var(delta_star_t[: self._B1], ddof=0))

            if self.use_variance_smoothing:
                w = 1.0 - 1.0 / nu_eff
                self._sigma2_star = max(
                    0.0, w * v_hat + (1.0 - w) * float(self._sigma2_star)
                )
            else:
                self._sigma2_star = max(0.0, v_hat)

            sigma_star = math.sqrt(max(self._sigma2_star, 1e-12))
            self._sigma_star = sigma_star
            self._delta_star = delta_star_t.copy()

            std_abs_tail = np.abs(delta_star_t[self._B1 :]) / sigma_star
            self._m_running[self._B1 :] = np.maximum(
                self._m_running[self._B1 :], std_abs_tail
            )

            if self._next_boundary_idx < len(self._boundaries):
                boundary_t = self._boundaries[self._next_boundary_idx]
                if t == boundary_t:
                    calib_slice = (
                        self._m_running[self._B1 :]
                        if (self._B2 > 0)
                        else self._m_running
                    )
                    q = float(
                        np.quantile(
                            calib_slice,
                            1.0 - self._alpha / float(self._K),
                        )
                    )
                    self._q_history[int(boundary_t)] = q
                    self._q_active = q
                    self._next_boundary_idx += 1

            self._data_estimator.update(x_t)
            mu_point_now = float(
                self._instantaneous_from_estimator(self._data_estimator)
            )
            self._last_mu_point = mu_point_now
            mu_star_t = mu_point_now + delta_star_t
            self._bootstrap_averages = mu_star_t.reshape(B, 1)
