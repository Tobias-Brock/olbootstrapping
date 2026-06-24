from __future__ import annotations

from typing import Optional

import numpy as np

from olbootstrap.experiments._ess import effective_sample_size

from ._base_bootstrap import MeanEstimator


class BlockBootstrap:
    """Batch moving-block bootstrap baseline for smoothed means."""

    def __init__(
        self,
        smoothing_method: str = None,
        eta: float = None,
        smoothing_beta: float = None,
        gamma: Optional[float] = None,
        seasonal_period: int = None,
        forecast_s: int = 0,  # unused, kept for signature compatibility
        rng: np.random.Generator = None,
        alpha: float = 0.05,
        t0: int = 0,
        var_warmup: int = 0,
        sample_size: Optional[int] = None,
        K: Optional[int] = None,
        block_length: Optional[int] = None,
        uniform: bool = True,
        recompute_every: int = 1,
        use_variance_smoothing: bool = False,
        transform: Optional[str] = None,  # unused, kept for compatibility
        transform_power: float = 1.0 / 3.0,  # unused, kept for compatibility
        **_ignored,
    ):
        """Initialize the moving-block bootstrap baseline.

        Args:
            smoothing_method (str, optional): Smoother name. Defaults to "ewma".
            eta (float, optional): Primary smoothing parameter. Defaults to 0.1.
            smoothing_beta (float, optional): Trend smoothing parameter.
            gamma (Optional[float], optional): Seasonal smoothing parameter.
            seasonal_period (int, optional): Seasonal period for seasonal smoothers.
            forecast_s (int, optional): Unused compatibility forecast horizon.
            rng (np.random.Generator, optional): Random number generator.
            alpha (float, optional): Nominal significance level. Defaults to 0.05.
            t0 (int, optional): Burn-in time. Defaults to 0.
            var_warmup (int, optional): Variance warmup length. Defaults to 0.
            sample_size (Optional[int], optional): Planned full sample size.
            K (Optional[int], optional): Fixed uniform correction factor.
            block_length (Optional[int], optional): Fixed moving-block length.
            uniform (bool, optional): If True, use uniform calibration.
            recompute_every (int, optional): Number of updates between block
                bootstrap recomputations. Defaults to 1.
            use_variance_smoothing (bool, optional): Whether to smooth the scale
                estimate. Defaults to False.
            transform (Optional[str], optional): Unused compatibility argument.
            transform_power (float, optional): Unused compatibility argument.
            **_ignored: Additional compatibility keyword arguments.
        """
        self._rng = rng or np.random.default_rng()
        self.alpha = float(alpha)
        self.t0 = int(t0 or 0)
        self._var_warmup = int(var_warmup or 0)

        if sample_size is None:
            sample_size = _ignored.get('n', None)
        if sample_size is None:
            sample_size = _ignored.get('t2', None)

        self._sample_size = None if sample_size is None else int(sample_size)
        self._K_fixed = None if K is None else int(max(1, K))

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

        self.block_length = None if block_length is None else int(max(1, block_length))

        self.uniform = bool(uniform)
        self.recompute_every = int(max(1, recompute_every))

        self.use_variance_smoothing = bool(use_variance_smoothing)
        self._sigma2_star_smoothed = None

        self._t = 0
        self._x_history: list[float] = []
        self._mu_center_history: list[float] = []
        self._mu_history: list[float] = []

        self._B = None
        self._B1 = None
        self._B2 = None

        self._bootstrap_averages = None
        self._last_bootstrap_errors = None

        self._sigma_star = np.nan
        self._q_active = np.nan
        self._last_mu_point = np.nan

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
        self,
        new_samples: np.ndarray,
        number_bootstrap_samples: Optional[int] = None,
    ):
        """Process new samples and update moving-block bootstrap state.

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

        if self._B1 is None or self._B2 is None:
            self._B1 = max(2, self._B // 5)
            self._B2 = self._B - self._B1
            if self._B2 <= 0:
                self._B1 = self._B
                self._B2 = 0

        burn_eff = int(self.t0) + int(self._var_warmup)

        for x in x_arr:
            self._t += 1
            t = self._t
            x_float = float(x)

            self._x_history.append(x_float)

            # Match the online AR bootstrap:
            # use the previous smoothed estimate as the residual center.
            mu_center = (
                float(self._last_mu_point) if np.isfinite(self._last_mu_point) else 0.0
            )
            self._mu_center_history.append(mu_center)

            # The actual point estimate is computed after defining the center.
            mu_hat = float(self._mean_est.update(x_float))
            self._last_mu_point = mu_hat
            self._mu_history.append(mu_hat)

            if t <= burn_eff:
                self._sigma_star = np.nan
                self._q_active = np.nan
                self._bootstrap_averages = np.full((self._B, 1), mu_hat)
                continue

            should_recompute = (
                self._last_bootstrap_errors is None or t % self.recompute_every == 0
            )

            if should_recompute:
                errors, sigma_star, q_active = self._compute_block_bootstrap()
                self._last_bootstrap_errors = errors
                self._sigma_star = sigma_star
                self._q_active = q_active

            if self._last_bootstrap_errors is not None:
                reps = mu_hat + self._last_bootstrap_errors
                self._bootstrap_averages = reps.reshape(-1, 1)
            else:
                self._bootstrap_averages = np.full((self._B, 1), mu_hat)

    def _current_block_length(self, n: int) -> int:
        """Return the current block length.

        Args:
            n (int): Number of observations currently available.

        Returns:
            int: Moving-block length clipped to `[1, n]`.
        """
        if n <= 0:
            return 1

        if self.block_length is not None:
            return int(min(max(1, self.block_length), n))

        default_length = int(np.ceil(4 * (float(self._nu_eff) ** (1.0 / 3.0))))
        return int(min(n, max(1, default_length)))

    def _calibration_bounds(self, n: int) -> tuple[int, int]:
        """Return zero-based calibration bounds for uniform calibration.

        This mimics the dyadic calibration schedule of the online AR bootstrap.
        The calibration window starts at t0 and grows at boundaries

            t0 + var_warmup * 2^k.

        At time n, we use the largest completed boundary not exceeding n.

        Args:
            n (int): Number of observations currently available.

        Returns:
            tuple[int, int]: Inclusive start and exclusive end indices for the
                calibration window.
        """
        if n <= 0:
            return 0, 0

        cal_start = min(max(0, self.t0), n - 1)

        if self._var_warmup <= 0:
            return cal_start, n

        K = self._current_K(n)
        boundaries = [int(self.t0 + self._var_warmup * (2**k)) for k in range(K)]

        completed = [b for b in boundaries if b <= n]

        if len(completed) == 0:
            cal_end = min(n, self.t0 + self._var_warmup)
        else:
            cal_end = max(completed)

        cal_end = min(max(cal_start + 1, cal_end), n)

        return cal_start, cal_end

    def _current_K(self, n: int) -> int:
        """Return the uniform correction factor K.

        Args:
            n (int): Number of observations currently available.

        Returns:
            int: Current dyadic uniform correction factor.
        """
        if self._K_fixed is not None:
            return self._K_fixed

        if self._var_warmup <= 0:
            return 1

        horizon_end = self._sample_size if self._sample_size is not None else n

        numerator = max(1, int(horizon_end) - int(self.t0))
        denominator = max(1, int(self._var_warmup))

        return int(max(1, np.ceil(np.log2(numerator / denominator))))

    def _quantile_level(self, n: int) -> float:
        """Return the quantile level used for the active critical value.

        Args:
            n (int): Number of observations currently available.

        Returns:
            float: Quantile level for pointwise or uniform calibration.
        """
        if not self.uniform:
            return float(1.0 - self.alpha)

        K = self._current_K(n)
        return float(1.0 - self.alpha / float(K))

    def _compute_block_bootstrap(self) -> tuple[np.ndarray, float, float]:
        """Compute block-bootstrap errors, scale, and active critical value.

        Returns:
            tuple[np.ndarray, float, float]: Final bootstrap errors, scale
                estimate, and active critical value.

        Raises:
            RuntimeError: If internal history arrays have inconsistent lengths.
        """
        x = np.asarray(self._x_history, dtype=float)
        mu_center_path = np.asarray(self._mu_center_history, dtype=float)
        n = x.size

        if mu_center_path.size != n:
            raise RuntimeError(
                'Internal error: x_history and mu_center_history have different lengths.'
            )

        # Match the online AR bootstrap centering:
        # residual_t = X_t - \hat{\mu}_eta(t - 1)
        residuals = x - mu_center_path

        # Remove finite-sample residual bias before resampling blocks.
        residuals = residuals - np.mean(residuals)

        block_length = self._current_block_length(n)
        boot_series = self._draw_circular_moving_blocks(
            y=residuals,
            B=self._B,
            block_length=block_length,
        )
        boot_paths = self._smooth_paths(boot_series)

        B1 = int(self._B1)
        B2 = int(self._B2)

        scale_paths = boot_paths[:B1]
        calib_paths = boot_paths[B1:] if B2 > 0 else boot_paths

        errors_final = calib_paths[:, -1]

        sigma_path = np.std(scale_paths, axis=0, ddof=1)
        sigma_final = float(sigma_path[-1])

        if self.use_variance_smoothing:
            if self._sigma2_star_smoothed is None:
                self._sigma2_star_smoothed = sigma_final**2
            else:
                w = 1.0 - 1.0 / self._nu_eff
                self._sigma2_star_smoothed = (
                    w * self._sigma2_star_smoothed + (1.0 - w) * sigma_final**2
                )
            sigma_final = float(np.sqrt(max(0.0, self._sigma2_star_smoothed)))

        eps = 1e-12

        if sigma_final <= eps or not np.isfinite(sigma_final):
            return errors_final, np.nan, np.nan

        if self.uniform:
            cal_start, cal_end = self._calibration_bounds(n)

            denom = np.maximum(sigma_path[cal_start:cal_end], eps)
            stats = np.max(
                np.abs(calib_paths[:, cal_start:cal_end]) / denom[None, :],
                axis=1,
            )
        else:
            stats = np.abs(errors_final) / max(sigma_final, eps)

        stats = stats[np.isfinite(stats)]

        if stats.size == 0:
            q_active = np.nan
        else:
            q_level = float(np.clip(self._quantile_level(n), 0.0, 1.0))
            # q_active = float(np.quantile(stats, q_level))
            q_active = self._upper_empirical_quantile(stats, q_level)

        return errors_final, sigma_final, q_active

    def _draw_circular_moving_blocks(
        self,
        y: np.ndarray,
        B: int,
        block_length: int,
    ) -> np.ndarray:
        """Draw circular moving-block bootstrap samples.

        Args:
            y (np.ndarray): Residual series to resample.
            B (int): Number of bootstrap replicates.
            block_length (int): Length of each circular moving block.

        Returns:
            np.ndarray: Bootstrap residual samples with shape `(B, len(y))`.
        """
        n = y.size
        L = int(min(max(1, block_length), n))
        n_blocks = int(np.ceil(n / L))

        starts = self._rng.integers(0, n, size=(B, n_blocks))
        offsets = np.arange(L)

        idx = (starts[:, :, None] + offsets[None, None, :]) % n
        idx = idx.reshape(B, -1)[:, :n]

        return y[idx]

    def _smooth_paths(self, samples: np.ndarray) -> np.ndarray:
        """Apply the same smoother to each bootstrap residual path.

        This uses `MeanEstimator` directly with `n_series=B`, so the bootstrap
        paths use exactly the same smoothing implementation and initialization
        behavior as the original point estimate.

        Args:
            samples (np.ndarray): Bootstrap residual samples with shape `(B, n)`.

        Returns:
            np.ndarray: Smoothed bootstrap residual paths with shape `(B, n)`.
        """
        B, n = samples.shape
        paths = np.empty((B, n), dtype=float)

        est = MeanEstimator(
            method=self._smoothing_method,
            eta=self._eta,
            beta=self._beta,
            n_series=B,
            seasonal_period=self.seasonal_period,
            gamma=self._gamma_smoother,
        )

        for i in range(n):
            values = est.update(samples[:, i])
            paths[:, i] = np.asarray(values, dtype=float).reshape(B)

        return paths

    def _upper_empirical_quantile(self, values: np.ndarray, level: float) -> float:
        """Return the upper empirical quantile at a given level.

        Args:
            values (np.ndarray): Values from which to compute the quantile.
            level (float): Quantile level in `[0, 1]`.

        Returns:
            float: Upper empirical quantile, or NaN if no finite values exist.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]

        if values.size == 0:
            return np.nan

        values = np.sort(values)
        idx = int(np.ceil(level * values.size)) - 1
        idx = min(max(idx, 0), values.size - 1)
        return float(values[idx])
