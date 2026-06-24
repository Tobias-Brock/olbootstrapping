from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from olbootstrap.means._mean_estimator import MeanEstimator


@dataclass
class ExperimentResults:
    """Container for recorded bootstrap experiment outputs.

    Args:
        times (np.ndarray): Recorded time indices.
        samples (Optional[np.ndarray], optional): Original sample path.
        bootstrap_means (Optional[np.ndarray], optional): Mean of bootstrap
            replicates at recorded times.
        lower_bounds (Optional[np.ndarray], optional): Pointwise lower bounds.
        upper_bounds (Optional[np.ndarray], optional): Pointwise upper bounds.
        mu_t (Optional[np.ndarray], optional): True mean sequence, if available.
        smoother_target (Optional[np.ndarray], optional): Smoothed target sequence.
        replicates_by_time (Optional[List[np.ndarray]], optional): Bootstrap
            replicate arrays at recorded times.
        rep_sds (Optional[np.ndarray], optional): Bootstrap replicate standard
            deviations at recorded times.
        smooth_estimates (Optional[np.ndarray], optional): Smoothed point estimates.
        uniform_lower (Optional[np.ndarray], optional): Uniform-band lower bounds.
        uniform_upper (Optional[np.ndarray], optional): Uniform-band upper bounds.
        uniform_record_times (Optional[np.ndarray], optional): Times associated
            with uniform bounds.
        uniform_widths (Optional[np.ndarray], optional): Uniform-band widths.
        uniform_mean_width (Optional[float], optional): Mean uniform-band width.
    """

    times: np.ndarray
    samples: Optional[np.ndarray] = None
    bootstrap_means: Optional[np.ndarray] = None
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None
    mu_t: Optional[np.ndarray] = None
    smoother_target: Optional[np.ndarray] = None
    replicates_by_time: Optional[List[np.ndarray]] = None
    rep_sds: Optional[np.ndarray] = None
    smooth_estimates: Optional[np.ndarray] = None
    uniform_lower: Optional[np.ndarray] = None
    uniform_upper: Optional[np.ndarray] = None
    uniform_record_times: Optional[np.ndarray] = None
    uniform_widths: Optional[np.ndarray] = None
    uniform_mean_width: Optional[float] = None


class BaseBootstrapExperiment:
    """Common wiring for AR-multiplier bootstrap experiments."""

    def __init__(
        self,
        *,
        sample_size: int = 500,
        process=None,
        burn_in: int = 0,
        B: int = 200,
        record_every: int = 1,
        smoothing_method: Optional[str] = None,
        eta: float = 0.15,
        var_warmup: int = 0,
        use_variance_smoothing: bool = False,
        smoothing_beta: Optional[float] = None,
        gamma: Optional[float] = None,
        seasonal_period: Optional[int] = None,
        bootstrap_cls=None,
        progress: bool = False,
        forecast_s: int = 0,
        rng_weights: Optional[np.random.Generator] = None,
        transform: str = 'student',
        transform_power: float = 1.0 / 3.0,
        rho_power: float = (-1.0 / 3.0),
        alpha: float = 0.05,
    ):
        """Initialize the experiment runner.

        Args:
            sample_size: Number of time steps to process (default 500).
            process: Time-series generator with generate_samples(n) and
                optional _meta attributes.
            burn_in: Number of initial observations to exclude from eval.
            B: Number of bootstrap replicates.
            record_every: Record state every this many observations.
            smoothing_method: Smoother name for the target series.
            eta: Primary smoothing parameter passed to bootstrap/smoother.
            var_warmup: Warmup length for variance / threshold scheduling.
            use_variance_smoothing: Whether to apply variance smoothing.
            smoothing_beta: Secondary smoother parameter (Holt/HW).
            gamma: Seasonal/trend smoothing parameter for HW.
            seasonal_period: Seasonal period (for Holt-Winters).
            bootstrap_cls: Class implementing the online bootstrap interface.
            progress: If True show a progress bar during run().
            forecast_s: Forecast horizon passed to bootstrap (0 = contempor.).
            rng_weights: RNG for multiplier weights.
            transform: Multiplier transform, i.e. 'student' or 'gauss'.
            transform_power: Power mapping effective sample size to df.
            rho_power: Exponent used for latent AR correlation scaling.
            alpha: Nominal significance level for calibration/quantiles.
        """
        self.process = process
        self.t2 = None if sample_size is None else int(sample_size)
        self.t0 = int(burn_in)

        self.B = int(B)
        self.record_every = int(record_every)

        self.smoothing_method = smoothing_method
        self.eta = float(eta) if eta is not None else None
        self.var_warmup = int(var_warmup)
        self.use_variance_smoothing = bool(use_variance_smoothing)
        self.smoothing_beta = (
            float(smoothing_beta) if smoothing_beta is not None else None
        )
        self._gamma = float(gamma) if gamma is not None else None
        self.seasonal_period = (
            int(seasonal_period) if seasonal_period is not None else None
        )

        self.bootstrap_cls = bootstrap_cls
        self.progress = bool(progress)
        self.forecast_s = int(forecast_s)
        self.transform = str(transform)
        self.transform_power = float(transform_power)
        self.rho_power = float(rho_power)
        self.K = int(np.ceil(np.log2((self.t2 - self.t0) / float(self.var_warmup))))
        self.K = max(1, self.K)
        self.alpha = float(alpha)

        self.times: Optional[np.ndarray] = None
        self.empirical_means: Optional[np.ndarray] = None
        self.bootstrap_means: Optional[np.ndarray] = None
        self.lower_bounds: Optional[np.ndarray] = None
        self.upper_bounds: Optional[np.ndarray] = None
        self.uniform_widths: Optional[np.ndarray] = None
        self.uniform_mean_width: Optional[float] = None

        self.uniform_lower: Optional[np.ndarray] = None
        self.uniform_upper: Optional[np.ndarray] = None
        self.uniform_record_times: Optional[np.ndarray] = None
        self._uniform_thresholds: Optional[np.ndarray] = None
        self._uniform_q_active: Optional[np.ndarray] = None
        self._uniform_sigma_star: Optional[np.ndarray] = None

        self.mu_t: Optional[np.ndarray] = None
        self.samples: Optional[np.ndarray] = None
        self.smooth_estimates: Optional[np.ndarray] = None
        self.replicates_by_time: Optional[List[np.ndarray]] = None
        self.rep_sds: Optional[np.ndarray] = None

        self._last_obs: Optional[float] = None
        self._bootstrap = None
        self._rng_weights = rng_weights or np.random.default_rng()

    def _ensure_bootstrap(self) -> None:
        """Construct the bootstrap instance.

        Returns:
            None
        """
        if self._bootstrap is not None:
            return
        self._bootstrap = self.bootstrap_cls(
            smoothing_method=self.smoothing_method,
            eta=self.eta,
            var_warmup=self.var_warmup,
            use_variance_smoothing=self.use_variance_smoothing,
            smoothing_beta=self.smoothing_beta,
            gamma=self._gamma,
            seasonal_period=self.seasonal_period,
            forecast_s=self.forecast_s,
            rng=self._rng_weights,
            transform=self.transform,
            transform_power=self.transform_power,
            rho_power=self.rho_power,
            K=self.K,
            t0=self.t0,
            alpha=self.alpha,
        )

    def _smoother_target_series(self, testing: bool = False) -> np.ndarray:
        """Compute smoothed target series used for coverage checks.

        Args:
            testing: If True, return an all-zero target series (for tests).

        Returns:
            np.ndarray: Smoothed (or zero) target series (float array).
        """
        mu = np.asarray(self.mu_t, dtype=float)

        if testing:
            return np.zeros_like(mu, dtype=float)

        me = MeanEstimator(
            method=self.smoothing_method,
            eta=self.eta,
            beta=self.smoothing_beta,
            gamma=self._gamma,
            seasonal_period=self.seasonal_period,
            n_series=1,
        )

        target = np.empty_like(mu)
        for t, m in enumerate(mu, start=1):
            _level = me.update(m)
            target[t - 1] = (
                float(_level)
                if self.smoothing_method == 'ewma'
                else float(me.forecast(0))
            )
        return target

    def results(self) -> ExperimentResults:
        """Assemble and return an ExperimentResults object.

        Returns:
            ExperimentResults: Container with times, samples, bootstrap means,
            bounds, smoothed targets and uniform-band information.
        """
        smoother_target = None
        if self.smoothing_method is not None:
            try:
                smoother_target = self._smoother_target_series()
            except Exception:
                pass

        return ExperimentResults(
            times=np.asarray(self.times, dtype=int),
            samples=(
                None if self.samples is None else np.asarray(self.samples, dtype=float)
            ),
            bootstrap_means=(
                None
                if self.bootstrap_means is None
                else np.asarray(self.bootstrap_means, dtype=float)
            ),
            lower_bounds=(
                None
                if self.lower_bounds is None
                else np.asarray(self.lower_bounds, dtype=float)
            ),
            upper_bounds=(
                None
                if self.upper_bounds is None
                else np.asarray(self.upper_bounds, dtype=float)
            ),
            mu_t=(None if self.mu_t is None else np.asarray(self.mu_t, dtype=float)),
            smoother_target=(
                None
                if smoother_target is None
                else np.asarray(smoother_target, dtype=float)
            ),
            replicates_by_time=getattr(self, 'replicates_by_time', None),
            rep_sds=(
                None
                if getattr(self, 'rep_sds', None) is None
                else np.asarray(self.rep_sds, dtype=float)
            ),
            smooth_estimates=(
                None
                if self.smooth_estimates is None
                else np.asarray(self.smooth_estimates, dtype=float)
            ),
            uniform_lower=(
                None
                if self.uniform_lower is None
                else np.asarray(self.uniform_lower, dtype=float)
            ),
            uniform_upper=(
                None
                if self.uniform_upper is None
                else np.asarray(self.uniform_upper, dtype=float)
            ),
            uniform_record_times=(
                None
                if self.uniform_record_times is None
                else np.asarray(self.uniform_record_times, dtype=int)
            ),
            uniform_widths=(
                None
                if self.uniform_widths is None
                else np.asarray(self.uniform_widths, dtype=float)
            ),
            uniform_mean_width=(
                None
                if getattr(self, 'uniform_mean_width', None) is None
                else float(self.uniform_mean_width)
            ),
        )
