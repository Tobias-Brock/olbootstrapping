from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from olbootstrap.means._mean_estimator import MeanEstimator


class BaseARBootstrap(ABC):
    """Base class for online autoregressive bootstrap."""

    def __init__(
        self,
        *,
        average_samples: Optional[np.ndarray] = None,
        index_time: Optional[int] = None,
        previous_latent: Optional[np.ndarray] = None,
        bootstrap_averages: Optional[np.ndarray] = None,
        average_weights: Optional[np.ndarray] = None,
        smoothing_method: Optional[str] = None,
        var_warmup: int = 0,
        eta: float = 0.02,
        smoothing_beta: Optional[float] = None,
        gamma: Optional[float] = None,
        seasonal_period: Optional[int] = None,
        forecast_s: int = 0,
        old_bootstrap: bool = False,
        rng: np.random.Generator | None = None,
        transform: str = 'student',  # "student" or "gauss"
        use_variance_smoothing: bool = True,
        transform_power: float = 1.0 / 3.0,
        K: int = None,
        t0: int = None,
        alpha: float = 0.05,
    ):
        """Initialize shared state for AR bootstrap implementations.

        Args:
            average_samples: Optional array of past average samples used to
                initialize estimators. Shape depends on caller convention.
            index_time: Optional current time index (integer).
            previous_latent: Optional array containing previous latent bootstrap
                paths / states.
            bootstrap_averages: Optional array to hold bootstrap averages (B,).
            average_weights: Optional weights used in averaging.
            smoothing_method: Smoother identifier (e.g. 'ewma', 'holt', 'brown').
            var_warmup: Number of warmup steps (integer) used for variance logic.
            eta: Smoothing rate used by estimators.
            smoothing_beta: Optional secondary smoothing parameter (e.g. Holt beta).
            gamma: Optional seasonal smoothing parameter.
            seasonal_period: Optional seasonal period (int) for seasonal smoothers.
            forecast_s: Forecast horizon used when querying estimators.
            old_bootstrap: If True, use the legacy bootstrap method.
            rng: NumPy random generator (defaults to ``np.random.default_rng()``).
            transform: Which reference transform to use, e.g. 'student' or 'gauss'.
            use_variance_smoothing: Whether to apply variance smoothing updates.
            transform_power: Power used in transform-related scaling (float).
            K: Number of dyadic boundary steps (int).
            t0: Starting time index for calibration window logic.
            alpha: Nominal significance level used for quantile computations.
        """
        self._bootstrap_averages = bootstrap_averages
        self._average_samples = average_samples
        self._average_weights = average_weights
        self._previous_latent = previous_latent
        self._index_time = index_time
        self._last_mu_point = None
        self.t0 = t0
        self._var_warmup = max(0, int(var_warmup))
        self._transform_power = float(transform_power)

        self._old_bootstrap = bool(old_bootstrap)
        self._forecast_horizon = int(forecast_s)
        self._rng = rng or np.random.default_rng()
        self.use_variance_smoothing = bool(use_variance_smoothing)
        self.seasonal_period = (
            int(seasonal_period) if seasonal_period is not None else None
        )

        self._smoothing_method = (
            smoothing_method.lower() if isinstance(smoothing_method, str) else None
        )
        self.eta = float(eta)
        self._beta = float(smoothing_beta) if smoothing_beta is not None else None
        self._gamma = float(gamma) if gamma is not None else None

        self._transform_kind = str(transform).lower()

        self._K = int(K)
        self._alpha = float(alpha)

        self._delta_star: Optional[np.ndarray] = None  # (B,)
        self._sigma_star: Optional[float] = None
        self._m_running: Optional[np.ndarray] = None  # (B,)
        self._q_active: Optional[float] = None
        self._q_history: Dict[int, float] = {}

        self._den_running_mean: Optional[np.ndarray] = None
        self._prev_multiplier: Optional[np.ndarray] = None

        if np.any(
            [
                average_samples is None,
                index_time is None,
                previous_latent is None,
                bootstrap_averages is None,
                average_weights is None,
            ]
        ):
            self._index_time = 0
            self._bootstrap_averages = None
            self._average_samples = None
            self._average_weights = None
            self._previous_latent = None

        self._boundaries = [
            int(self.t0 + self._var_warmup * (2**k)) for k in range(int(self._K))
        ]
        self._next_boundary_idx = 0

    def _ensure_estimators(self, B: int):
        """Create the MeanEstimator instances.

        Args:
            B: Number of bootstrap replicates used for per-bootstrap estimators
               (used to size the offset/num/den estimators).

        Raises:
            ValueError: If ``self._smoothing_method`` is None or invalid.
        """
        if self._smoothing_method is None:
            raise ValueError(
                "A smoothing_method ('ewma'|'holt'|'brown'|'holtwinters') is required."
            )

        self._data_estimator = MeanEstimator(
            method=self._smoothing_method,
            eta=self.eta,
            beta=self._beta,
            gamma=self._gamma,
            seasonal_period=self.seasonal_period,
            n_series=1,
        )

        self._var_estimator = MeanEstimator(
            method=self._smoothing_method,
            eta=self.eta,
            beta=self._beta,
            gamma=self._gamma,
            seasonal_period=self.seasonal_period,
            n_series=1,
        )

        self._offset_estimator = MeanEstimator(
            method=self._smoothing_method,
            eta=self.eta,
            beta=self._beta,
            gamma=self._gamma,
            seasonal_period=self.seasonal_period,
            n_series=B,
        )

        if self._old_bootstrap:
            self._num_estimator = MeanEstimator(
                method=self._smoothing_method,
                eta=self.eta,
                beta=self._beta,
                gamma=self._gamma,
                seasonal_period=self.seasonal_period,
                n_series=B,
            )
            self._den_estimator = MeanEstimator(
                method=self._smoothing_method,
                eta=self.eta,
                beta=self._beta,
                gamma=self._gamma,
                seasonal_period=self.seasonal_period,
                n_series=B,
            )

    def _instantaneous_from_estimator(self, est: MeanEstimator):
        """Return the instantaneous/fitted point from `est` (level or forecast)."""
        return est.forecast(m=self._forecast_horizon)

    @abstractmethod
    def _step_new_bootstrap(
        self, x_t: float, mu_point: float, B: int, *, nu_eff: float, rho: float
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute (mu_star_t, delta_star_t, sigma_star) for the new path."""
        raise NotImplementedError

    @abstractmethod
    def _step_old_bootstrap(
        self, t: int, x_t: float, mu_point: float, B: int
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Compute (mu_star_t, delta_star_t, sigma_star) for the old path."""
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        new_samples: np.ndarray,
        number_bootstrap_samples: Optional[int] = None,
    ) -> None:
        """Process a batch of new samples online and update the bootstrap state."""
        raise NotImplementedError

    @property
    def bootstrap_averages(self) -> np.ndarray:
        """Get the current bootstrap averages.

        Returns:
            The array of bootstrap averages (ndarray) or None if not initialized.
        """
        return self._bootstrap_averages

    @property
    def mu_point(self) -> float:
        """Return the most-recent smoothed point estimate.

        Returns:
            The last smoothed point as a float, or ``nan`` if not set.
        """
        return (
            float(self._last_mu_point)
            if self._last_mu_point is not None
            else float('nan')
        )

    @property
    def sigma_star(self) -> Optional[float]:
        """Return the current scale estimate for the bootstrap.

        Returns:
            The current sigma* value as a float, or None if not available.
        """
        return None if self._sigma_star is None else float(self._sigma_star)

    @property
    def q_active(self) -> Optional[float]:
        """Return the currently active studentized max statistic.

        Returns:
            The active q statistic (float) or None if not set.
        """
        return None if self._q_active is None else float(self._q_active)
