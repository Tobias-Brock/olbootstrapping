from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._base_process import BaseProcess


class AR1Process(BaseProcess):
    """AR(p) time-series generator."""

    def __init__(
        self,
        mean: float,
        phi: Union[float, Sequence[float]],
        *,
        noise_std: float = 1.0,
        noise_dist: str = 'normal',
        t_df: Optional[float] = None,
        trend_slope: float = 0.0,
        quadratic_trend: bool = False,
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
        structural_break: bool = False,
        break_time: Optional[int] = None,
        structural_break_times: Optional[Sequence[int]] = None,
        n_structural_breaks: int = 0,
        break_margin: Optional[int] = None,
        min_break_distance: Optional[int] = None,
        structural_break_jump_sizes: Optional[Union[float, Sequence[float]]] = None,
        positive_trend_noise_scale: float = 0.7,
        negative_trend_noise_scale: float = 1.5,
    ):
        """Initialize the AR(p) process.

        Args:
            mean (float): Baseline constant mean.
            phi (Union[float, Sequence[float]]): Autoregressive coefficient(s).
                If a scalar is provided, an AR(1) process is generated. If a
                sequence is provided, its length determines the AR order p.
            noise_std (float, optional): Innovation standard deviation.
                Defaults to 1.0.
            noise_dist (str, optional): Innovation distribution, either 'normal'
                or 'student'. Defaults to 'normal'.
            t_df (Optional[float], optional): Degrees of freedom for Student-t
                innovations. Required if noise_dist == 'student'.
            trend_slope (float, optional): Trend coefficient. If
                `quadratic_trend=False`, this gives a linear trend
                `trend_slope * t`. If `quadratic_trend=True`, this gives a
                quadratic trend `trend_slope * t**2`. Defaults to 0.0.
            quadratic_trend (bool, optional): If True, use a quadratic trend
                term instead of a linear trend term. Defaults to False.
            seasonal_amplitude (float, optional): Amplitude of seasonal component.
                Defaults to 0.0.
            seasonal_period (Optional[float], optional): Period of seasonality.
                Required if seasonal_amplitude != 0. Defaults to None.
            seasonal_phase (float, optional): Phase shift for seasonal component.
                Defaults to 0.0.
            rng (Optional[np.random.Generator], optional): Random number generator.
                Defaults to np.random.default_rng().
            shock_type (str, optional): One of {'none', 'permanent', 'transient'}.
                Defaults to 'none'.
            jump_prob (float, optional): Probability of a random jump at each time
                step. Defaults to 0.0.
            jump_scale (float, optional): Std. dev. of sampled jumps.
                Defaults to 1.0.
            decay (float, optional): Decay factor for transient shocks. Only used
                when shock_type == 'transient'. Defaults to 0.95.
            jump_times (Optional[Sequence[int]], optional): Deterministic 1-based
                jump times handled by the base shock mechanism. Defaults to None.
            jump_sizes (Optional[Sequence[float]], optional): Deterministic jump
                sizes aligned with jump_times. Defaults to None.
            structural_break (bool, optional): If True, flips trend sign and
                noise regime at structural break time(s). Defaults to False.
            break_time (Optional[int], optional): Backward-compatible single
                1-based structural break time. Used only if
                `structural_break_times` is None and `n_structural_breaks == 0`.
                Defaults to None.
            structural_break_times (Optional[Sequence[int]], optional): Explicit
                1-based structural break times. Takes priority over random
                break sampling. Defaults to None.
            n_structural_breaks (int, optional): Number of random structural
                break times to sample if no explicit break times are provided.
                Defaults to 0.
            break_margin (Optional[int], optional): Minimum distance from the
                start and end of the series when randomly sampling break times.
                If None, defaults to n // 10. Defaults to None.
            min_break_distance (Optional[int], optional): Minimum distance
                between randomly sampled break times. Defaults to None.
            structural_break_jump_sizes (Optional[float | Sequence[float]], optional):
                Optional level jump(s) applied directly at structural break
                times. If scalar, the same jump is used at every break. If a
                sequence is provided, its length must match the number of
                structural breaks. Defaults to None.
            positive_trend_noise_scale (float, optional): Multiplicative noise
                scale used when the local trend coefficient is nonnegative.
                Defaults to 0.7.
            negative_trend_noise_scale (float, optional): Multiplicative noise
                scale used when the local trend coefficient is negative.
                Defaults to 1.5.

        Raises:
            ValueError: If inputs are invalid.
        """
        if noise_std <= 0:
            raise ValueError('noise_std must be positive.')
        if positive_trend_noise_scale <= 0:
            raise ValueError('positive_trend_noise_scale must be positive.')
        if negative_trend_noise_scale <= 0:
            raise ValueError('negative_trend_noise_scale must be positive.')
        if n_structural_breaks < 0:
            raise ValueError('n_structural_breaks must be nonnegative.')

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

        self.phi = np.atleast_1d(np.asarray(phi, dtype=float))
        if self.phi.ndim != 1 or self.phi.size == 0:
            raise ValueError('phi must be a float or a non-empty 1D sequence.')

        self.p = int(self.phi.size)
        self.noise_std = float(noise_std)
        self.quadratic_trend = bool(quadratic_trend)

        noise_dist = str(noise_dist).lower()
        if noise_dist not in {'normal', 'student'}:
            raise ValueError("noise_dist must be 'normal' or 'student'")
        self.noise_dist = noise_dist

        self.t_df = None if t_df is None else float(t_df)
        if self.noise_dist == 'student' and (self.t_df is None or self.t_df <= 0):
            raise ValueError("t_df > 0 required when noise_dist='student'")

        self.structural_break = bool(structural_break)
        self.break_time = None if break_time is None else int(break_time)
        self.structural_break_times = (
            None
            if structural_break_times is None
            else [int(t) for t in structural_break_times]
        )
        self.n_structural_breaks = int(n_structural_breaks)
        self.break_margin = None if break_margin is None else int(break_margin)
        self.min_break_distance = (
            None if min_break_distance is None else int(min_break_distance)
        )
        self.structural_break_jump_sizes = structural_break_jump_sizes

        self.positive_trend_noise_scale = float(positive_trend_noise_scale)
        self.negative_trend_noise_scale = float(negative_trend_noise_scale)

        self.samples = np.array([])
        self._meta.sigma_seq = None
        self._meta.noise_dist = self.noise_dist
        self._meta.t_df = self.t_df
        self._meta.ar_order = self.p
        self._meta.phi = self.phi.copy()
        self._meta.structural_break = self.structural_break
        self._meta.quadratic_trend = self.quadratic_trend

    def generate_samples(self, number_samples: int) -> np.ndarray:
        """Generate `number_samples` from the AR(p) process.

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

        sigma_seq = np.full(n, float(self.noise_std), dtype=float)
        trend_seq, seasonal_seq = self._init_common(n)

        if not self.structural_break:
            trend_seq = self._apply_trend_shape(n=n, trend_seq=trend_seq)

        break_indices = self._resolve_break_indices(n)
        trend_seq, sigma_seq = self._apply_structural_break(
            trend_seq=trend_seq,
            sigma_seq=sigma_seq,
            break_indices=break_indices,
        )

        eps = self._generate_innovations(n=n, sigma_seq=sigma_seq)

        x = np.empty(n, dtype=float)
        mean_seq = np.empty(n, dtype=float)

        for t in range(n):
            time1 = t + 1

            if t == 0:
                self._maybe_apply_jump_t1()
            else:
                self._decay_transient_before_step()
                self._maybe_apply_jump(time1)

            mean_seq[t] = (
                self.mean + trend_seq[t] + seasonal_seq[t] + self._shock_component()
            )

            ar_part = 0.0
            max_lag = min(self.p, t)

            for lag in range(1, max_lag + 1):
                ar_part += self.phi[lag - 1] * (x[t - lag] - mean_seq[t - lag])

            x[t] = mean_seq[t] + ar_part + eps[t]
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
                'ar_order': self.p,
                'phi': self.phi.copy(),
                'quadratic_trend': self.quadratic_trend,
                'structural_break': self.structural_break,
                'break_times': [int(idx + 1) for idx in break_indices],
                'n_structural_breaks': self.n_structural_breaks,
                'break_margin': self.break_margin,
                'min_break_distance': self.min_break_distance,
                'structural_break_jump_sizes': (
                    None
                    if self.structural_break_jump_sizes is None
                    else np.asarray(
                        self.structural_break_jump_sizes,
                        dtype=float,
                    ).copy()
                ),
                'positive_trend_noise_scale': self.positive_trend_noise_scale,
                'negative_trend_noise_scale': self.negative_trend_noise_scale,
            },
        )
        return self.samples

    def _apply_trend_shape(self, n: int, trend_seq: np.ndarray) -> np.ndarray:
        """Apply either the default linear trend or the quadratic trend.

        Args:
            n (int): Number of observations.
            trend_seq (np.ndarray): Trend sequence returned by `_init_common`.

        Returns:
            np.ndarray: Updated trend sequence.
        """
        if not self.quadratic_trend:
            return trend_seq

        t = np.arange(n, dtype=float)
        return float(self.trend_slope) * (t**2)

    def _resolve_break_indices(self, n: int) -> list[int]:
        """Resolve zero-based structural break indices.

        Priority:
            1. Use explicit `structural_break_times` if provided.
            2. Use backward-compatible single `break_time` if provided.
            3. Randomly sample `n_structural_breaks` break times if positive.
            4. Fall back to one midpoint break if `structural_break=True`.

        Args:
            n (int): Number of generated observations.

        Returns:
            list[int]: Sorted zero-based structural break indices. Empty if
                structural breaks are disabled.
        """
        if not self.structural_break or n <= 1:
            return []

        if self.structural_break_times is not None:
            break_indices = [int(t) - 1 for t in self.structural_break_times]

        elif self.break_time is not None:
            break_indices = [int(self.break_time) - 1]

        elif self.n_structural_breaks > 0:
            break_indices = self._sample_random_break_indices(n)

        else:
            break_indices = [int(np.ceil(n / 2)) - 1]

        clipped = []
        for break_idx in break_indices:
            break_idx = int(np.clip(break_idx, 1, max(1, n - 1)))
            clipped.append(break_idx)

        return sorted(set(clipped))

    def _sample_random_break_indices(self, n: int) -> list[int]:
        """Sample random zero-based structural break indices.

        Breaks are sampled away from the boundaries. If `min_break_distance`
        is provided, the method tries to enforce a minimum distance between
        consecutive breaks.

        Args:
            n (int): Number of generated observations.

        Returns:
            list[int]: Sorted zero-based break indices.

        Raises:
            ValueError: If the requested break configuration cannot be sampled.
        """
        n_breaks = int(self.n_structural_breaks)
        if n_breaks <= 0:
            return []

        margin = self.break_margin
        if margin is None:
            margin = max(1, n // 10)

        lower = max(1, int(margin))
        upper = min(n - 1, n - int(margin))

        if upper <= lower:
            lower = 1
            upper = max(1, n - 1)

        candidates = np.arange(lower, upper + 1, dtype=int)

        if candidates.size == 0:
            return []

        if n_breaks > candidates.size:
            raise ValueError(
                f'Cannot sample {n_breaks} structural breaks from only '
                f'{candidates.size} candidate positions.'
            )

        min_dist = self.min_break_distance

        if min_dist is None or min_dist <= 1:
            sampled = self.rng.choice(candidates, size=n_breaks, replace=False)
            return sorted(int(x) for x in sampled)

        min_dist = int(min_dist)

        for _ in range(1_000):
            sampled = np.sort(self.rng.choice(candidates, size=n_breaks, replace=False))
            if np.all(np.diff(sampled) >= min_dist):
                return [int(x) for x in sampled]

        raise ValueError(
            'Could not sample structural breaks satisfying min_break_distance. '
            'Try reducing n_structural_breaks, min_break_distance, or break_margin.'
        )

    def _resolve_structural_break_jump_sizes(self, n_breaks: int) -> np.ndarray:
        """Resolve level jumps attached to structural breaks.

        Args:
            n_breaks (int): Number of structural breaks.

        Returns:
            np.ndarray: Jump sizes of length `n_breaks`.

        Raises:
            ValueError: If jump-size dimensions are incompatible.
        """
        if n_breaks <= 0:
            return np.zeros(0, dtype=float)

        if self.structural_break_jump_sizes is None:
            return np.zeros(n_breaks, dtype=float)

        arr = np.asarray(self.structural_break_jump_sizes, dtype=float)

        if arr.ndim == 0:
            return np.full(n_breaks, float(arr), dtype=float)

        arr = arr.reshape(-1)

        if arr.size == 1:
            return np.full(n_breaks, float(arr[0]), dtype=float)

        if arr.size != n_breaks:
            raise ValueError(
                'structural_break_jump_sizes must be scalar, length 1, '
                'or have the same length as the number of structural breaks.'
            )

        return arr.astype(float)

    def _apply_structural_break(
        self,
        trend_seq: np.ndarray,
        sigma_seq: np.ndarray,
        break_indices: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply piecewise trend, level, and noise changes.

        The trend sign flips at every structural break. If `quadratic_trend`
        is False, each segment uses a linear trend

            level + a * tau.

        If `quadratic_trend` is True, each segment uses a quadratic trend

            level + a * tau**2.

        In both cases, tau is the local time since the beginning of the current
        segment. The innovation scale is smaller when the current trend
        coefficient is nonnegative and larger when it is negative. Optional
        level jumps can be attached directly to structural breaks.

        Args:
            trend_seq (np.ndarray): Original trend sequence.
            sigma_seq (np.ndarray): Original innovation standard deviation sequence.
            break_indices (list[int]): Zero-based structural break indices.

        Returns:
            tuple[np.ndarray, np.ndarray]: Updated trend and innovation scale
                sequences.
        """
        if len(break_indices) == 0:
            return trend_seq, sigma_seq

        n = trend_seq.size
        trend_seq = np.asarray(trend_seq, dtype=float).copy()
        sigma_seq = np.asarray(sigma_seq, dtype=float).copy()

        jump_sizes = self._resolve_structural_break_jump_sizes(len(break_indices))

        current_coef = float(self.trend_slope)
        current_level = 0.0
        segment_start = 0

        all_breaks = list(break_indices) + [n]

        for segment_idx, break_idx in enumerate(all_breaks):
            for t in range(segment_start, break_idx):
                tau = float(t - segment_start)

                if self.quadratic_trend:
                    trend_seq[t] = current_level + current_coef * (tau**2)
                else:
                    trend_seq[t] = current_level + current_coef * tau

                if current_coef >= 0:
                    sigma_seq[t] = self.noise_std * self.positive_trend_noise_scale
                else:
                    sigma_seq[t] = self.noise_std * self.negative_trend_noise_scale

            if break_idx < n:
                tau_break = float(break_idx - segment_start)

                if self.quadratic_trend:
                    current_level = current_level + current_coef * (tau_break**2)
                else:
                    current_level = current_level + current_coef * tau_break

                current_level += float(jump_sizes[segment_idx])

                segment_start = break_idx
                current_coef = -current_coef

        return trend_seq, sigma_seq

    def _generate_innovations(self, n: int, sigma_seq: np.ndarray) -> np.ndarray:
        """Generate innovations with the requested marginal innovation scale.

        Args:
            n (int): Number of innovations to generate.
            sigma_seq (np.ndarray): Innovation standard deviations.

        Returns:
            np.ndarray: Innovation sequence of length `n`.
        """
        if self.noise_dist == 'normal':
            base = self.rng.normal(loc=0.0, scale=1.0, size=n)
            return base * sigma_seq

        base_t = self.rng.standard_t(df=self.t_df, size=n)

        if self.t_df > 2:
            std_correction = np.sqrt(self.t_df / (self.t_df - 2.0))
            base_unitvar = base_t / std_correction
            return base_unitvar * sigma_seq

        return base_t * sigma_seq

    @property
    def variance(self) -> Optional[float]:
        """Unconditional variance under stationarity and finite innovation variance.

        Returns:
            Optional[float]: Approximate unconditional variance of the centered
                AR(p) process if stationary. Returns None otherwise.
        """
        if (
            self._meta.n_generated == 0
            or getattr(self._meta, 'sigma_seq', None) is None
        ):
            return None

        if (
            getattr(self, 'noise_dist', 'normal') == 'student'
            and self.t_df is not None
            and self.t_df <= 2
        ):
            return None

        if not self._is_stationary():
            return None

        sig2 = float(np.mean(np.asarray(self._meta.sigma_seq, dtype=float) ** 2))
        return self._ar_unconditional_variance(sig2)

    def instantaneous_variance(self, t: int) -> float:
        """Instantaneous unconditional variance at time index `t`.

        Args:
            t (int): Zero-based time index.

        Returns:
            float: Instantaneous unconditional variance at time t.

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

        if not self._is_stationary():
            return float('inf')

        sig2 = float(self._meta.sigma_seq[t] ** 2)
        return self._ar_unconditional_variance(sig2)

    def _is_stationary(self) -> bool:
        """Check stationarity of the centered AR(p) process.

        Returns:
            bool: True if all eigenvalues of the companion matrix lie inside
                the unit circle.
        """
        companion = self._companion_matrix()
        eigvals = np.linalg.eigvals(companion)
        return bool(np.max(np.abs(eigvals)) < 1.0)

    def _companion_matrix(self) -> np.ndarray:
        """Construct the AR(p) companion matrix.

        Returns:
            np.ndarray: Companion matrix of shape `(p, p)`.
        """
        companion = np.zeros((self.p, self.p), dtype=float)
        companion[0, :] = self.phi

        if self.p > 1:
            companion[1:, :-1] = np.eye(self.p - 1)

        return companion

    def _ar_unconditional_variance(self, innovation_variance: float) -> float:
        """Compute the unconditional variance of a stationary AR(p) process.

        Args:
            innovation_variance (float): Innovation variance.

        Returns:
            float: Unconditional variance of the first state component.
        """
        companion = self._companion_matrix()

        q = np.zeros((self.p, self.p), dtype=float)
        q[0, 0] = float(innovation_variance)

        eye = np.eye(self.p * self.p)
        kron = np.kron(companion, companion)

        vec_sigma = np.linalg.solve(eye - kron, q.reshape(-1))
        sigma = vec_sigma.reshape(self.p, self.p)

        return float(max(0.0, sigma[0, 0]))
