from typing import Optional, Sequence

import numpy as np

from ._base_process import BaseProcess


class MovingAverage(BaseProcess):
    def __init__(
        self,
        mean: float,
        parameters: np.ndarray,
        *,
        noise_std: float = 1.0,
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
        """Moving average process MA(q-1) simulator.

        Notes
        -----
        - `parameters` should provide theta_1..theta_{q-1}; internally a leading 1.0
          is prepended so `self.parameters = [1, theta_1, ..., theta_{q-1}]`.
        - `noise_std` is the (constant) standard deviation of the Gaussian innovations.
        - Shock arguments (shock_type, jump_prob, ...) are forwarded to BaseProcess.
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

        theta = np.asarray(parameters, dtype=float)
        self.parameters = np.concatenate(([1.0], theta))
        self.q = len(self.parameters)

        # keep only the single noise_std attribute
        self.noise_std = float(noise_std)

        self.samples = np.array([])
        # store sigma_seq in meta as a constant array once generate_samples is called
        self._meta.sigma_seq = None

    def generate_samples(self, number_samples: int) -> np.ndarray:
        n = int(number_samples)
        if n < 0:
            raise ValueError('number_samples must be non-negative')
        if n == 0:
            self.samples = np.zeros(0, dtype=float)
            _ = self._finalize_meta(0, trend_seq=np.zeros(0), seasonal_seq=np.zeros(0))
            return self.samples

        q = self.q
        offset = q - 1

        # constant sigma path: sigma_t == noise_std for all t
        sigma_seq = np.full(n, float(self.noise_std), dtype=float)

        trend_seq, seasonal_seq = self._init_common(n)

        # for MA we need n + offset epsilons (prepended history)
        sigma_eps = np.empty(n + offset, dtype=float)
        if offset > 0:
            sigma_eps[:offset] = sigma_seq[0]
        sigma_eps[offset:] = sigma_seq

        # eps standard normal scaled by noise_std via sigma_eps
        eps = self.rng.normal(loc=0.0, scale=1.0, size=n + offset) * sigma_eps

        theta = self.parameters
        samples = np.empty(n, dtype=float)

        # apply any jump at t=1 (before generating first sample)
        self._maybe_apply_jump_t1()

        idx_offsets = np.arange(offset, offset + n)
        for i, i_off in enumerate(idx_offsets):
            time1 = i + 1
            self._decay_transient_before_step()
            if time1 >= 2:
                self._maybe_apply_jump(time1)

            m_t = self.mean + trend_seq[i] + seasonal_seq[i] + self._shock_component()
            idx = i_off - np.arange(q)
            samples[i] = m_t + np.dot(theta, eps[idx])

            self._log_shock_state(i)

        self.samples = samples
        self._finalize_meta(
            n,
            trend_seq=trend_seq,
            seasonal_seq=seasonal_seq,
            extra_meta={'sigma_seq': sigma_seq},
        )
        return self.samples

    @property
    def variance(self) -> Optional[float]:
        """
        Unconditional variance under constant noise_std:
        Var(X_t) = (noise_std^2) * sum_j theta_j^2
        """
        if (
            self._meta.n_generated == 0
            or getattr(self._meta, 'sigma_seq', None) is None
        ):
            return None
        sum_sq_theta = float(np.sum(self.parameters**2))
        return float(self.noise_std**2) * sum_sq_theta

    def instantaneous_variance(self, t: int) -> float:
        if (
            self._meta.n_generated == 0
            or getattr(self._meta, 'sigma_seq', None) is None
        ):
            raise RuntimeError(
                'No samples generated yet; cannot provide instantaneous variance.'
            )
        if t < 0 or t >= self._meta.n_generated:
            raise IndexError('t out of range')
        sum_sq_theta = float(np.sum(self.parameters**2))
        # instantaneous sigma is the constant noise_std
        sigma_t = float(self.noise_std)
        return (sigma_t**2) * sum_sq_theta
