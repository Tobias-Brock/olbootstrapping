from types import SimpleNamespace
from typing import Optional, Sequence, Tuple

import numpy as np


class BaseProcess:
    """Reusable base for time-series generators."""

    def __init__(
        self,
        *,
        mean: float,
        trend_slope: float = 0.0,
        seasonal_amplitude: float = 0.0,
        seasonal_period: Optional[float] = None,
        seasonal_phase: float = 0.0,
        rng: Optional[np.random.Generator] = None,
        shock_type: str = 'none',  # "none" | "permanent" | "transient"
        jump_prob: float = 0.0,
        jump_scale: float = 1.0,
        decay: float = 0.95,
        jump_times: Optional[Sequence[int]] = None,
        jump_sizes: Optional[Sequence[float]] = None,
    ):
        """Reusable base for time-series generators.

        Args:
            mean (float): Baseline constant mean.
            trend_slope (float, optional): Linear trend slope. Defaults to 0.0.
            seasonal_amplitude (float, optional): Amplitude of seasonal component.
                Defaults to 0.0.
            seasonal_period (Optional[float], optional): Period of seasonality.
                Required if seasonal_amplitude != 0. Defaults to None.
            seasonal_phase (float, optional): Phase shift for seasonal component.
                Defaults to 0.0.
            rng (Optional[np.random.Generator], optional): Random number generator.
                Defaults to ``np.random.default_rng()``.
            shock_type (str, optional): One of {'none','permanent','transient'}.
                Defaults to "none".
            jump_prob (float, optional): Probability of a random jump at each time step.
                Defaults to 0.0.
            jump_scale (float, optional): Std dev of jump size when sampled.
                Defaults to 1.0.
            decay (float, optional): Decay factor for transient shocks (only used when
                shock_type == 'transient'). Defaults to 0.95.
            jump_times (Optional[Sequence[int]], optional): Deterministic 1-based jump
                times. Defaults to None.
            jump_sizes (Optional[Sequence[float]], optional): Optional deterministic
                jump sizes aligned with jump_times. Defaults to None.

        Raises:
            ValueError: If seasonal_amplitude != 0 and seasonal_period is not positive.
            ValueError: If shock_type is not one of {'none', 'permanent', 'transient'}.
        """
        self.mean = float(mean)
        self.trend_slope = float(trend_slope)
        self.seasonal_amplitude = float(seasonal_amplitude)
        self.seasonal_period = (
            None if seasonal_period is None else float(seasonal_period)
        )
        self.seasonal_phase = float(seasonal_phase)
        if self.seasonal_amplitude != 0.0 and (
            self.seasonal_period is None or self.seasonal_period <= 0.0
        ):
            raise ValueError(
                'seasonal_period must be positive when seasonal_amplitude != 0'
            )

        self.rng = rng if rng is not None else np.random.default_rng()

        self.shock_type = str(shock_type).lower()
        if self.shock_type not in ('none', 'permanent', 'transient'):
            raise ValueError("shock_type must be one of 'none','permanent','transient'")
        self.jump_prob = float(jump_prob)
        self.jump_scale = float(jump_scale)
        self.decay = float(decay) if self.shock_type == 'transient' else None

        if jump_times is not None:
            jt = np.asarray(jump_times, dtype=int)
            self._det_jump_times = {int(t) for t in jt if t >= 1}
        else:
            self._det_jump_times = set()
        self._det_jump_sizes = list(jump_sizes) if jump_sizes is not None else None

        self._level_jump = 0.0
        self._transient = 0.0
        self._det_sizes_map = {}
        self._jump_times_list = []
        self._jump_sizes_list = []
        self._level_jumps = None
        self._transient_seq = None

        self._meta = SimpleNamespace(
            trend_seq=None,
            seasonal_seq=None,
            level_jump_seq=None,
            transient_seq=None,
            jump_times=None,
            jump_sizes=None,
            n_generated=0,
        )

    def _build_trend_seasonal(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build deterministic trend and seasonal sequences.

        Args:
            n (int): Number of time steps to generate.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A pair `(trend_seq, seasonal_seq)`, each of
                length `n`.

        Notes:
            - `trend_seq[t] = trend_slope * (t+1)` for t in 0..n-1 (1-based time).
            - If seasonal_amplitude == 0, `seasonal_seq` is a zero array.
        """
        trend_seq = self.trend_slope * np.arange(1, n + 1, dtype=float)
        if self.seasonal_amplitude != 0.0 and self.seasonal_period is not None:
            t_idx = np.arange(1, n + 1, dtype=float)
            seasonal_seq = self.seasonal_amplitude * np.sin(
                2.0 * np.pi * t_idx / self.seasonal_period + self.seasonal_phase
            )
        else:
            seasonal_seq = np.zeros(n, dtype=float)
        return trend_seq, seasonal_seq

    def _prepare_deterministic_jump_map(self):
        """Prepare an internal map of deterministic jump times to sizes.

        Returns:
            None
        """
        self._det_sizes_map = {}
        if not self._det_jump_times:
            return
        if self._det_jump_sizes is not None and len(self._det_jump_sizes) == len(
            self._det_jump_times
        ):
            for idx, t in enumerate(sorted(self._det_jump_times)):
                self._det_sizes_map[t] = float(self._det_jump_sizes[idx])
        else:
            for t in self._det_jump_times:
                self._det_sizes_map[t] = None  # size sampled at runtime

    def _init_common(self, n: int):
        """Initialize runtime state common to all sample-generation routines.

        Args:
            n (int): Number of time steps to initialize for.

        Returns:
            Tuple[np.ndarray, np.ndarray]: `(trend_seq, seasonal_seq)` for the given `n`.
        """
        self._level_jump = 0.0
        self._transient = 0.0
        self._jump_times_list = []
        self._jump_sizes_list = []
        self._level_jumps = np.zeros(n, dtype=float)
        self._transient_seq = np.zeros(n, dtype=float)
        self._prepare_deterministic_jump_map()
        trend_seq, seasonal_seq = self._build_trend_seasonal(n)
        return trend_seq, seasonal_seq

    def _decay_transient_before_step(self):
        """Decay transient shock component before generating the next time step.

        Returns:
            None
        """
        if self.shock_type == 'transient':
            self._transient *= float(self.decay)

    def _maybe_apply_jump_t1(self):
        """Apply deterministic or random jump before generating the first sample.

        Returns:
            None
        """
        if self.shock_type == 'none':
            return
        j = None
        if 1 in self._det_jump_times:
            sz = self._det_sizes_map.get(1)
            j = sz if sz is not None else self.rng.normal(scale=self.jump_scale)
        elif self.rng.random() < self.jump_prob:
            j = self.rng.normal(scale=self.jump_scale)
        if j is not None:
            if self.shock_type == 'permanent':
                self._level_jump += j
            else:
                self._transient += j
            self._jump_times_list.append(1)
            self._jump_sizes_list.append(j)

    def _maybe_apply_jump(self, time1: int):
        """Apply deterministic or random jump at a given 1-based time.

        Args:
            time1 (int): 1-based time index at which to evaluate jumps.

        Returns:
            None
        """
        if self.shock_type == 'none':
            return
        j = None
        if time1 in self._det_jump_times:
            sz = self._det_sizes_map.get(time1)
            j = sz if sz is not None else self.rng.normal(scale=self.jump_scale)
        elif self.rng.random() < self.jump_prob:
            j = self.rng.normal(scale=self.jump_scale)
        if j is not None:
            if self.shock_type == 'permanent':
                self._level_jump += j
            else:
                self._transient += j
            self._jump_times_list.append(time1)
            self._jump_sizes_list.append(j)

    def _shock_component(self) -> float:
        """Return the active shock contribution to the process mean.

        Returns:
            float: Current permanent or transient shock component.
        """
        if self.shock_type == 'permanent':
            return self._level_jump
        elif self.shock_type == 'transient':
            return self._transient
        else:
            return 0.0

    def _log_shock_state(self, i: int):
        """Store the current shock state at a zero-based sample index.

        Args:
            i (int): Zero-based sample index to write into metadata arrays.

        Returns:
            None
        """
        self._level_jumps[i] = self._level_jump
        self._transient_seq[i] = self._transient

    def _finalize_meta(
        self, n: int, *, trend_seq, seasonal_seq, extra_meta: dict = None
    ):
        """Finalize metadata after sample generation and return a summary namespace.

        Args:
            n (int): Number of generated time steps.
            trend_seq (np.ndarray): Trend sequence produced by `_init_common`.
            seasonal_seq (np.ndarray): Seasonal sequence produced by `_init_common`.
            extra_meta (dict, optional): Additional key-value pairs to attach to the
                returned meta object.

        Returns:
            SimpleNamespace: The `_meta` object with fields:
                - trend_seq, seasonal_seq, level_jump_seq, transient_seq,
                jump_times, jump_sizes, n_generated, and any keys from `extra_meta`.
        """
        self._meta.trend_seq = trend_seq
        self._meta.seasonal_seq = seasonal_seq
        self._meta.level_jump_seq = self._level_jumps
        self._meta.transient_seq = self._transient_seq
        self._meta.jump_times = list(self._jump_times_list)
        self._meta.jump_sizes = list(self._jump_sizes_list)
        self._meta.n_generated = n
        if extra_meta:
            for k, v in extra_meta.items():
                setattr(self._meta, k, v)
        return self._meta
