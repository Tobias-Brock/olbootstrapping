from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

from olbootstrap.online_bootstrap._online_ar_bootstrap import OnlineARBootstrap

from ._base_experiment import BaseBootstrapExperiment


class OnlineARBootstrapExperiment(BaseBootstrapExperiment):
    """Online AR-multiplier bootstrap experiment runner."""

    def __init__(
        self,
        sample_size: int = 500,
        process=None,
        B: int = 200,
        record_every: int = 1,
        smoothing_method: Optional[str] = None,
        eta: float = 0.15,
        burn_in: int = 0,
        var_warmup: int = 0,
        use_variance_smoothing: bool = False,
        smoothing_beta: Optional[float] = None,
        gamma: Optional[float] = None,
        seasonal_period: Optional[int] = None,
        bootstrap_cls=OnlineARBootstrap,
        progress: bool = False,
        forecast_s: int = 0,
        rng_weights: Optional[np.random.Generator] = None,
        transform: Optional[str] = None,
        transform_power: float = 1.0 / 3.0,
        rho_power: float = (-1.0 / 3.0),
        alpha: float = 0.05,
    ):
        """Initialize the online experiment.

        Args:
            sample_size: Default number of observations to simulate if no
                explicit samples are provided.
            process: Optional process object with generate_samples(n).
            B: Number of bootstrap replicates.
            record_every: Record state every this many observations.
            smoothing_method: Smoother name for target series ('ewma','holt',...).
            eta: Primary smoothing parameter used by smoothers and bootstrap.
            burn_in: Initial observations to exclude from evaluations.
            var_warmup: Warmup length for variance/threshold scheduling.
            use_variance_smoothing: Apply variance smoothing if True.
            smoothing_beta: Secondary smoother parameter for Holt/HW.
            gamma: Seasonal/trend smoothing parameter for HW.
            seasonal_period: Seasonal period for Holt-Winters.
            bootstrap_cls: Class implementing the online bootstrap interface.
            progress: Show a progress bar when True.
            forecast_s: Forecast horizon passed to bootstrap (0 = contemporaneous).
            rng_weights: RNG for multiplier weights; default is new RNG.
            transform: Multiplier transform ('student'|'gauss') or None.
            transform_power: Power mapping effective sample size to df.
            rho_power: Exponent used for latent AR correlation scaling.
            alpha: Nominal significance level for calibration/quantiles.

        Notes:
            Argument validation is performed lazily where needed.
        """
        super().__init__(
            sample_size=sample_size,
            process=process,
            B=B,
            record_every=record_every,
            smoothing_method=smoothing_method,
            eta=eta,
            burn_in=burn_in,
            var_warmup=var_warmup,
            use_variance_smoothing=use_variance_smoothing,
            smoothing_beta=smoothing_beta,
            gamma=gamma,
            seasonal_period=seasonal_period,
            bootstrap_cls=bootstrap_cls,
            progress=progress,
            forecast_s=forecast_s,
            rng_weights=rng_weights,
            transform=transform,
            transform_power=transform_power,
            rho_power=rho_power,
            alpha=alpha,
        )

    def run(
        self,
        samples: Optional[Sequence[float]] = None,
        sample_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run the online bootstrap over a sample stream.

        Args:
            samples: Optional sequence of observations to process. If None,
                `self.process.generate_samples(sample_size)` is used.
            sample_size: Number of observations to simulate when `samples` is
                None and a process generator is provided.

        Raises:
            ValueError: If neither samples nor a process+sample_size are given.

        Returns:
            Dict[str, Any]: The experiment results as produced by `results()`.
        """
        if samples is None:
            samples = self.process.generate_samples(sample_size)
        else:
            samples = np.asarray(list(samples))

        self.samples = np.asarray(samples)
        n = len(self.samples)

        self._ensure_bootstrap()
        bootstrap = self._bootstrap

        times = []
        bootstrap_means = []
        replicates_by_time: list[np.ndarray] = []
        rep_sds: list[float] = []
        smooth_estimates_list = []
        uniform_lower_list: list[float] = []
        uniform_upper_list: list[float] = []
        thresholds_list: list[float] = []
        q_active_list: list[float] = []
        sigma_star_list: list[float] = []

        iterator = enumerate(self.samples, start=1)
        if self.progress:
            iterator = tqdm(
                iterator, total=len(self.samples), desc='Bootstrapping', leave=False
            )

        for t_idx, x in iterator:
            self._last_obs = float(x)
            if t_idx == 1:
                bootstrap(np.array([x]), number_bootstrap_samples=self.B)
            else:
                bootstrap(np.array([x]))

            if (t_idx % self.record_every) != 0:
                continue

            times.append(t_idx)

            reps_arr = np.asarray(bootstrap.bootstrap_averages)
            rep_vals = (
                reps_arr[:, 0]
                if reps_arr.ndim == 2 and reps_arr.shape[1] >= 1
                else reps_arr.flatten()
            ).astype(float)
            replicates_by_time.append(rep_vals.copy())

            mu_point_t = float(getattr(bootstrap, 'mu_point', float('nan')))
            smooth_estimates_list.append(mu_point_t)
            rep_sds.append(float(np.std(rep_vals, ddof=0)))
            bootstrap_means.append(float(np.mean(rep_vals)))

            q_act = getattr(bootstrap, 'q_active', None)
            sigs = getattr(bootstrap, 'sigma_star', None)

            if (
                (q_act is None)
                or (sigs is None)
                or (not np.isfinite(q_act))
                or (not np.isfinite(sigs))
                or sigs <= 0.0
            ):
                thresholds_list.append(np.nan)
                q_active_list.append(np.nan)
                sigma_star_list.append(np.nan)
                uniform_lower_list.append(np.nan)
                uniform_upper_list.append(np.nan)
            else:
                c_t = float(q_act) * float(sigs)
                thresholds_list.append(c_t)
                q_active_list.append(float(q_act))
                sigma_star_list.append(float(sigs))
                uniform_lower_list.append(mu_point_t - c_t)
                uniform_upper_list.append(mu_point_t + c_t)

        self.times = np.array(times, dtype=int)

        self.bootstrap_means = np.array(bootstrap_means, dtype=float)
        self.smooth_estimates = np.asarray(smooth_estimates_list, dtype=float)
        self.replicates_by_time = replicates_by_time
        self.rep_sds = np.array(rep_sds, dtype=float)

        self._uniform_record_times_all = self.times.copy()
        self._uniform_lower_all = np.asarray(uniform_lower_list, dtype=float)
        self._uniform_upper_all = np.asarray(uniform_upper_list, dtype=float)
        self._uniform_thresholds_all = np.asarray(thresholds_list, dtype=float)
        self._uniform_sigma_star_all = np.asarray(sigma_star_list, dtype=float)
        self._uniform_q_active = np.asarray(q_active_list, dtype=float)  # <- persist q

        self.uniform_record_times = self._uniform_record_times_all.copy()
        self.uniform_lower = self._uniform_lower_all.copy()
        self.uniform_upper = self._uniform_upper_all.copy()
        self._uniform_thresholds = self._uniform_thresholds_all.copy()
        self._uniform_sigma_star = self._uniform_sigma_star_all.copy()

        if hasattr(self.process, '_meta'):
            meta = self.process._meta
            trend = np.asarray(getattr(meta, 'trend_seq', np.zeros(n)), dtype=float)
            seasonal = np.asarray(
                getattr(meta, 'seasonal_seq', np.zeros(n)), dtype=float
            )
            level_jump = np.asarray(getattr(meta, 'level_jump_seq', 0.0), dtype=float)
            self.mu_t = self.process.mean + trend + seasonal + level_jump
        else:
            self.mu_t = np.full(
                n,
                (self.process.mean if self.process is not None else np.nan),
            )

        return self.results()

    def compute_intervals(
        self,
        quantile: str = 'normal',  # "empirical" or "normal"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute pointwise confidence intervals centered at smooth_estimates.

        Args:
            quantile: 'empirical' to use bootstrap quantiles, 'normal' to use
                Gaussian approximation.

        Returns:
            (low, high) arrays of length T where T is the number of recorded
            times.

        Raises:
            ValueError: If quantile is not 'empirical' or 'normal'.
        """
        if quantile not in ('empirical', 'normal'):
            raise ValueError("quantile must be 'empirical' or 'normal'")

        T = len(self.replicates_by_time)
        lowers = np.empty(T, dtype=float)
        uppers = np.empty(T, dtype=float)

        if quantile == 'normal':
            z = float(norm.ppf(1.0 - self.alpha / 2.0))  # same for all t
            q_lo, q_hi = -z, z

        for t in range(T):
            rep_vals = np.asarray(self.replicates_by_time[t], dtype=float).reshape(-1)

            if rep_vals.size == 0 or not np.all(np.isfinite(rep_vals)):
                center, sd = float('nan'), 0.0
            else:
                center = float(self.smooth_estimates[t])
                sd = float(np.std(rep_vals, ddof=0))

            if not np.isfinite(sd) or sd == 0.0:
                lowers[t] = uppers[t] = center
                continue

            if quantile == 'empirical':
                std = (rep_vals - center) / sd
                q_lo = float(np.quantile(std, self.alpha / 2.0))
                q_hi = float(np.quantile(std, 1.0 - self.alpha / 2.0))

            lowers[t] = center - q_hi * sd
            uppers[t] = center - q_lo * sd

        self.lower_bounds = lowers
        self.upper_bounds = uppers

        self._pointwise_times_all = self.times.copy()
        self._pointwise_lower_all = lowers.copy()
        self._pointwise_upper_all = uppers.copy()

        self.pointwise_times = self._pointwise_times_all
        self.pointwise_lower = self._pointwise_lower_all
        self.pointwise_upper = self._pointwise_upper_all

        return lowers, uppers

    def compute_uniform_bands(self) -> tuple[np.ndarray, np.ndarray, float | None]:
        """Compute and store uniform bands after masking by burn_in/warmup.

        Returns:
            (uniform_lower, uniform_upper, q_latest) where arrays are restricted
            to times after burn_in+var_warmup and q_latest is the most recent
            active q statistic (or None if unavailable).
        """
        if not hasattr(self, '_uniform_record_times_all'):
            self._uniform_record_times_all = np.asarray(
                self.uniform_record_times, dtype=int
            )
            self._uniform_lower_all = np.asarray(self.uniform_lower, dtype=float)
            self._uniform_upper_all = np.asarray(self.uniform_upper, dtype=float)

        burn_eff = int(self.t0) + int(self.var_warmup)

        times_full = self._uniform_record_times_all
        uL_full = self._uniform_lower_all
        uU_full = self._uniform_upper_all

        mask = (times_full > burn_eff) & np.isfinite(uL_full) & np.isfinite(uU_full)

        self.uniform_record_times = times_full[mask]
        self.uniform_lower = uL_full[mask]
        self.uniform_upper = uU_full[mask]

        widths_full = (uU_full - uL_full).astype(float)
        self.uniform_widths = widths_full.copy()

        if np.any(mask):
            self.uniform_mean_width = float(np.nanmean(widths_full[mask]))
        else:
            self.uniform_mean_width = float('nan')

        q_vec = getattr(self, '_uniform_q_active', None)
        q_latest = None
        if q_vec is not None:
            q_vec = np.asarray(q_vec, dtype=float)
            ok = np.isfinite(q_vec)
            if np.any(ok):
                q_latest = float(q_vec[ok][-1])

        return self.uniform_lower, self.uniform_upper, q_latest

    def compute_coverage(
        self,
        band: str = 'pointwise',
    ) -> Dict[str, Any]:
        """Check coverage of the stored bands against the smoothed mean.

        Args:
            band: One of 'pointwise', 'uniform' or 'both'.

        Returns:
            Dict with masks, coverage booleans and used times for the requested
            band type(s).

        Raises:
            ValueError: If `band` is not one of the allowed values.
            RuntimeError: If no recorded times remain after burn_in for the
                pointwise check.
        """
        if band not in ('pointwise', 'uniform', 'both'):
            raise ValueError("band must be 'pointwise', 'uniform' or 'both'")

        target_full = self._smoother_target_series()  # length n
        target_at_recorded_full = target_full[self.times - 1]

        out: Dict[str, Any] = {}

        if band in ('pointwise', 'both'):
            times_full = np.asarray(self.times, dtype=int)
            mask_pw = times_full > int(self.t0)
            if not np.any(mask_pw):
                raise RuntimeError(
                    'No times remain after applying burn_in; choose smaller burn_in.'
                )

            lb_full = np.asarray(self.lower_bounds)
            ub_full = np.asarray(self.upper_bounds)
            if lb_full.ndim == 0:
                lb_full = np.full(times_full.shape, float(lb_full))
            if ub_full.ndim == 0:
                ub_full = np.full(times_full.shape, float(ub_full))

            times_used_pw = times_full[mask_pw]
            target_at_times_pw = np.asarray(target_at_recorded_full)[mask_pw]
            lb = lb_full[mask_pw]
            ub = ub_full[mask_pw]

            pw_mask = (target_at_times_pw >= lb) & (target_at_times_pw <= ub)
            out.update(
                {
                    'pointwise_mask': pw_mask,
                    'pointwise_coverage': float(np.mean(pw_mask))
                    if pw_mask.size > 0
                    else float('nan'),
                    'used_times_pointwise': times_used_pw.copy(),
                    'burn_in_pointwise': int(self.t0),
                }
            )

        if band in ('uniform', 'both'):
            burn_eff = int(self.t0) + int(self.var_warmup)
            rec_times_full = np.asarray(
                getattr(self, '_uniform_record_times_all', self.uniform_record_times),
                dtype=int,
            )
            u_lower_full = np.asarray(
                getattr(self, '_uniform_lower_all', self.uniform_lower), dtype=float
            )
            u_upper_full = np.asarray(
                getattr(self, '_uniform_upper_all', self.uniform_upper), dtype=float
            )

            if u_lower_full.ndim == 0:
                u_lower_full = np.full(rec_times_full.shape, float(u_lower_full))
            if u_upper_full.ndim == 0:
                u_upper_full = np.full(rec_times_full.shape, float(u_upper_full))

            mask_u = rec_times_full > burn_eff
            finite = np.isfinite(u_lower_full) & np.isfinite(u_upper_full)
            mask_u &= finite

            if not np.any(mask_u):
                out.update(
                    {
                        'uniform_mask': np.array([], dtype=bool),
                        'uniform_covered': False,
                        'used_times_uniform': np.array([], dtype=int),
                        'burn_in_uniform': int(burn_eff),
                    }
                )
            else:
                rec_times = rec_times_full[mask_u]

                times_all = np.asarray(self.times, dtype=int)
                mask_after_burn_pw = times_all > burn_eff
                times_used_pw = times_all[mask_after_burn_pw]
                target_at_used_pw = np.asarray(target_at_recorded_full)[
                    mask_after_burn_pw
                ]

                time_to_pos = {int(t): idx for idx, t in enumerate(times_used_pw)}
                pos_list = np.array([time_to_pos[int(t)] for t in rec_times], dtype=int)

                target_for_uniform = target_at_used_pw[pos_list]
                u_lower = u_lower_full[mask_u]
                u_upper = u_upper_full[mask_u]

                u_mask = (target_for_uniform >= u_lower) & (
                    target_for_uniform <= u_upper
                )
                out.update(
                    {
                        'uniform_mask': u_mask,
                        'uniform_covered': bool(np.all(u_mask)),
                        'used_times_uniform': rec_times.copy(),
                        'burn_in_uniform': int(burn_eff),
                    }
                )

        return out
