from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from joblib import Parallel, delayed
import numpy as np
from tqdm.auto import tqdm

from olbootstrap.experiments._experiments import OnlineARBootstrapExperiment
from olbootstrap.study._base_study import BaseCoverageStudy, SweepResults


class UniformCoverageStudy(BaseCoverageStudy):
    """Concrete study: implements run + the two sweep methods, inherits helpers."""

    def run(self, position: int = 0, leave_inner: bool = False) -> Dict[str, Any]:
        """Run `n_series` replications and compute coverage summaries.

        Args:
            position: tqdm display position when nested progress bars are used.
            leave_inner: If True, inner progress bar is left after completion.

        Returns:
            Dict[str, Any]: A dictionary containing arrays and summary metrics:
                - series_pointwise_frac
                - series_pointwise_all_ok
                - series_uniform_frac
                - series_uniform_ok
                - avg_pointwise_time_fraction
                - avg_uniform_time_fraction
                - uniform_over_series_pointwise
                - uniform_over_series_uniform
                - uniform_coverage
                - series_uniform_mean_width
                - avg_uniform_mean_width
        """
        for attr in (
            'series_uniform_ok',
            'series_pointwise_frac',
            'series_pointwise_all_ok',
            'series_uniform_frac',
            'series_uniform_width_mean',
        ):
            lst = getattr(self, attr, None)
            if lst is None:
                setattr(self, attr, [])
            else:
                lst.clear()

        master = np.random.SeedSequence(self.seed)
        rep_seqs = master.spawn(self.n_series)
        rep_iter = (
            tqdm(
                range(self.n_series),
                desc='replications',
                total=self.n_series,
                position=position,
                leave=leave_inner,
                dynamic_ncols=True,
                mininterval=0.1,
            )
            if self.progress
            else range(self.n_series)
        )

        for k in rep_iter:
            data_seq, weights_seq = rep_seqs[k].spawn(2)
            rng_data = np.random.default_rng(data_seq)
            rng_weights = np.random.default_rng(weights_seq)

            process_k = self._clone_process(
                self.process_template, overrides={}, rng=rng_data
            )
            samples = process_k.generate_samples(self.sample_size)

            ekw = dict(self.experiment_kwargs)

            # defaults, but don't overwrite method-specific overrides
            ekw.setdefault('alpha', float(self.alpha))
            ekw.setdefault('transform', str(self.transform))
            ekw.setdefault('transform_power', float(self.transform_power))
            ekw.setdefault('rho_power', float(self.rho_power))
            ekw.setdefault('sample_size', int(self.sample_size))
            ekw.setdefault('burn_in', int(self.burn_in))
            ekw.setdefault('var_warmup', int(self.var_warmup))
            ekw.setdefault('use_variance_smoothing', False)

            method_label = ekw.get('method_label', None)
            boot = ekw.get('bootstrap_cls', None)

            if method_label == 'GaussMix':
                from olbootstrap.online_bootstrap._online_gaussian_bootstrap import (
                    OnlineGaussianMixtureAsympCSSmoothedBootstrap,
                )

                assert (
                    boot is OnlineGaussianMixtureAsympCSSmoothedBootstrap
                ), f"GaussMix mislabeled: bootstrap_cls={getattr(boot,'__name__',boot)}"

            exp_kwargs = dict(ekw)
            exp_kwargs.pop('method_label', None)  # keep for naming, don't pass to init

            exp = OnlineARBootstrapExperiment(
                process=process_k,
                rng_weights=rng_weights,
                **exp_kwargs,
            )
            exp.run(samples=samples)
            exp.compute_intervals(quantile='normal')
            exp.compute_uniform_bands()
            u_mean = getattr(exp, 'uniform_mean_width', float('nan'))
            self.series_uniform_width_mean.append(float(u_mean))

            cov_pw = exp.compute_coverage(band='pointwise')
            pw_mask = cov_pw.get('pointwise_mask', None)
            if pw_mask is None or pw_mask.size == 0:
                self.series_pointwise_frac.append(np.nan)
                self.series_pointwise_all_ok.append(False)
            else:
                self.series_pointwise_frac.append(float(np.mean(pw_mask)))
                self.series_pointwise_all_ok.append(bool(np.all(pw_mask)))

            cov_u = exp.compute_coverage(band='uniform')
            u_mask = cov_u.get('uniform_mask', None)
            if u_mask is None or u_mask.size == 0:
                self.series_uniform_frac.append(np.nan)
                self.series_uniform_ok.append(False)
            else:
                self.series_uniform_frac.append(float(np.mean(u_mask)))
                self.series_uniform_ok.append(bool(np.all(u_mask)))

        self.uniform_over_series_uniform = float(np.mean(self.series_uniform_ok))
        self.uniform_coverage = self.uniform_over_series_uniform
        self.avg_pointwise_time_fraction = float(np.nanmean(self.series_pointwise_frac))
        self.avg_uniform_time_fraction = float(np.nanmean(self.series_uniform_frac))
        self.uniform_over_series_pointwise = float(
            np.mean(self.series_pointwise_all_ok)
        )

        return {
            'series_pointwise_frac': np.asarray(
                self.series_pointwise_frac, dtype=float
            ),
            'series_pointwise_all_ok': np.asarray(
                self.series_pointwise_all_ok, dtype=bool
            ),
            'series_uniform_frac': np.asarray(self.series_uniform_frac, dtype=float),
            'series_uniform_ok': np.asarray(self.series_uniform_ok, dtype=bool),
            'avg_pointwise_time_fraction': self.avg_pointwise_time_fraction,
            'avg_uniform_time_fraction': self.avg_uniform_time_fraction,
            'uniform_over_series_pointwise': self.uniform_over_series_pointwise,
            'uniform_over_series_uniform': self.uniform_over_series_uniform,
            'uniform_coverage': self.uniform_coverage,
            'series_uniform_mean_width': np.asarray(
                self.series_uniform_width_mean, dtype=float
            ),
            'avg_uniform_mean_width': float(np.nanmean(self.series_uniform_width_mean))
            if len(self.series_uniform_width_mean) > 0
            else float('nan'),
        }

    def run_smoothing_sweep(
        self,
        smoothing_grid: Sequence[float],
        save_path: Optional[str] = None,
        parallel: bool = True,
        n_jobs: int = -1,
        verbose: int = 10,
    ) -> dict:
        """Run coverage study for each smoothing value in `smoothing_grid`.

        Args:
            smoothing_grid: Sequence of smoothing `eta` values to evaluate.
            save_path: Optional path to write the sweep (.npz) for this study.
            parallel: If True, run per-eta jobs in parallel using joblib.
            n_jobs: Number of parallel jobs for joblib (default -1 -> all cores).
            verbose: Verbosity passed to joblib.Parallel.

        Returns:
            SweepResults: Dataclass with arrays for eta, avg pointwise/uniform
            time fractions, uniform-over-series summaries and mean widths.
        """
        smoothing_grid = [float(x) for x in smoothing_grid]
        ss = np.random.SeedSequence(self.seed)
        alpha_seeds = ss.spawn(len(smoothing_grid))

        eta_vals, avg_pw_time, avg_u_time = [], [], []
        u_over_series_pw, u_over_series_u = [], []
        transform_val = self.transform

        if parallel:
            job_args = []
            for j, eta in enumerate(smoothing_grid):
                seed_j = int(alpha_seeds[j].entropy)
                job_args.append(
                    (
                        UniformCoverageStudy,
                        float(eta),
                        self.process_template,
                        int(self.sample_size),
                        dict(self.experiment_kwargs),
                        int(self.n_series),
                        int(self.burn_in),
                        int(self.var_warmup),
                        float(self.alpha),
                        int(seed_j),
                        transform_val,
                        float(self.transform_power),
                        float(self.rho_power),
                    )
                )

            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
                delayed(self._run_one_eta_worker_joblib)(*args) for args in job_args
            )

            avg_u_width = []

            for eta_j, res in results:
                eta_vals.append(float(eta_j))
                avg_pw_time.append(res['avg_pointwise_time_fraction'])
                avg_u_time.append(res['avg_uniform_time_fraction'])
                u_over_series_pw.append(res['uniform_over_series_pointwise'])
                u_over_series_u.append(res['uniform_over_series_uniform'])
                avg_u_width.append(res.get('avg_uniform_mean_width', float('nan')))
        else:
            avg_u_width = []
            for j, eta in enumerate(smoothing_grid):
                ekw = dict(self.experiment_kwargs)
                ekw['alpha'] = float(self.alpha)
                ekw['eta'] = float(eta)
                ekw.setdefault('sample_size', int(self.sample_size))
                ekw.setdefault('burn_in', int(self.burn_in))
                ekw.setdefault('var_warmup', int(self.var_warmup))
                ekw.setdefault('transform', str(self.transform))
                ekw.setdefault('use_variance_smoothing', False)

                substudy = UniformCoverageStudy(
                    process_template=self.process_template,
                    sample_size=self.sample_size,
                    experiment_kwargs=ekw,
                    n_series=self.n_series,
                    burn_in=self.burn_in,
                    var_warmup=self.var_warmup,
                    alpha=self.alpha,
                    seed=int(alpha_seeds[j].entropy),
                    progress=self.progress,
                    transform=transform_val,
                    transform_power=float(self.transform_power),
                    rho_power=float(self.rho_power),
                )
                res = substudy.run(position=1)

                eta_vals.append(eta)
                avg_pw_time.append(res['avg_pointwise_time_fraction'])
                avg_u_time.append(res['avg_uniform_time_fraction'])
                u_over_series_pw.append(res['uniform_over_series_pointwise'])
                u_over_series_u.append(res['uniform_over_series_uniform'])
                avg_u_width.append(res.get('avg_uniform_mean_width', float('nan')))

        sweep = SweepResults(
            eta=np.asarray(eta_vals, dtype=float),
            avg_pointwise_time_fraction=np.asarray(avg_pw_time, dtype=float),
            avg_uniform_time_fraction=np.asarray(avg_u_time, dtype=float),
            uniform_over_series_pointwise=np.asarray(u_over_series_pw, dtype=float),
            uniform_over_series_uniform=np.asarray(u_over_series_u, dtype=float),
            avg_uniform_mean_width=np.asarray(avg_u_width, dtype=float),
            alpha=self.alpha,
        )
        self._last_sweep = sweep
        if save_path:
            self.save_sweep(save_path, sweep)
        return sweep

    @classmethod
    def run_sweeps(
        cls,
        base_process_template,
        sample_size: int = 2000,
        dgp_overrides: Optional[List[dict]] = None,
        exp_kwargs_overrides: Optional[List[dict]] = None,
        smoothing_grid: Sequence[float] = (2 / 20, 2 / 50, 2 / 100, 2 / 250, 2 / 500),
        outdir: Union[str, Path] = Path('..') / 'experiments',
        n_series: int = 100,
        burn_in: int = 500,
        alpha: float = 0.05,
        base_exp_kwargs: Optional[dict] = None,
        seed: Optional[int] = None,
        progress: bool = True,
        parallel: bool = True,
        n_jobs: int = -1,
        verbose: int = 10,
        *,
        transform: str = 'student',
        transform_power: float = 1.0 / 3.0,
        rho_power: float = (-1.0 / 3.0),
        var_warmup: int = 0,
        save: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Run sweeps over DGP and experiment overrides.

        Args:
            base_process_template: Process template to clone for each run.
            sample_size (int, optional): Number of samples per simulated series.
            dgp_overrides (Optional[List[dict]], optional): Process parameter
                overrides.
            exp_kwargs_overrides (Optional[List[dict]], optional): Experiment
                keyword overrides.
            smoothing_grid (Sequence[float], optional): Smoothing values.
            outdir (Union[str, Path], optional): Output directory.
            n_series (int, optional): Number of simulated series per setting.
            burn_in (int, optional): Burn-in length.
            alpha (float, optional): Nominal significance level.
            base_exp_kwargs (Optional[dict], optional): Base experiment kwargs.
            seed (Optional[int], optional): Master RNG seed.
            progress (bool, optional): If True, show progress bars.
            parallel (bool, optional): If True, run jobs in parallel.
            n_jobs (int, optional): Number of joblib workers.
            verbose (int, optional): Joblib verbosity.
            transform (str, optional): Multiplier transform.
            transform_power (float, optional): Transform power.
            rho_power (float, optional): Latent correlation exponent.
            var_warmup (int, optional): Variance warmup length.
            save (bool, optional): If True, save sweep files to `outdir`.

        Returns:
            Dict[str, Dict[str, Any]]: Sweep outputs keyed by run name.
        """
        outdir = Path(outdir)
        if save:
            outdir.mkdir(parents=True, exist_ok=True)

        dgp_overrides = dgp_overrides or [{}]
        exp_kwargs_overrides = exp_kwargs_overrides or [{}]
        base_exp_kwargs = base_exp_kwargs or {}

        total_runs = len(dgp_overrides) * len(exp_kwargs_overrides)
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(total_runs)

        results: Dict[str, Dict[str, Any]] = {}
        combos = list(
            product(enumerate(dgp_overrides), enumerate(exp_kwargs_overrides))
        )
        outer_iter = tqdm(combos, desc='combo runs', disable=not progress)

        proc_label = type(base_process_template).__name__

        for combo_idx, ((_, dgp_ov), (i_exp, exp_ov)) in enumerate(outer_iter):
            this_seed = int(child_seeds[combo_idx].entropy)
            rng = np.random.default_rng(this_seed)

            proc_variant = cls._clone_process(base_process_template, dgp_ov, rng)

            ekw = dict(base_exp_kwargs)
            ekw.update(exp_ov)
            ekw.setdefault('alpha', float(alpha))
            ekw.setdefault('transform', str(transform))
            ekw.setdefault('transform_power', float(transform_power))
            ekw.setdefault('rho_power', float(rho_power))
            ekw.setdefault('sample_size', int(sample_size))
            ekw.setdefault('burn_in', int(burn_in))
            ekw.setdefault('var_warmup', int(var_warmup))
            ekw.setdefault('use_variance_smoothing', False)
            effective_tr = ekw.get('transform', transform)

            smooth_part = (
                ekw.get('smoothing_method')
                or ekw.get('smoothing')
                or ekw.get('method')
                or f'exp{i_exp}'
            )
            eta_part = ekw.get('eta', ekw.get('smoothing_alpha', None))
            B_val = int(ekw.get('B', base_exp_kwargs.get('B', 0)))

            is_garch = all(hasattr(proc_variant, a) for a in ('omega', 'alpha', 'beta'))

            if hasattr(proc_variant, 'parameters'):
                try:
                    params = np.asarray(proc_variant.parameters)
                    q = int(getattr(proc_variant, 'q', params.size))
                except Exception:
                    q = getattr(proc_variant, 'q', None) or 'NA'

                dgp_id = f'ma-q{q}'

            elif hasattr(proc_variant, 'phi'):
                phi_arr = np.asarray(proc_variant.phi, dtype=float).reshape(-1)

                if phi_arr.size == 1:
                    # Keep previous AR(1) naming unchanged.
                    dgp_id = f'phi-{cls._safe_str(proc_variant.phi)}'
                else:
                    # Compact AR(p) naming for rebuttal processes.
                    p = int(phi_arr.size)
                    nz = int(np.sum(np.abs(phi_arr) > 1e-12))
                    mass = float(np.sum(phi_arr))
                    dgp_id = f'ar{p}' f'-nz{nz}' f'-mass{cls._safe_str(round(mass, 4))}'

                if bool(getattr(proc_variant, 'structural_break', False)):
                    sb_times = getattr(proc_variant, 'structural_break_times', None)
                    sb_single = getattr(proc_variant, 'break_time', None)
                    sb_random = int(getattr(proc_variant, 'n_structural_breaks', 0))

                    if sb_times is not None:
                        n_sb = len(sb_times)
                    elif sb_single is not None:
                        n_sb = 1
                    elif sb_random > 0:
                        n_sb = sb_random
                    else:
                        n_sb = 1

                    dgp_id += f'__SB{n_sb}'

            elif is_garch:
                # Compact GARCH naming. Do not include omega/alpha/beta or nonlinear settings.
                dgp_id = 'garch'

            else:
                dgp_id = 'dgp'

            # Keep old AR shock naming, but suppress useless default no-shock info for GARCH.
            if any(
                hasattr(proc_variant, k)
                for k in ('shock_type', 'jump_prob', 'jump_scale', 'decay')
            ):
                st = getattr(proc_variant, 'shock_type', 'none')
                jp = getattr(proc_variant, 'jump_prob', 0.0)
                js = getattr(proc_variant, 'jump_scale', 1.0)
                dc = getattr(proc_variant, 'decay', None)

                is_default_no_shock = (
                    str(st) == 'none' and float(jp) == 0.0 and float(js) == 1.0
                )

                if not (is_garch and is_default_no_shock):
                    shock_bits = [
                        f'type-{cls._safe_str(st)}',
                        f'p-{cls._safe_str(jp)}',
                        f'scale-{cls._safe_str(js)}',
                    ]
                    if dc is not None:
                        shock_bits.append(f'decay-{cls._safe_str(dc)}')

                    dgp_id += '__shock-' + '_'.join(shock_bits)

            method_label = ekw.get('method_label', None)

            name_parts: List[str] = [
                f'proc-{cls._safe_str(proc_label)}',
                dgp_id,
                f'smooth-{cls._safe_str(smooth_part)}',
            ]

            if eta_part is not None:
                name_parts.append(f'eta-{cls._safe_str(eta_part)}')

            name_parts += [
                f'tr-{cls._safe_str(effective_tr)}',
                f'bi-{cls._safe_str(burn_in)}',
                f'vw-{cls._safe_str(var_warmup)}',
                f'alpha-{cls._safe_str(alpha)}',
                f'n-{cls._safe_str(sample_size)}',
                f'nseries-{cls._safe_str(n_series)}',
                f'B-{cls._safe_str(B_val)}',
            ]

            if not bool(ekw.get('use_variance_smoothing', True)):
                name_parts.append('nosmooth-var')

            if method_label is not None:
                name_parts.append(f'method-{cls._safe_str(method_label)}')

            default_tp = 1.0 / 3.0
            rel_tol, abs_tol = 1e-12, 1e-12
            tp = float(ekw.get('transform_power', default_tp))
            if abs(tp - default_tp) > max(
                rel_tol * max(abs(tp), abs(default_tp)), abs_tol
            ):
                name_parts.append(f'tp-{cls._safe_str(tp)}')

            trend_slope = getattr(proc_variant, 'trend_slope', None)
            if trend_slope not in (None, 0.0):
                if bool(getattr(proc_variant, 'quadratic_trend', False)):
                    name_parts.append(f'qtrend-{cls._safe_str(trend_slope)}')
                else:
                    # Keep previous linear trend naming unchanged.
                    name_parts.append(f'trend-{cls._safe_str(trend_slope)}')

            sea_amp = getattr(proc_variant, 'seasonal_amplitude', None)
            sea_per = getattr(proc_variant, 'seasonal_period', None)
            if sea_amp not in (None, 0.0):
                name_parts.append(f'seaA-{cls._safe_str(sea_amp)}')
                if sea_per is not None:
                    name_parts.append(f'seaP-{cls._safe_str(sea_per)}')

            name = '__'.join(name_parts)
            save_path = (outdir / f'{name}.npz') if save else None

            study = cls(
                process_template=proc_variant,
                sample_size=sample_size,
                experiment_kwargs=ekw,
                n_series=n_series,
                burn_in=burn_in,
                var_warmup=int(var_warmup),
                alpha=alpha,
                seed=this_seed,
                progress=progress,
                transform=transform,
                transform_power=transform_power,
                rho_power=rho_power,
            )

            sweep = study.run_smoothing_sweep(
                smoothing_grid=smoothing_grid,
                save_path=(str(save_path) if save_path is not None else None),
                parallel=parallel,
                n_jobs=n_jobs,
                verbose=verbose,
            )

            results[name] = {
                'sweep': sweep,
                'path': save_path if save else None,
                'name': name,
            }

        return results


@dataclass
class UniformCoverageTestRun:
    """Container for a coverage run augmented with testing outputs.

    Args:
        series_uniform_ok (np.ndarray): Per-series uniform coverage indicators.
        series_uniform_frac (np.ndarray): Per-series uniform time fractions.
        avg_uniform_time_fraction (float): Average uniform time fraction.
        uniform_over_series_uniform (float): Fraction of series uniformly covered.
        times (np.ndarray): Recorded time indices.
        tau (np.ndarray): First rejection times per series.
        rejected (np.ndarray): Per-series final rejection indicators.
        reject_rate_final (float): Final rejection rate over series.
        power_curve (np.ndarray): Reject-by-time power curve.
        burn_eff (int): Effective burn-in plus variance warmup.
        alpha (float): Nominal significance level.
    """

    series_uniform_ok: np.ndarray
    series_uniform_frac: np.ndarray
    avg_uniform_time_fraction: float
    uniform_over_series_uniform: float
    times: np.ndarray
    tau: np.ndarray
    rejected: np.ndarray
    reject_rate_final: float
    power_curve: np.ndarray
    burn_eff: int
    alpha: float


class UniformCoverageStudyWithTesting(UniformCoverageStudy):
    """Extends UniformCoverageStudy by computing hypothesis testing quantities."""

    @staticmethod
    def _tau_from_uniform_band(
        *,
        times: np.ndarray,
        uL_all: np.ndarray,
        uU_all: np.ndarray,
        target_at_recorded: np.ndarray,
        burn_eff: int,
    ) -> float:
        """Compute the first time a uniform band excludes the target.

        Args:
            times (np.ndarray): Recorded time indices.
            uL_all (np.ndarray): Uniform lower bounds at recorded times.
            uU_all (np.ndarray): Uniform upper bounds at recorded times.
            target_at_recorded (np.ndarray): Target values at recorded times.
            burn_eff (int): Effective burn-in plus variance warmup.

        Returns:
            float: First rejection time, or infinity if no rejection occurs.
        """
        times = np.asarray(times, dtype=int)
        uL_all = np.asarray(uL_all, dtype=float)
        uU_all = np.asarray(uU_all, dtype=float)
        target_at_recorded = np.asarray(target_at_recorded, dtype=float)

        T = min(times.size, uL_all.size, uU_all.size, target_at_recorded.size)
        times = times[:T]
        uL_all = uL_all[:T]
        uU_all = uU_all[:T]
        target_at_recorded = target_at_recorded[:T]

        ok = (
            (times > int(burn_eff))
            & np.isfinite(uL_all)
            & np.isfinite(uU_all)
            & np.isfinite(target_at_recorded)
        )
        if not np.any(ok):
            return float('inf')

        out = ok & ((target_at_recorded < uL_all) | (target_at_recorded > uU_all))
        if not np.any(out):
            return float('inf')

        return float(times[np.argmax(out)])

    @staticmethod
    def _power_curve_from_tau(times: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """Convert first rejection times to a reject-by-time power curve.

        Args:
            times (np.ndarray): Recorded time indices.
            tau (np.ndarray): First rejection times per series.

        Returns:
            np.ndarray: Empirical rejection probability at each recorded time.
        """
        times = np.asarray(times, dtype=int)
        tau = np.asarray(tau, dtype=float)
        return np.mean(tau[:, None] <= times[None, :], axis=0)

    def run(self, position: int = 0, leave_inner: bool = False) -> Dict[str, Any]:
        """Run coverage and testing replications.

        Args:
            position (int, optional): Progress-bar display position.
            leave_inner (bool, optional): If True, leave nested progress bars.

        Returns:
            Dict[str, Any]: Coverage summaries plus tau, rejection, and power data.

        Raises:
            RuntimeError: If recorded times differ across replications.
        """
        out_cov = super().run(position=position, leave_inner=leave_inner)
        master = np.random.SeedSequence(self.seed)
        rep_seqs = master.spawn(self.n_series)

        tau_list: List[float] = []
        times_ref: Optional[np.ndarray] = None

        rep_iter = (
            tqdm(
                range(self.n_series),
                desc='testing (tau)',
                total=self.n_series,
                position=position,
                leave=leave_inner,
                dynamic_ncols=True,
                mininterval=0.1,
            )
            if self.progress
            else range(self.n_series)
        )

        for k in rep_iter:
            data_seq, weights_seq = rep_seqs[k].spawn(2)
            rng_data = np.random.default_rng(data_seq)
            rng_weights = np.random.default_rng(weights_seq)

            process_k = self._clone_process(
                self.process_template, overrides={}, rng=rng_data
            )
            samples = process_k.generate_samples(self.sample_size)

            ekw = dict(self.experiment_kwargs)
            ekw.setdefault('alpha', float(self.alpha))
            ekw.setdefault('transform', str(self.transform))
            ekw.setdefault('transform_power', float(self.transform_power))
            ekw.setdefault('rho_power', float(self.rho_power))
            ekw.setdefault('sample_size', int(self.sample_size))
            ekw.setdefault('burn_in', int(self.burn_in))
            ekw.setdefault('var_warmup', int(self.var_warmup))
            ekw.setdefault('use_variance_smoothing', False)

            exp_kwargs = dict(ekw)
            exp_kwargs.pop('method_label', None)

            exp = OnlineARBootstrapExperiment(
                process=process_k,
                rng_weights=rng_weights,
                **exp_kwargs,
            )
            exp.run(samples=samples)

            times = np.asarray(exp.times, dtype=int)
            uL_all = np.asarray(
                getattr(exp, '_uniform_lower_all', np.full(times.shape, np.nan)),
                dtype=float,
            )
            uU_all = np.asarray(
                getattr(exp, '_uniform_upper_all', np.full(times.shape, np.nan)),
                dtype=float,
            )
            target_full = exp._smoother_target_series(testing=True)  # length n
            target_at_recorded = np.asarray(target_full, float)[times - 1]

            burn_eff = int(exp.t0) + int(exp.var_warmup)

            tau_k = self._tau_from_uniform_band(
                times=times,
                uL_all=uL_all,
                uU_all=uU_all,
                target_at_recorded=target_at_recorded,
                burn_eff=burn_eff,
            )
            tau_list.append(tau_k)

            if times_ref is None:
                times_ref = times.copy()
            else:
                if times_ref.shape != times.shape or not np.all(times_ref == times):
                    raise RuntimeError(
                        'Recorded times differ across reps; cannot build a single power curve.'
                    )

        tau = np.asarray(tau_list, dtype=float)
        assert times_ref is not None
        times = times_ref

        rejected = np.isfinite(tau)
        reject_rate_final = float(np.mean(rejected))
        power_curve = self._power_curve_from_tau(times, tau)

        burn_eff = int(self.burn_in) + int(self.var_warmup)

        return {
            **out_cov,
            'times': times,
            'tau': tau,
            'rejected': rejected,
            'reject_rate_final': reject_rate_final,
            'power_curve': power_curve,
            'burn_eff': burn_eff,
            'alpha': float(self.alpha),
        }

    def run_with_testing(
        self, position: int = 0, leave_inner: bool = False
    ) -> UniformCoverageTestRun:
        """Return testing outputs as a structured dataclass.

        Args:
            position (int, optional): Progress-bar display position.
            leave_inner (bool, optional): If True, leave nested progress bars.

        Returns:
            UniformCoverageTestRun: Structured coverage and testing outputs.
        """
        d = self.run(position=position, leave_inner=leave_inner)
        return UniformCoverageTestRun(
            series_uniform_ok=np.asarray(d['series_uniform_ok'], dtype=bool),
            series_uniform_frac=np.asarray(d['series_uniform_frac'], dtype=float),
            avg_uniform_time_fraction=float(d['avg_uniform_time_fraction']),
            uniform_over_series_uniform=float(d['uniform_over_series_uniform']),
            times=np.asarray(d['times'], dtype=int),
            tau=np.asarray(d['tau'], dtype=float),
            rejected=np.asarray(d['rejected'], dtype=bool),
            reject_rate_final=float(d['reject_rate_final']),
            power_curve=np.asarray(d['power_curve'], dtype=float),
            burn_eff=int(d['burn_eff']),
            alpha=float(d['alpha']),
        )

    @classmethod
    def _save_power_run(
        cls,
        *,
        path: Union[str, Path],
        result: Any,
        meta: Dict[str, Any],
    ) -> None:
        """Save a single power/coverage+testing run to npz.

        Args:
            path (Union[str, Path]): Output `.npz` path.
            result (Any): Object returned by `run_with_testing`.
            meta (Dict[str, Any]): Metadata to store with the run.

        Returns:
            None
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            str(path),
            times=np.asarray(result.times, dtype=int),
            power_curve=np.asarray(result.power_curve, dtype=float),
            tau=np.asarray(result.tau, dtype=float),
            rejected=np.asarray(result.rejected, dtype=bool),
            series_uniform_ok=np.asarray(result.series_uniform_ok, dtype=bool),
            series_uniform_frac=np.asarray(result.series_uniform_frac, dtype=float),
            uniform_over_series_uniform=float(result.uniform_over_series_uniform),
            avg_uniform_time_fraction=float(result.avg_uniform_time_fraction),
            reject_rate_final=float(result.reject_rate_final),
            burn_eff=int(result.burn_eff),
            alpha=float(result.alpha),
            meta=np.array(meta, dtype=object),
        )

    @classmethod
    def run_power_sweep(
        cls,
        *,
        base_process_template,
        etas: Sequence[float],
        phis: Sequence[float],
        trend_slopes: Sequence[float],
        sample_size: int = 3500,
        n_series: int = 100,
        burn_in: int = 500,
        var_warmup: int = 400,
        alpha: float = 0.1,
        base_exp_kwargs: Optional[dict] = None,
        outdir: Union[str, Path] = Path('..') / 'experiments_power',
        seed: Optional[int] = None,
        progress: bool = True,
        parallel: bool = True,
        n_jobs: int = -1,
        verbose: int = 10,
        transform: str = 'student',
        transform_power: float = 1.0 / 3.0,
        rho_power: float = (-1.0 / 3.0),
        save: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        """Run power experiments over eta, phi, and trend grids.

        Args:
            base_process_template: Process template to clone for each run.
            etas (Sequence[float]): Smoothing values to evaluate.
            phis (Sequence[float]): AR coefficient values to evaluate.
            trend_slopes (Sequence[float]): Trend slopes to evaluate.
            sample_size (int, optional): Number of samples per simulated series.
            n_series (int, optional): Number of simulated series per setting.
            burn_in (int, optional): Burn-in length.
            var_warmup (int, optional): Variance warmup length.
            alpha (float, optional): Nominal significance level.
            base_exp_kwargs (Optional[dict], optional): Base experiment kwargs.
            outdir (Union[str, Path], optional): Output directory.
            seed (Optional[int], optional): Master RNG seed.
            progress (bool, optional): If True, show progress bars.
            parallel (bool, optional): If True, run jobs in parallel.
            n_jobs (int, optional): Number of joblib workers.
            verbose (int, optional): Joblib verbosity.
            transform (str, optional): Multiplier transform.
            transform_power (float, optional): Transform power.
            rho_power (float, optional): Latent correlation exponent.
            save (bool, optional): If True, save power files to `outdir`.

        Returns:
            Dict[str, Dict[str, Any]]: Power outputs keyed by run name.
        """
        outdir = Path(outdir)
        if save:
            outdir.mkdir(parents=True, exist_ok=True)

        base_exp_kwargs = {} if base_exp_kwargs is None else dict(base_exp_kwargs)

        etas = [float(x) for x in etas]
        phis = [float(x) for x in phis]
        trend_slopes = [float(x) for x in trend_slopes]

        n_total = len(etas) * len(phis) * len(trend_slopes)
        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n_total)

        jobs: List[Tuple[float, float, float, int]] = []
        idx = 0
        for eta in etas:
            for phi in phis:
                for ts in trend_slopes:
                    jobs.append((eta, phi, ts, int(child_seeds[idx].entropy)))
                    idx += 1

        proc_label = type(base_process_template).__name__

        def _run_one(eta: float, phi: float, ts: float, cfg_seed: int):
            """Run one power-grid configuration.

            Args:
                eta (float): Smoothing value.
                phi (float): AR coefficient value.
                ts (float): Trend slope value.
                cfg_seed (int): Configuration-specific RNG seed.

            Returns:
                tuple: `(name, result, save_path, meta)` for the configuration.
            """
            rng = np.random.default_rng(cfg_seed)
            proc = cls._clone_process(
                base_process_template,
                overrides={'phi': phi, 'trend_slope': ts},
                rng=rng,
            )

            ekw = dict(base_exp_kwargs)
            ekw['eta'] = float(eta)
            ekw.setdefault('alpha', float(alpha))
            ekw.setdefault('transform', str(transform))
            ekw.setdefault('transform_power', float(transform_power))
            ekw.setdefault('rho_power', float(rho_power))
            ekw.setdefault('sample_size', int(sample_size))
            ekw.setdefault('burn_in', int(burn_in))
            ekw.setdefault('var_warmup', int(var_warmup))
            ekw.setdefault('use_variance_smoothing', False)

            method_label = ekw.get('method_label', None)
            exp_kwargs = dict(ekw)
            exp_kwargs.pop('method_label', None)

            name_parts = [
                f'proc-{cls._safe_str(proc_label)}',
                f'phi-{cls._safe_str(phi)}',
                f'trend-{cls._safe_str(ts)}',
                f"smooth-{cls._safe_str(exp_kwargs.get('smoothing_method', 'NA'))}",
                f'eta-{cls._safe_str(eta)}',
                f"tr-{cls._safe_str(exp_kwargs.get('transform', transform))}",
                f'bi-{cls._safe_str(burn_in)}',
                f'vw-{cls._safe_str(var_warmup)}',
                f'alpha-{cls._safe_str(alpha)}',
                f'n-{cls._safe_str(sample_size)}',
                f'nseries-{cls._safe_str(n_series)}',
                f"B-{cls._safe_str(int(exp_kwargs.get('B', 0)))}",
            ]
            if not bool(exp_kwargs.get('use_variance_smoothing', True)):
                name_parts.append('nosmooth-var')
            if method_label is not None:
                name_parts.append(f'method-{cls._safe_str(method_label)}')
            seaA = getattr(proc, 'seasonal_amplitude', 0.0)
            seaP = getattr(proc, 'seasonal_period', None)
            if seaA not in (None, 0.0):
                name_parts.append(f'seaA-{cls._safe_str(seaA)}')
                name_parts.append(f'seaP-{cls._safe_str(seaP)}')

            name = '__'.join(name_parts)
            save_path = (outdir / f'{name}.npz') if save else None

            study = cls(
                process_template=proc,
                sample_size=int(sample_size),
                experiment_kwargs=ekw,
                n_series=int(n_series),
                burn_in=int(burn_in),
                var_warmup=int(var_warmup),
                alpha=float(alpha),
                seed=int(cfg_seed),
                progress=False,
                transform=str(transform),
                transform_power=float(transform_power),
                rho_power=float(rho_power),
            )

            result = study.run_with_testing()

            meta = {
                'name': name,
                'seed': int(cfg_seed),
                'eta': float(eta),
                'phi': float(phi),
                'trend_slope': float(ts),
                'sample_size': int(sample_size),
                'n_series': int(n_series),
                'burn_in': int(burn_in),
                'var_warmup': int(var_warmup),
                'alpha': float(alpha),
                'experiment_kwargs': dict(ekw),
            }

            if save_path is not None:
                cls._save_power_run(path=save_path, result=result, meta=meta)

            return name, result, save_path, meta

        if parallel:
            it = jobs
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=verbose)(
                delayed(_run_one)(eta, phi, ts, cfg_seed)
                for (eta, phi, ts, cfg_seed) in it
            )
        else:
            iterator = tqdm(jobs, desc='power grid', disable=not progress)
            results = [
                _run_one(eta, phi, ts, cfg_seed)
                for (eta, phi, ts, cfg_seed) in iterator
            ]

        out: Dict[str, Dict[str, Any]] = {}
        for name, result, save_path, meta in results:
            out[name] = {
                'result': result,
                'path': save_path,
                'name': name,
                'meta': meta,
            }

        return out
