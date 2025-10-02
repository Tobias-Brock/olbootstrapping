from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

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
            ekw.setdefault('alpha', float(self.alpha))
            ekw['transform'] = str(self.transform)
            ekw.setdefault('transform_power', float(self.transform_power))
            ekw.setdefault('sample_size', int(self.sample_size))
            ekw.setdefault('burn_in', int(self.burn_in))
            ekw.setdefault('var_warmup', int(self.var_warmup))
            ekw.setdefault('use_variance_smoothing', True)

            exp = OnlineARBootstrapExperiment(
                process=process_k,
                rng_weights=rng_weights,
                **ekw,
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
                        UniformCoverageStudy,  # pass the study class
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
                ekw['transform'] = str(transform_val)
                ekw.setdefault('use_variance_smoothing', True)

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
        var_warmup: int = 0,
        save: bool = True,  # <-- new flag
    ) -> Dict[str, Dict[str, Any]]:
        """Run sweeps over DGP and experiment overrides.

        Args:
            base_process_template: Prototype process instance to clone/modify.
            sample_size: Number of time points per series.
            dgp_overrides: List of dicts with DGP parameter overrides.
            exp_kwargs_overrides: List of dicts with experiment kwargs overrides.
            smoothing_grid: Grid of smoothing parameters to evaluate.
            outdir: Directory where sweep files will be stored.
            n_series: Number of series per sweep.
            burn_in: Burn-in length for experiments.
            alpha: Nominal significance level.
            base_exp_kwargs: Base experiment kwargs applied before overrides.
            seed: RNG seed for reproducibility.
            progress: If True, show progress bars.
            parallel: If True, allow internal parallelization for smoothing sweep.
            n_jobs: Number of workers for parallel execution.
            verbose: Verbosity level passed to joblib.
            transform: Multiplier transform to use in experiments.
            transform_power: Power mapping effective sample size to df.
            var_warmup: Variance warmup length.
            save: If True save each sweep to disk (default True).

        Returns:
            Dict[str, Dict[str, Any]]: Mapping of sweep-name -> {sweep, path, name}.
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
            ekw.setdefault('sample_size', int(sample_size))
            ekw.setdefault('burn_in', int(burn_in))
            ekw.setdefault('var_warmup', int(var_warmup))
            ekw.setdefault('use_variance_smoothing', True)

            smooth_part = (
                ekw.get('smoothing_method')
                or ekw.get('smoothing')
                or ekw.get('method')
                or f'exp{i_exp}'
            )
            eta_part = ekw.get('eta', ekw.get('smoothing_alpha', None))
            B_val = int(ekw.get('B', base_exp_kwargs.get('B', 0)))

            if hasattr(proc_variant, 'parameters'):
                try:
                    params = np.asarray(proc_variant.parameters)
                    q = int(getattr(proc_variant, 'q', params.size))
                except Exception:
                    q = getattr(proc_variant, 'q', None) or 'NA'
                dgp_id = f'ma-q{q}'

            elif hasattr(proc_variant, 'phi'):
                dgp_id = f'phi-{cls._safe_str(proc_variant.phi)}'
            elif all(hasattr(proc_variant, a) for a in ('omega', 'alpha', 'beta')):
                dgp_id = 'garch-' + cls._safe_str(
                    {
                        'omega': proc_variant.omega,
                        'alpha': proc_variant.alpha,
                        'beta': proc_variant.beta,
                    }
                )
            else:
                dgp_id = 'dgp'

            if any(
                hasattr(proc_variant, k)
                for k in ('shock_type', 'jump_prob', 'jump_scale', 'decay')
            ):
                st = getattr(proc_variant, 'shock_type', 'none')
                jp = getattr(proc_variant, 'jump_prob', 0.0)
                js = getattr(proc_variant, 'jump_scale', 1.0)
                dc = getattr(proc_variant, 'decay', None)
                shock_bits = [
                    f'type-{cls._safe_str(st)}',
                    f'p-{cls._safe_str(jp)}',
                    f'scale-{cls._safe_str(js)}',
                ]
                if dc is not None:
                    shock_bits.append(f'decay-{cls._safe_str(dc)}')
                dgp_id += '__shock-' + '_'.join(shock_bits)

            name_parts: List[str] = [
                f'proc-{cls._safe_str(proc_label)}',
                dgp_id,
                f'smooth-{cls._safe_str(smooth_part)}',
            ]
            if eta_part is not None:
                name_parts.append(f'eta-{cls._safe_str(eta_part)}')
            name_parts += [
                f'tr-{cls._safe_str(transform)}',
                f'bi-{cls._safe_str(burn_in)}',
                f'vw-{cls._safe_str(var_warmup)}',
                f'alpha-{cls._safe_str(alpha)}',
                f'n-{cls._safe_str(sample_size)}',
                f'nseries-{cls._safe_str(n_series)}',
                f'B-{cls._safe_str(B_val)}',
            ]
            if not bool(ekw.get('use_variance_smoothing', True)):
                name_parts.append('nosmooth-var')

            default_tp = 1.0 / 3.0
            rel_tol, abs_tol = 1e-12, 1e-12
            tp = float(ekw.get('transform_power', default_tp))
            if abs(tp - default_tp) > max(
                rel_tol * max(abs(tp), abs(default_tp)), abs_tol
            ):
                name_parts.append(f'tp-{cls._safe_str(tp)}')

            trend_slope = getattr(proc_variant, 'trend_slope', None)
            if trend_slope not in (None, 0.0):
                name_parts.append(f'trend-{cls._safe_str(trend_slope)}')

            sea_amp = getattr(proc_variant, 'seasonal_amplitude', None)
            sea_per = getattr(proc_variant, 'seasonal_period', None)
            if sea_amp not in (None, 0.0):
                name_parts.append(f'seaA-{cls._safe_str(sea_amp)}')
                if sea_per is not None:
                    name_parts.append(f'seaP-{cls._safe_str(sea_per)}')

            name = '__'.join(name_parts)
            save_path = (outdir / f'sweep_{name}.npz') if save else None

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
