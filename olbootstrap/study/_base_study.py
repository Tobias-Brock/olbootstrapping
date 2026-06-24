from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from olbootstrap.experiments._experiments import OnlineARBootstrapExperiment


@dataclass
class SweepResults:
    """Container for coverage sweep summaries over smoothing rates.

    Args:
        eta (np.ndarray): Smoothing rates evaluated in the sweep.
        avg_pointwise_time_fraction (np.ndarray): Average pointwise coverage
            fraction over time.
        avg_uniform_time_fraction (np.ndarray): Average uniform coverage fraction
            over time.
        uniform_over_series_pointwise (np.ndarray): Fraction of series with full
            pointwise coverage over time.
        uniform_over_series_uniform (np.ndarray): Fraction of series with full
            uniform coverage over time.
        avg_uniform_mean_width (np.ndarray): Average uniform band widths.
        alpha (Optional[float], optional): Nominal significance level.
    """

    eta: np.ndarray
    avg_pointwise_time_fraction: np.ndarray
    avg_uniform_time_fraction: np.ndarray
    uniform_over_series_pointwise: np.ndarray
    uniform_over_series_uniform: np.ndarray
    avg_uniform_mean_width: np.ndarray
    alpha: Optional[float] = None


class BaseCoverageStudy(ABC):
    """Abstract parent for coverage studies (uniform/time-simultaneous)."""

    def __init__(
        self,
        *,
        process_template,
        sample_size: int,
        experiment_kwargs: Dict[str, Any],
        n_series: int = 100,
        burn_in: int = 0,
        var_warmup: int = 0,
        alpha: float = 0.05,
        seed: Optional[int] = None,
        progress: bool = False,
        transform: Optional[str] = 'student',
        transform_power: float = 1.0 / 3.0,
        rho_power: float = (-1.0 / 3.0),
    ):
        """Initialize study settings and storage.

        Args:
            process_template: Prototype process instance (used for introspection
                and cloning).
            sample_size: Number of time points to generate per series.
            experiment_kwargs: Keyword arguments forwarded to experiment runner.
            n_series: Number of independent series to simulate.
            burn_in: Burn-in count excluded from evaluation.
            var_warmup: Additional warmup for variance/threshold scheduling.
            alpha: Nominal significance level for calibration.
            seed: Optional RNG seed for reproducibility.
            progress: If True, show progress indicators.
            transform: Multiplier transform name ('student'|'gauss') or None.
            transform_power: Power mapping effective sample size to df.
            rho_power: Exponent used for latent AR correlation scaling.
        """
        self.process_template = process_template
        self.sample_size = int(sample_size)
        self.experiment_kwargs = dict(experiment_kwargs)
        self.n_series = int(n_series)
        self.burn_in = int(burn_in)
        self.var_warmup = int(var_warmup)
        self.alpha = float(alpha)
        self.seed = seed
        self.progress = bool(progress)
        self.transform = transform if transform is None else str(transform)
        self.transform_power = float(transform_power)
        self.rho_power = float(rho_power)

        self.series_uniform_ok: List[bool] = []
        self.series_pointwise_frac: List[float] = []
        self.series_pointwise_all_ok: List[bool] = []
        self.series_uniform_frac: List[float] = []
        self.series_uniform_width_mean: List[float] = []

        self.uniform_coverage: Optional[float] = None
        self.avg_pointwise_time_fraction: Optional[float] = None
        self.avg_uniform_time_fraction: Optional[float] = None
        self.uniform_over_series_pointwise: Optional[float] = None
        self.uniform_over_series_uniform: Optional[float] = None
        self._last_sweep: Optional[Dict[str, Any]] = None

    @staticmethod
    def _safe_str(x: Any) -> str:
        """Convert an object to a filename-safe compact string.

        Args:
            x (Any): Object to format.

        Returns:
            str: Filename-safe string representation.
        """
        if x is None:
            return 'NA'
        if isinstance(x, (list, tuple, np.ndarray)):
            return '-'.join(BaseCoverageStudy._safe_str(v) for v in x)
        s = f'{x:g}' if isinstance(x, float) else str(x)
        return s.replace('.', 'p').replace('-', 'm').replace(' ', '')

    @staticmethod
    def _build_process_kwargs(base, overrides: dict, rng) -> Dict[str, Any]:
        """Build constructor kwargs for supported process templates.

        Args:
            base: Template process instance used for introspection.
            overrides: Dict of parameter overrides (takes precedence).
            rng: RNG instance to pass into the new process.

        Returns:
            Dict[str, Any]: Keyword arguments suitable for instantiating the
            same process class as `base`.

        Raises:
            TypeError: If the template type is unsupported.
        """
        ov = overrides or {}

        def add_common_time_parts(d: Dict[str, Any]):
            """Add shared mean, trend, seasonal, and RNG kwargs.

            Args:
                d (Dict[str, Any]): Keyword dictionary to mutate.

            Returns:
                Dict[str, Any]: Mutated keyword dictionary.
            """
            d['mean'] = ov.get('mean', float(getattr(base, 'mean', 0.0)))
            d['trend_slope'] = ov.get(
                'trend_slope', float(getattr(base, 'trend_slope', 0.0))
            )
            d['seasonal_amplitude'] = ov.get(
                'seasonal_amplitude', float(getattr(base, 'seasonal_amplitude', 0.0))
            )
            sp = getattr(base, 'seasonal_period', None)
            d['seasonal_period'] = ov.get(
                'seasonal_period', (None if sp is None else float(sp))
            )
            d['seasonal_phase'] = ov.get(
                'seasonal_phase', float(getattr(base, 'seasonal_phase', 0.0))
            )
            d['rng'] = rng
            return d

        def maybe_add_noise_std(d: Dict[str, Any]):
            """Add noise standard deviation if the process supports it.

            Args:
                d (Dict[str, Any]): Keyword dictionary to mutate.

            Returns:
                Dict[str, Any]: Mutated keyword dictionary.
            """
            d['noise_std'] = ov.get('noise_std', float(getattr(base, 'noise_std', 1.0)))
            return d

        def maybe_add_shock_kwargs(d: Dict[str, Any]):
            """Add shock-related kwargs when present on the template.

            Args:
                d (Dict[str, Any]): Keyword dictionary to mutate.

            Returns:
                Dict[str, Any]: Mutated keyword dictionary.
            """
            shock_keys = (
                'shock_type',
                'jump_prob',
                'jump_scale',
                'decay',
                'jump_times',
                'jump_sizes',
            )
            has_any = any(hasattr(base, k) for k in shock_keys)
            if not has_any and not any(k in ov for k in shock_keys):
                return d
            for k in shock_keys:
                if k in ov:
                    d[k] = ov[k]
                elif hasattr(base, k):
                    d[k] = getattr(base, k)
            return d

        if hasattr(base, 'parameters') and hasattr(base, 'q'):
            theta = np.asarray(base.parameters, dtype=float)
            theta_arg = (
                theta[1:] if (theta.size > 0 and np.isclose(theta[0], 1.0)) else theta
            )

            kwargs = {}
            add_common_time_parts(kwargs)
            maybe_add_noise_std(kwargs)
            maybe_add_shock_kwargs(kwargs)

            kwargs['parameters'] = ov.get('parameters', theta_arg)
            return kwargs

        if hasattr(base, 'phi'):
            kwargs = {}
            add_common_time_parts(kwargs)
            maybe_add_noise_std(kwargs)
            maybe_add_shock_kwargs(kwargs)

            phi_val = ov.get('phi', base.phi)
            phi_arr = np.asarray(phi_val, dtype=float)

            if phi_arr.ndim == 0 or phi_arr.size == 1:
                kwargs['phi'] = float(phi_arr.reshape(-1)[0])
            else:
                kwargs['phi'] = phi_arr.reshape(-1).copy()
            return kwargs

        if all(hasattr(base, a) for a in ('omega', 'alpha', 'beta')):
            kwargs = {}
            add_common_time_parts(kwargs)
            maybe_add_noise_std(kwargs)
            maybe_add_shock_kwargs(kwargs)

            kwargs['omega'] = float(ov.get('omega', base.omega))
            kwargs['alpha'] = float(ov.get('alpha', base.alpha))
            kwargs['beta'] = float(ov.get('beta', base.beta))
            return kwargs

        raise TypeError(f'Unsupported process type: {type(base).__name__}')

    @staticmethod
    def _clone_process(base, overrides: dict, rng):
        """Instantiate the same class as `base` with overrides and fresh RNG.

        Args:
            base: Template process instance.
            overrides: Parameter overrides dict.
            rng: RNG instance to pass to the new process.

        Returns:
            An instance of type(base) constructed with built kwargs.
        """
        return type(base)(
            **BaseCoverageStudy._build_process_kwargs(base, overrides, rng)
        )

    @staticmethod
    def _centers_sds_from_replicates(
        exp: 'OnlineARBootstrapExperiment',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract centers and standard deviations from an experiment.

        Args:
            exp: An OnlineARBootstrapExperiment with recorded replicates.

        Returns:
            (centers, sds): Tuple of arrays of length T containing the smoothed
            centers and corresponding standard deviations per recorded time.
        """
        T = len(exp.replicates_by_time)
        centers = np.empty(T, dtype=float)
        sds = np.empty(T, dtype=float)
        for t in range(T):
            rep_vals = np.asarray(exp.replicates_by_time[t], dtype=float).reshape(-1)
            centers[t] = float(exp.smooth_estimates[t])
            sd = float(np.std(rep_vals, ddof=0))
            sds[t] = sd if np.isfinite(sd) else 0.0
        return centers, sds

    @staticmethod
    def _run_one_eta_worker_joblib(
        study_cls,
        eta: float,
        process_template,
        sample_size: int,
        experiment_kwargs: Dict[str, Any],
        n_series: int,
        burn_in: int,
        var_warmup: int,
        alpha: float,
        seed: int,
        transform: Optional[str],
        transform_power: float,
        rho_power: float,
    ):
        """Joblib-friendly worker that runs a single eta sweep.

        Args:
            study_cls: CoverageStudy subclass to instantiate.
            eta: Smoothing parameter for this worker.
            process_template: Template process instance to clone.
            sample_size: Number of time points per series.
            experiment_kwargs: Experiment kwargs forwarded to the runner.
            n_series: Number of series to simulate.
            burn_in: Burn-in count.
            var_warmup: Variance warmup.
            alpha: Nominal significance level.
            seed: RNG seed for reproducibility.
            transform: Multiplier transform name.
            transform_power: Power mapping effective sample size to df.
            rho_power: Exponent used for latent AR correlation scaling.

        Returns:
            Tuple[float, Dict]: (eta, result_dict) where result_dict is the
            return value of substudy.run(...).
        """
        ekw = dict(experiment_kwargs)
        ekw['eta'] = float(eta)
        ekw.setdefault('sample_size', int(sample_size))
        ekw.setdefault('burn_in', int(burn_in))
        ekw.setdefault('var_warmup', int(var_warmup))
        ekw.setdefault('transform', None if transform is None else str(transform))

        ekw.setdefault('transform_power', float(transform_power))
        ekw['alpha'] = float(alpha)
        ekw.setdefault('use_variance_smoothing', False)
        ekw.setdefault('rho_power', float(rho_power))

        effective_transform = ekw.get('transform', transform)

        substudy = study_cls(
            process_template=process_template,
            sample_size=sample_size,
            experiment_kwargs=ekw,
            n_series=n_series,
            burn_in=burn_in,
            var_warmup=var_warmup,
            alpha=alpha,
            seed=seed,
            progress=False,
            transform=effective_transform,  # KEY FIX
            transform_power=transform_power,
            rho_power=rho_power,
        )
        return eta, substudy.run(position=0, leave_inner=False)

    @staticmethod
    def save_sweep(path: Union[str, Path], sweep: SweepResults) -> None:
        """Save any SweepResults to `path` using compressed npz format.

        Args:
            path: Output filename (string or Path).
            sweep: SweepResults dataclass instance to serialize.
        """
        path = str(path)
        np.savez_compressed(path, **asdict(sweep))

    @classmethod
    def load_sweep(cls, path: Union[str, Path]) -> SweepResults:
        """Load a SweepResults object from a compressed npz file.

        Args:
            path: Path to the saved npz file.

        Returns:
            SweepResults: Reconstructed sweep dataclass (backwards-compatible).
        """
        path = str(path)
        data = np.load(path, allow_pickle=False)
        eta = np.asarray(data['eta'])
        avg_pw = np.asarray(data['avg_pointwise_time_fraction'])
        avg_u = np.asarray(data['avg_uniform_time_fraction'])
        u_over_pw = np.asarray(data['uniform_over_series_pointwise'])
        u_over_u = np.asarray(data['uniform_over_series_uniform'])
        alpha = float(data['alpha']) if 'alpha' in data else None
        if 'avg_uniform_mean_width' in data:
            avg_width = np.asarray(data['avg_uniform_mean_width'])
        else:
            avg_width = np.full_like(eta, np.nan, dtype=float)

        return SweepResults(
            eta=eta,
            avg_pointwise_time_fraction=avg_pw,
            avg_uniform_time_fraction=avg_u,
            uniform_over_series_pointwise=u_over_pw,
            uniform_over_series_uniform=u_over_u,
            avg_uniform_mean_width=avg_width,
            alpha=alpha,
        )

    @abstractmethod
    def run(self, position: int = 0, leave_inner: bool = False) -> Dict[str, Any]:
        """Run the study and return coverage summaries.

        Args:
            position (int, optional): Progress-bar display position.
            leave_inner (bool, optional): If True, leave nested progress bars.

        Returns:
            Dict[str, Any]: Coverage summary dictionary.
        """
        ...

    @abstractmethod
    def run_smoothing_sweep(
        self,
        smoothing_grid: Sequence[float],
        save_path: Optional[str] = None,
        parallel: bool = True,
        n_jobs: int = -1,
        verbose: int = 10,
    ) -> dict:
        """Run a smoothing-parameter sweep.

        Args:
            smoothing_grid (Sequence[float]): Smoothing values to evaluate.
            save_path (Optional[str], optional): Optional path for saved sweep.
            parallel (bool, optional): If True, run sweep jobs in parallel.
            n_jobs (int, optional): Number of joblib workers.
            verbose (int, optional): Joblib verbosity.

        Returns:
            dict: Sweep result object or mapping.
        """
        ...

    @classmethod
    @abstractmethod
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
    ) -> Dict[str, Dict[str, Any]]:
        """Run a collection of smoothing sweeps over DGP and experiment settings.

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

        Returns:
            Dict[str, Dict[str, Any]]: Sweep outputs keyed by run name.
        """
        ...
