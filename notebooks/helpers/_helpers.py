from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from olbootstrap.study._study import (
    UniformCoverageStudy,
)


def run_sweeps_for_process_list(
    processes,
    *,
    labels=None,
    outdir=Path('..') / 'experiments',
    seed=None,
    sample_size=2000,
    smoothing_grid=(2 / 20, 2 / 50, 2 / 100, 2 / 250, 2 / 500),
    n_series=100,
    burn_in=500,
    alpha=0.05,
    base_exp_kwargs=None,
    exp_kwargs_overrides=None,
    progress=True,
    parallel=True,
    n_jobs=-1,
    var_warmup=0,
    transform='student',
    transform_power=(1.0 / 3.0),
    rho_power=-(1.0 / 3.0),
    run_sweeps_kwargs=None,
    study_cls=None,
):
    """Run smoothing sweeps for a list of process templates.

    Args:
        processes (Sequence[Any]): Process templates to pass to the study runner.
        labels (Optional[Sequence[str]], optional): Panel labels aligned with
            ``processes``. Defaults to class names when omitted.
        outdir (Union[str, Path], optional): Directory where per-process outputs are
            written. Defaults to ``../experiments``.
        seed (Optional[int], optional): Master seed used to derive per-process seeds.
            Defaults to None.
        sample_size (int, optional): Number of observations per simulated series.
            Defaults to 2000.
        smoothing_grid (Sequence[float], optional): Smoothing parameters to sweep.
            Defaults to ``(2/20, 2/50, 2/100, 2/250, 2/500)``.
        n_series (int, optional): Number of Monte Carlo series per sweep point.
            Defaults to 100.
        burn_in (int, optional): Number of initial observations excluded from
            coverage summaries. Defaults to 500.
        alpha (float, optional): Nominal miscoverage level. Defaults to 0.05.
        base_exp_kwargs (Optional[Dict[str, Any]], optional): Base experiment
            keyword arguments shared across all runs. Defaults to None.
        exp_kwargs_overrides (Optional[Sequence[Dict[str, Any]]], optional):
            Experiment override dictionaries passed to the study. Defaults to None.
        progress (bool, optional): Whether to display progress output. Defaults to
            True.
        parallel (bool, optional): Whether to run the study in parallel. Defaults to
            True.
        n_jobs (int, optional): Number of parallel workers. Defaults to -1.
        var_warmup (int, optional): Number of observations used for variance warmup.
            Defaults to 0.
        transform (str, optional): Transformation used by the bootstrap experiment.
            Defaults to "student".
        transform_power (float, optional): Power used by the variance transform.
            Defaults to 1.0 / 3.0.
        rho_power (float, optional): Power used for the rho calibration. Defaults to
            -1.0 / 3.0.
        run_sweeps_kwargs (Optional[Dict[str, Any]], optional): Extra keyword
            arguments forwarded to ``study_cls.run_sweeps``. Defaults to None.
        study_cls (Optional[type], optional): Study class exposing ``run_sweeps``.
            Defaults to ``UniformCoverageStudy``.

    Returns:
        Dict[str, Any]: Mapping from process label to the corresponding sweep result.

    Raises:
        ValueError: If ``labels`` and ``processes`` have different lengths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if study_cls is None:
        study_cls = UniformCoverageStudy

    labels = (
        list(labels) if labels is not None else [type(p).__name__ for p in processes]
    )
    if len(labels) != len(processes):
        raise ValueError('labels length must match processes length')

    run_sweeps_kwargs = {} if run_sweeps_kwargs is None else dict(run_sweeps_kwargs)
    base_exp_kwargs = {} if base_exp_kwargs is None else dict(base_exp_kwargs)
    exp_kwargs_overrides = exp_kwargs_overrides or [{}]

    ss = np.random.SeedSequence(seed) if seed is not None else None
    child_seqs = ss.spawn(len(processes)) if ss is not None else [None] * len(processes)

    results = {}
    for i, (proc, label) in enumerate(zip(processes, labels, strict=False)):
        proc_outdir = outdir / label
        proc_outdir.mkdir(parents=True, exist_ok=True)
        this_seed = None if child_seqs[i] is None else int(child_seqs[i].entropy)

        print(
            f'[run_sweeps_for_process_list] Running sweeps for {label} -> {proc_outdir}'
        )

        res = study_cls.run_sweeps(
            base_process_template=proc,
            sample_size=sample_size,
            dgp_overrides=None,
            exp_kwargs_overrides=exp_kwargs_overrides,
            smoothing_grid=smoothing_grid,
            outdir=proc_outdir,
            n_series=n_series,
            burn_in=burn_in,
            alpha=alpha,
            base_exp_kwargs=base_exp_kwargs,
            seed=this_seed,
            progress=progress,
            parallel=parallel,
            n_jobs=n_jobs,
            transform=transform,
            transform_power=transform_power,
            rho_power=rho_power,
            var_warmup=var_warmup,
            **run_sweeps_kwargs,
        )
        results[label] = res

    return results


def _safe_token(x: Any) -> str:
    """Convert a value to a filename-safe token.

    Args:
        x (Any): Value to encode.

    Returns:
        str: Token with decimal points, minus signs, and spaces normalized.
    """
    s = f'{x:g}' if isinstance(x, float) else str(x)
    return s.replace('.', 'p').replace('-', 'm').replace(' ', '')


def run_experiment_ablation(
    *,
    base_process_template,
    param_name: str,
    param_values: Sequence[Any],
    outdir: Union[str, Path],
    sample_size: int,
    smoothing_grid: Sequence[float],
    n_series: int,
    burn_in: int,
    alpha: float,
    var_warmup: int,
    base_exp_kwargs: Optional[Dict[str, Any]] = None,
    transform: str = 'student',
    transform_power: float = 1.0 / 3.0,
    rho_power: float = -1.0 / 3.0,
    seed_master: int = 1234,
    seed_per_value: bool = True,
    parallel: bool = True,
    n_jobs: int = -1,
    progress: bool = True,
    verbose: int = 10,
    save: bool = True,
    per_value_subfolder: bool = True,  # <--- NEW
) -> Dict[Any, Dict[str, Any]]:
    """Run a one-parameter ablation over a sequence of values.

    Args:
        base_process_template (Any): Process template used for every ablation run.
        param_name (str): Experiment keyword to vary.
        param_values (Sequence[Any]): Values assigned to ``param_name``.
        outdir (Union[str, Path]): Directory where ablation outputs are written.
        sample_size (int): Number of observations per simulated series.
        smoothing_grid (Sequence[float]): Smoothing parameters to sweep.
        n_series (int): Number of Monte Carlo series per sweep point.
        burn_in (int): Number of initial observations excluded from summaries.
        alpha (float): Nominal miscoverage level.
        var_warmup (int): Number of observations used for variance warmup.
        base_exp_kwargs (Optional[Dict[str, Any]], optional): Base experiment
            keyword arguments shared across ablation values. Defaults to None.
        transform (str, optional): Transformation used by the bootstrap experiment.
            Defaults to "student".
        transform_power (float, optional): Power used by the variance transform.
            Defaults to 1.0 / 3.0.
        rho_power (float, optional): Power used for the rho calibration. Defaults to
            -1.0 / 3.0.
        seed_master (int, optional): Base random seed. Defaults to 1234.
        seed_per_value (bool, optional): Whether to offset the seed for each value.
            Defaults to True.
        parallel (bool, optional): Whether to run sweeps in parallel. Defaults to
            True.
        n_jobs (int, optional): Number of parallel workers. Defaults to -1.
        progress (bool, optional): Whether to display progress output. Defaults to
            True.
        verbose (int, optional): Verbosity passed to the study runner. Defaults to
            10.
        save (bool, optional): Whether to write result files. Defaults to True.
        per_value_subfolder (bool, optional): Whether to store each value in its own
            subdirectory. Defaults to True.

    Returns:
        Dict[Any, Dict[str, Any]]: Mapping from ablation value to result payload.
    """
    outdir = Path(outdir)
    if save:
        outdir.mkdir(parents=True, exist_ok=True)

    base_exp_kwargs = {} if base_exp_kwargs is None else dict(base_exp_kwargs)

    results_by_value: Dict[Any, Dict[str, Any]] = {}

    for idx, val in enumerate(param_values):
        seed = seed_master + idx if seed_per_value else seed_master

        run_outdir = outdir
        if per_value_subfolder:
            run_outdir = outdir / f'{param_name}_{_safe_token(val)}'
            if save:
                run_outdir.mkdir(parents=True, exist_ok=True)

        exp_kwargs_overrides = [{param_name: val}]

        res_map = UniformCoverageStudy.run_sweeps(
            base_process_template=base_process_template,
            sample_size=sample_size,
            dgp_overrides=[{}],
            exp_kwargs_overrides=exp_kwargs_overrides,
            smoothing_grid=smoothing_grid,
            outdir=run_outdir,
            n_series=n_series,
            burn_in=burn_in,
            alpha=alpha,
            base_exp_kwargs=base_exp_kwargs,
            seed=seed,
            progress=progress,
            parallel=parallel,
            n_jobs=n_jobs,
            verbose=verbose,
            transform=transform,
            transform_power=transform_power,
            rho_power=rho_power,
            var_warmup=var_warmup,
            save=save,
        )

        ((name, payload),) = res_map.items()
        results_by_value[val] = payload
        print(f"[done] {param_name}={val} -> {payload.get('path')}")

    return results_by_value


def load_panels_from_paths(panels_paths: Dict[str, dict], *, loader) -> Dict[str, dict]:
    """Load multiple panels of sweeps from given file paths.

    Args:
        panels_paths (Dict[str, dict]): Mapping from panel title to a dictionary with
            ``paths`` and ``labels`` entries.
        loader (Callable[[str], Any]): Function used to load each path.

    Returns:
        Dict[str, dict]: Mapping from panel title to loaded sweeps and labels.

    Raises:
        ValueError: If any panel entry is missing ``paths`` or ``labels``.
    """
    panels_sweeps: Dict[str, dict] = {}
    for title, entry in panels_paths.items():
        if 'paths' not in entry or 'labels' not in entry:
            raise ValueError(f"Panel '{title}' must have 'paths' and 'labels'.")
        sweeps = [loader(str(p)) for p in entry['paths']]
        panels_sweeps[title] = {'sweeps': sweeps, 'labels': list(entry['labels'])}
    return panels_sweeps


def _to_token(x: str) -> str:
    """Convert a decimal string to the token format used in result filenames.

    Args:
        x (str): String or value to encode.

    Returns:
        str: Filename token with decimal points replaced by ``p`` when needed.
    """
    x = str(x)
    return x if 'p' in x else x.replace('.', 'p')


def build_panels_paths_generic(
    *,
    root: Path,
    phis: Dict[str, str],
    methods: Iterable[Tuple[str, str]] = (('ARmmult', 'student'),),
    method_display: Optional[Dict[str, str]] = None,
    panel_specs: Dict[str, Dict[str, str]],
    smooth: str = 'ewma',
    alpha: str = '0.1',
    bi: int = 500,
    vw: int = 400,
    n: int = 3500,
    nseries: int = 150,
    B: int = 200,
    nosmooth_var: bool = True,
    sweep_prefix: bool = False,
    include_method_token: bool = True,
    panel_dirname_fn=None,
    phi_dirname_fn=None,
    check_exists: bool = True,
) -> Dict[str, dict]:
    """Build coverage panel path mappings from filename conventions.

    Args:
        root (Path): Root directory containing panel subdirectories.
        phis (Dict[str, str]): Mapping from phi value strings to display labels.
        methods (Iterable[Tuple[str, str]], optional): Method key and transform
            distribution pairs included in each panel. Defaults to
            ``(("ARmmult", "student"),)``.
        method_display (Optional[Dict[str, str]], optional): Display labels for
            method keys. Defaults to None.
        panel_specs (Dict[str, Dict[str, str]]): Panel metadata containing shock
            tokens and optional filename tails.
        smooth (str, optional): Smoothing method token. Defaults to "ewma".
        alpha (str, optional): Alpha token or decimal string. Defaults to "0.1".
        bi (int, optional): Burn-in token expected in filenames. Defaults to 500.
        vw (int, optional): Variance warmup token expected in filenames. Defaults to
            400.
        n (int, optional): Sample size token expected in filenames. Defaults to 3500.
        nseries (int, optional): Series-count token expected in filenames. Defaults
            to 150.
        B (int, optional): Bootstrap replicate token expected in filenames. Defaults
            to 200.
        nosmooth_var (bool, optional): Whether filenames include the no-smoothing
            variance token. Defaults to True.
        sweep_prefix (bool, optional): Whether filenames include a ``sweep_`` prefix.
            Defaults to False.
        include_method_token (bool, optional): Whether filenames include a method
            token. Defaults to True.
        panel_dirname_fn (Optional[Callable[[str], str]], optional): Function mapping
            panel titles to directory names. Defaults to identity.
        phi_dirname_fn (Optional[Callable[[str], str]], optional): Function mapping
            phi values to directory names. Defaults to ``phi=<value>``.
        check_exists (bool, optional): Whether to validate all matched paths.
            Defaults to True.

    Returns:
        Dict[str, dict]: Mapping from panel title to matched paths and labels.

    Raises:
        FileNotFoundError: If a required directory or result file is missing.
    """
    root = Path(root)
    method_display = {} if method_display is None else dict(method_display)

    if panel_dirname_fn is None:
        panel_dirname_fn = lambda panel_title: panel_title
    if phi_dirname_fn is None:
        phi_dirname_fn = lambda phi_str: f'phi={phi_str}'

    alpha_tok = _to_token(alpha)

    def _match_one(dirpath: Path, pattern: str) -> Path:
        """Return the newest file in a directory matching a shell-style pattern.

        Args:
            dirpath (Path): Directory to search.
            pattern (str): Filename pattern interpreted by ``fnmatch``.

        Returns:
            Path: Newest matching file.

        Raises:
            FileNotFoundError: If the directory is missing or no files match.
        """
        if not dirpath.exists():
            raise FileNotFoundError(f'Directory does not exist: {dirpath}')

        matches = [
            p
            for p in dirpath.iterdir()
            if p.is_file() and fnmatch.fnmatch(p.name, pattern)
        ]
        if not matches:
            avail = [p.name for p in sorted(dirpath.iterdir()) if p.is_file()]
            preview = '\n'.join(avail[:10])
            raise FileNotFoundError(
                f'No files matched pattern:\n  dir={dirpath}\n  pat={pattern}\n'
                f'Available files (first 10):\n{preview}'
            )

        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest wins
        return matches[0]

    panels_paths: Dict[str, dict] = {}

    for panel_title, spec in panel_specs.items():
        shock = spec['shock']
        tail = spec.get('tail', '')

        paths: List[Path] = []
        labels: List[str] = []

        for phi_str, phi_label in phis.items():
            phi_tok = _to_token(phi_str)

            for method_key, tr_dist in methods:
                panel_dir = (
                    root / panel_dirname_fn(panel_title) / phi_dirname_fn(phi_str)
                )

                prefix = (
                    'sweep_' if sweep_prefix else ''
                ) + f'proc-AR1Process__phi-{phi_tok}'
                mid = '__nosmooth-var' if nosmooth_var else ''

                method_part = f'__method-{method_key}' if include_method_token else ''
                filename_pattern = (
                    f'{prefix}'
                    f'{shock}'
                    f'__smooth-{smooth}__tr-{tr_dist}'
                    f'__bi-{bi}__vw-{vw}__alpha-{alpha_tok}'
                    f'__n-{n}__nseries-{nseries}__B-{B}'
                    f'{mid}'
                    f'{method_part}'
                    f'*{tail}*.npz'
                )

                p = _match_one(panel_dir, filename_pattern)
                paths.append(p)

                short_method = method_display.get(method_key, method_key)
                labels.append(f'{phi_label} {short_method}'.strip())

        panels_paths[panel_title] = {'paths': paths, 'labels': labels}

    if check_exists:
        missing = [
            p
            for entry in panels_paths.values()
            for p in entry['paths']
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f'Missing {len(missing)} files (first: {missing[0]})'
            )

    return panels_paths


def build_power_panels_paths_generic(
    *,
    root: Path,
    panel_specs: Dict[str, Dict[str, str]],
    phis: Dict[str, str],
    trend_slopes: Dict[str, str],
    methods: Iterable[Tuple[str, str]] = (('ARmmult', 'student'),),
    method_display: Optional[Dict[str, str]] = None,
    smooth: str = 'ewma',
    alpha: str = '0.1',
    bi: int = 500,
    vw: int = 400,
    n: int = 3500,
    nseries: int = 100,
    B: int = 200,
    nosmooth_var: bool = True,
    seaA: Optional[str] = None,
    seaP: Optional[str] = None,
    check_exists: bool = True,
) -> Dict[str, dict]:
    """Build power panel path mappings from filename conventions.

    Args:
        root (Path): Directory containing power run result files.
        panel_specs (Dict[str, Dict[str, str]]): Panel metadata containing eta
            values.
        phis (Dict[str, str]): Mapping from phi value strings to display labels.
        trend_slopes (Dict[str, str]): Mapping from trend slope strings to display
            labels.
        methods (Iterable[Tuple[str, str]], optional): Method key and transform
            distribution pairs included in each panel. Defaults to
            ``(("ARmmult", "student"),)``.
        method_display (Optional[Dict[str, str]], optional): Display labels for
            method keys. Defaults to None.
        smooth (str, optional): Smoothing method token. Defaults to "ewma".
        alpha (str, optional): Alpha token or decimal string. Defaults to "0.1".
        bi (int, optional): Burn-in token expected in filenames. Defaults to 500.
        vw (int, optional): Variance warmup token expected in filenames. Defaults to
            400.
        n (int, optional): Sample size token expected in filenames. Defaults to 3500.
        nseries (int, optional): Series-count token expected in filenames. Defaults
            to 100.
        B (int, optional): Bootstrap replicate token expected in filenames. Defaults
            to 200.
        nosmooth_var (bool, optional): Whether filenames include the no-smoothing
            variance token. Defaults to True.
        seaA (Optional[str], optional): Seasonal amplitude token appended to matches.
            Defaults to None.
        seaP (Optional[str], optional): Seasonal period token appended to matches.
            Defaults to None.
        check_exists (bool, optional): Whether to validate all matched paths.
            Defaults to True.

    Returns:
        Dict[str, dict]: Mapping from panel title to matched paths and labels.

    Raises:
        ValueError: If a panel specification omits ``eta``.
        FileNotFoundError: If a required directory or result file is missing.
    """
    root = Path(root)
    method_display = {} if method_display is None else dict(method_display)

    alpha_tok = _to_token(alpha)

    def _match_one(dirpath: Path, pattern: str) -> Path:
        """Return the newest file in a directory matching a shell-style pattern.

        Args:
            dirpath (Path): Directory to search.
            pattern (str): Filename pattern interpreted by ``fnmatch``.

        Returns:
            Path: Newest matching file.

        Raises:
            FileNotFoundError: If the directory is missing or no files match.
        """
        if not dirpath.exists():
            raise FileNotFoundError(f'Directory does not exist: {dirpath}')

        matches = [
            p
            for p in dirpath.iterdir()
            if p.is_file() and fnmatch.fnmatch(p.name, pattern)
        ]
        if not matches:
            avail = [p.name for p in sorted(dirpath.iterdir()) if p.is_file()]
            preview = '\n'.join(avail[:12])
            raise FileNotFoundError(
                f'No files matched pattern:\n  dir={dirpath}\n  pat={pattern}\n'
                f'Available files (first 12):\n{preview}'
            )
        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # newest wins
        return matches[0]

    panels_paths: Dict[str, dict] = {}

    mid = '__nosmooth-var' if nosmooth_var else ''
    sea_tail = ''
    if seaA is not None:
        sea_tail += f'__seaA-{_to_token(seaA)}'
    if seaP is not None:
        sea_tail += f'__seaP-{_to_token(seaP)}'

    for panel_title, spec in panel_specs.items():
        eta_val = spec.get('eta', None)
        if eta_val is None:
            raise ValueError(f"panel_specs['{panel_title}'] must include key 'eta'")

        eta_tok = _to_token(eta_val)

        paths: List[Path] = []
        labels: List[str] = []

        for phi_str, phi_label in phis.items():
            phi_tok = _to_token(phi_str)

            for trend_str, trend_label in trend_slopes.items():
                trend_tok = _to_token(trend_str)

                for method_key, tr_dist in methods:
                    filename_pattern = (
                        f'proc-AR1Process__phi-{phi_tok}'
                        f'__trend-{trend_tok}'
                        f'__smooth-{smooth}'
                        f'__eta-{eta_tok}'
                        f'__tr-{tr_dist}'
                        f'__bi-{bi}__vw-{vw}__alpha-{alpha_tok}'
                        f'__n-{n}__nseries-{nseries}__B-{B}'
                        f'{mid}'
                        f'__method-{method_key}'
                        f'*{sea_tail}*.npz'
                    )

                    p = _match_one(root, filename_pattern)
                    paths.append(p)

                    short_method = method_display.get(method_key, method_key)
                    labels.append(f'{phi_label} {trend_label} {short_method}'.strip())

        panels_paths[panel_title] = {'paths': paths, 'labels': labels}

    if check_exists:
        missing = [
            p
            for entry in panels_paths.values()
            for p in entry['paths']
            if not p.exists()
        ]
        if missing:
            raise FileNotFoundError(
                f'Missing {len(missing)} files (first: {missing[0]})'
            )

    return panels_paths


def load_power_run_npz(path: str) -> dict:
    """Load a saved power run into plain NumPy arrays and scalars.

    Args:
        path (str): Path to a ``.npz`` power run file.

    Returns:
        dict: Dictionary containing times, power curve, alpha, and effective burn-in.
    """
    d = np.load(path, allow_pickle=False)
    out = {
        'times': np.asarray(d['times'], dtype=int),
        'power_curve': np.asarray(d['power_curve'], dtype=float),
        'alpha': float(d['alpha']) if 'alpha' in d else float('nan'),
    }
    if 'burn_eff' in d:
        out['burn_eff'] = int(np.asarray(d['burn_eff']).item())
    else:
        bi = int(np.asarray(d['burn_in']).item()) if 'burn_in' in d else 0
        vw = int(np.asarray(d['var_warmup']).item()) if 'var_warmup' in d else 0
        out['burn_eff'] = bi + vw

    T = min(out['times'].size, out['power_curve'].size)
    out['times'] = out['times'][:T]
    out['power_curve'] = out['power_curve'][:T]
    return out


def build_ablation_panels_paths(
    *,
    base: Path,
    phi: str = '0.5',
    smooth: str = 'ewma',
    alpha: str = '0.1',
    bi: int = 500,
    vw_in_fname: int = 400,
    n: int = 3500,
    nseries: int = 150,
    B_in_fname: int = 200,
    nosmooth_var: bool = True,
    panels: Optional[Dict[str, dict]] = None,
    check_exists: bool = True,
) -> Dict[str, dict]:
    """Build ablation panel path mappings from filename conventions.

    Args:
        base (Path): Base directory containing ablation subdirectories.
        phi (str, optional): Phi token or decimal string expected in filenames.
            Defaults to "0.5".
        smooth (str, optional): Smoothing method token. Defaults to "ewma".
        alpha (str, optional): Alpha token or decimal string. Defaults to "0.1".
        bi (int, optional): Burn-in token expected in filenames. Defaults to 500.
        vw_in_fname (int, optional): Variance warmup token expected in filenames.
            Defaults to 400.
        n (int, optional): Sample size token expected in filenames. Defaults to 3500.
        nseries (int, optional): Series-count token expected in filenames. Defaults
            to 150.
        B_in_fname (int, optional): Bootstrap replicate token expected in filenames.
            Defaults to 200.
        nosmooth_var (bool, optional): Whether filenames include the no-smoothing
            variance token. Defaults to True.
        panels (Optional[Dict[str, dict]], optional): Panel configuration mapping.
            Defaults to a standard ablation layout.
        check_exists (bool, optional): Whether to validate all matched paths.
            Defaults to True.

    Returns:
        Dict[str, dict]: Mapping from panel title to matched paths and labels.

    Raises:
        ValueError: If an ablation panel has mismatched values and labels.
        FileNotFoundError: If a required directory or result file is missing.
    """
    base = Path(base)
    alpha_tok = _to_token(alpha)
    phi_tok = _to_token(phi)

    def _match_one(dirpath: Path, filename_pattern: str) -> Path:
        """Return the newest file matching an ablation filename pattern.

        Args:
            dirpath (Path): Directory to search.
            filename_pattern (str): Filename pattern interpreted by ``fnmatch``.

        Returns:
            Path: Newest matching file.

        Raises:
            FileNotFoundError: If the directory is missing or no files match.
        """
        if not dirpath.exists():
            raise FileNotFoundError(f'Directory does not exist: {dirpath}')

        matches = [
            p
            for p in dirpath.iterdir()
            if p.is_file() and fnmatch.fnmatch(p.name, filename_pattern)
        ]
        if not matches:
            msg = (
                f'No files matched pattern:\n  dir={dirpath}\n  pat={filename_pattern}\n'
                f'  available (first 10):\n    '
                + '\n    '.join(
                    [p.name for p in sorted(dirpath.iterdir()) if p.is_file()][:10]
                )
            )
            raise FileNotFoundError(msg)

        matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0]

    if panels is None:
        panels = {
            'AR(1) with varying $B$': {
                'ablation': 'AR1_phi0p5_ablate_B',
                'folder_prefix': 'B',
                'values': ['20', '50', '100', '200'],
                'labels': [r'$B=20$', r'$B=50$', r'$B=100$', r'$B=200$'],
                'B_from_value': True,
                'extra_tail': '',
            },
            'AR(1) with varying calibration': {
                'ablation': 'AR1_phi0p5_ablate_vw',
                'folder_prefix': 'var_warmup',
                'values': ['20', '100', '200', '400'],
                'labels': [
                    r'$t_1-t_0=20$',
                    r'$t_1-t_0=100$',
                    r'$t_1-t_0=200$',
                    r'$t_1-t_0=400$',
                ],
                'B_from_value': False,
                'extra_tail': '',
            },
            r'AR(1) with varying powers for $\nu$': {
                'ablation': 'AR1_phi0p5_ablate_tp',
                'folder_prefix': 'transform_power',
                'values': ['0p25', '0p333333', '0p5', '1'],
                'labels': [r'$\nu^{1/4}$', r'$\nu^{1/3}$', r'$\nu^{1/2}$', r'$\nu$'],
                'B_from_value': False,
                'extra_tail': '',
            },
            r'AR(1) with varying $\chi$': {
                'ablation': 'AR1_phi0p5_ablate_rho',
                'folder_prefix': 'rho_power',
                'values': ['m0p5', 'm0p333333', 'm0p25', '0'],
                'labels': [
                    r'$\chi={1/2}$',
                    r'$\chi={1/3}$',
                    r'$\chi={1/4}$',
                    r'$\chi={0}$',
                ],
                'B_from_value': False,
                'extra_tail': '',
            },
        }

    panels_paths: Dict[str, dict] = {}

    for panel_title, cfg in panels.items():
        ablation = cfg['ablation']
        folder_prefix = cfg['folder_prefix']
        values = list(cfg['values'])
        labels = list(cfg['labels'])
        B_from_value = bool(cfg.get('B_from_value', False))

        if len(values) != len(labels):
            raise ValueError(f'{panel_title}: values and labels must have same length.')

        paths: List[Path] = []

        for v in values:
            run_dir = base / ablation / f'{folder_prefix}_{v}'

            B_tok = _to_token(v) if B_from_value else _to_token(str(B_in_fname))

            prefix = f'proc-AR1Process__phi-{phi_tok}__shock-type-none_p-0_scale-1'
            core = (
                f'{prefix}'
                f'__smooth-{smooth}__tr-student'
                f'__bi-{bi}__vw-{vw_in_fname}__alpha-{alpha_tok}'
                f'__n-{n}__nseries-{nseries}__B-{B_tok}'
            )

            mid = '__nosmooth-var' if nosmooth_var else '*'
            filename_pattern = f'{core}{mid}*.npz'
            p = _match_one(run_dir, filename_pattern)
            paths.append(p)

        if check_exists:
            missing = [p for p in paths if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f'{panel_title}: missing {len(missing)} files (first: {missing[0]})'
                )

        panels_paths[panel_title] = {'paths': paths, 'labels': labels}

    return panels_paths
