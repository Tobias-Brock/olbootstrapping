from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from olbootstrap.experiments._base_experiment import ExperimentResults


class BasePlotter:
    """Stateless plotter that applies a consistent plotting style.

    Subclasses may call super().__init__(...) and optionally pass an
    `rc_overrides` dict to tweak default rc parameters.
    """

    DEFAULT_RCPARAMS: Dict[str, Any] = {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.titlecolor': 'black',
        'font.size': 12,
        'xtick.color': 'black',
        'ytick.color': 'black',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'grid.color': 'lightgray',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
    }

    def __init__(
        self,
        style: str = 'science',
        figsize: Tuple[float, float] = (12, 6),
        dpi: int = 150,
        rc_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the base plotter style state.

        Args:
            style (str, optional): Matplotlib style name or "science".
                Defaults to "science".
            figsize (Tuple[float, float], optional): Default figure size.
                Defaults to (12, 6).
            dpi (int, optional): Default figure DPI. Defaults to 150.
            rc_overrides (Optional[Dict[str, Any]], optional): Runtime rcParam
                overrides merged after defaults. Defaults to None.
        """
        self.style = style
        self.figsize = figsize
        self.dpi = int(dpi)
        self._rc_overrides = dict(rc_overrides or {})
        self._apply_style()

    def _apply_style(self) -> None:
        """Apply plotting style and merged rcParams.

        Returns:
            None
        """
        try:
            if isinstance(self.style, str) and self.style.lower() == 'science':
                plt.style.use(['science', 'no-latex'])
            else:
                plt.style.use(self.style)
        except Exception:
            plt.style.use('classic')

        rc = dict(self.DEFAULT_RCPARAMS)
        rc.update(
            {
                'figure.figsize': self.figsize,
                'figure.dpi': self.dpi,
            }
        )
        rc.update(self._rc_overrides)
        plt.rcParams.update(rc)

    def update_style(self, *, rc_overrides: Optional[Dict[str, Any]] = None) -> None:
        """Update runtime rc overrides and re-apply style.

        Args:
            rc_overrides (Optional[Dict[str, Any]], optional): Additional rcParam
                overrides to merge into the current overrides.

        Returns:
            None
        """
        if rc_overrides:
            self._rc_overrides.update(rc_overrides)
        self._apply_style()

    def _save_figure(
        self,
        fig,
        save_path: Optional[Union[str, Path]],
        *,
        dpi: Optional[int] = None,
        save_kwargs: Optional[dict] = None,
        default_suffix: str = '.png',
    ) -> Optional[str]:
        """Save a figure once, consistently, and return the saved path (str).

        Args:
            fig: Matplotlib Figure.
            save_path: Target path (str/Path) or None to skip saving.
            dpi: Optional DPI override if not already in save_kwargs.
            save_kwargs: Extra kwargs for `fig.savefig`.
            default_suffix: Used if `save_path` has no extension.

        Returns:
            The saved path as a string, or None if `save_path` is None.
        """
        if save_path is None:
            return None

        p = Path(save_path)
        if not p.suffix:
            p = p.with_suffix(default_suffix)

        if p.parent:
            p.parent.mkdir(parents=True, exist_ok=True)

        opts = {} if save_kwargs is None else dict(save_kwargs)
        if ('dpi' not in opts) and (dpi is not None):
            opts['dpi'] = dpi
        opts.setdefault('bbox_inches', 'tight')
        fig.savefig(p, **opts)
        return str(p)

    def _plot_bootstrap_on_ax(
        self,
        ax: plt.Axes,
        res: 'ExperimentResults',
        *,
        plot_series: bool = False,
        alpha: float = 0.02,
        center_label: str = 'point estimate',
        legend_fontsize: int = 16,
        tick_labelsize: int = 16,
        axis_labelsize: int = 18,
        title: Optional[str] = None,
        title_size: int = 18,
        marker_size: int = 6,
        line_width: float = 2.5,
        show_pointwise_band: bool = True,
        show_uniform_band: bool = True,
        show_mu: bool = True,
    ) -> plt.Axes:
        """Draw a single bootstrap panel on an existing axis.

        Args:
            ax (plt.Axes): Axis on which to draw the panel.
            res (ExperimentResults): Experiment result container.
            plot_series (bool, optional): If True, show observed samples.
            alpha (float, optional): Nominal significance level.
            center_label (str, optional): Label for the bootstrap center.
            legend_fontsize (int, optional): Legend font size.
            tick_labelsize (int, optional): Tick label font size.
            axis_labelsize (int, optional): Axis label font size.
            title (Optional[str], optional): Optional axis title.
            title_size (int, optional): Title font size.
            marker_size (int, optional): Marker size for line handles.
            line_width (float, optional): Line width for plotted series.
            show_pointwise_band (bool, optional): If True, draw pointwise bands.
            show_uniform_band (bool, optional): If True, draw uniform bands.
            show_mu (bool, optional): If True, draw the true mean when available.

        Returns:
            plt.Axes: The axis that was drawn on.

        Raises:
            RuntimeError: If `res.times` is missing.
        """
        if res.times is None:
            raise RuntimeError('res.times required')

        times = np.asarray(res.times, dtype=int)

        if plot_series and (res.samples is not None):
            ax.plot(
                times,
                res.samples[times - 1],
                linestyle='None',
                marker='o',
                markersize=1.9,
                alpha=0.43,
                color='0.10',
                markeredgewidth=0.35,
                markeredgecolor='white',
                zorder=4,
                rasterized=True,
            )
        if res.bootstrap_means is not None:
            ax.plot(
                times,
                res.bootstrap_means,
                linestyle='--',
                lw=line_width,
                label=center_label,
                zorder=5,
                markersize=marker_size,
                color='red',
            )

        if (
            show_pointwise_band
            and (res.lower_bounds is not None)
            and (res.upper_bounds is not None)
        ):
            pc = ax.fill_between(
                times,
                res.lower_bounds,
                res.upper_bounds,
                alpha=0.18,
                label=f'Bootstrap {(100*(1-alpha)):.1f}% pointwise band',
                zorder=1,
            )
            pc.set_edgecolor('none')
            pc.set_linewidth(0.0)

        if (
            show_uniform_band
            and (res.uniform_lower is not None)
            and (res.uniform_upper is not None)
            and (res.uniform_record_times is not None)
        ):
            urt = np.asarray(res.uniform_record_times, dtype=int)
            uc = ax.fill_between(
                urt,
                res.uniform_lower,
                res.uniform_upper,
                alpha=0.40,
                label=f'Bootstrap {(100*(1-alpha)):.1f}% uniform band',
                zorder=2,
            )
            uc.set_edgecolor('black')  # or a darker version of the facecolor
            uc.set_linewidth(0.8)

        if show_mu and (res.mu_t is not None):
            ax.plot(
                times,
                res.mu_t[times - 1],
                linestyle='-',
                linewidth=line_width,
                label=r'$\mu(t)$',
                zorder=6,
                markersize=marker_size,
            )

        if res.smoother_target is not None:
            st = np.asarray(res.smoother_target, dtype=float)
            if st.size >= times.max():
                ax.plot(
                    times,
                    st[times - 1],
                    linewidth=line_width,
                    label=r'$\mu_\eta(t)$',
                    zorder=7,
                    markersize=marker_size,
                    color='orange',
                )

        ax.set_xlabel(r'$t$', fontsize=axis_labelsize)
        ax.set_ylabel('Level', fontsize=axis_labelsize)
        if title:
            ax.set_title(title, fontsize=title_size)

        ax.set_xlim(float(times.min()), float(times.max()))
        ax.margins(x=0)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        ax.tick_params(axis='both', which='minor', labelsize=max(tick_labelsize - 1, 8))

        return ax

    def plot_sweep(
        self,
        sweeps: Sequence,
        *,
        nrows: int = 1,
        ncols: int = 1,
        titles: Optional[Sequence[str]] = None,
        show_target: bool = True,
        figsize: Optional[tuple] = None,
        dpi: Optional[int] = None,
        save_path: Optional[Union[str, Path]] = None,
        save_kwargs: Optional[dict] = None,
        show: bool = True,
        legend_fontsize: int = 14,
        tick_labelsize: int = 14,
        axis_labelsize: int = 18,
        title_size: int = 18,
        marker_size: int = 6,
        line_width: float = 1.5,
        reduce_whitespace: bool = True,
        tight_pad: float = 0.6,
        hspace: float = 0.15,
        wspace: float = 0.12,
    ):
        """Plot one or many SweepResults on an nrows x ncols grid.

        Args:
            sweeps (Sequence): SweepResults objects or dict-like equivalents.
            nrows (int, optional): Number of subplot rows.
            ncols (int, optional): Number of subplot columns.
            titles (Optional[Sequence[str]], optional): Optional subplot titles.
            show_target (bool, optional): If True, draw the nominal coverage line.
            figsize (Optional[tuple], optional): Figure size override.
            dpi (Optional[int], optional): Figure DPI override.
            save_path (Optional[Union[str, Path]], optional): Optional output path.
            save_kwargs (Optional[dict], optional): Extra `fig.savefig` kwargs.
            show (bool, optional): If True, call `plt.show()`.
            legend_fontsize (int, optional): Legend font size.
            tick_labelsize (int, optional): Tick label font size.
            axis_labelsize (int, optional): Axis label font size.
            title_size (int, optional): Title font size.
            marker_size (int, optional): Marker size.
            line_width (float, optional): Line width.
            reduce_whitespace (bool, optional): If True, use compact layout.
            tight_pad (float, optional): Padding passed to `tight_layout`.
            hspace (float, optional): Subplot vertical spacing.
            wspace (float, optional): Subplot horizontal spacing.

        Returns:
            tuple: `(fig, axes)` where axes has shape `(nrows, ncols)`.

        Raises:
            ValueError: If no sweeps are provided or the grid is too small.
        """
        K = len(sweeps)
        cells = nrows * ncols
        if K == 0:
            raise ValueError('No sweeps provided.')
        if K > cells:
            raise ValueError(f'{K} sweeps but only {cells} grid cells (nrows*ncols).')

        use_figsize = self.figsize if figsize is None else figsize
        use_dpi = self.dpi if dpi is None else dpi
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=use_figsize, dpi=use_dpi
        )
        axes = np.array(axes).reshape(nrows, ncols)
        flat_axes = axes.ravel()

        def _unpack(s):
            """Extract sorted sweep arrays from an object or mapping.

            Args:
                s: SweepResults-like object or mapping.

            Returns:
                tuple[np.ndarray, np.ndarray, np.ndarray, float]: Sorted eta,
                    average uniform-time coverage, uniform-over-series coverage,
                    and alpha.
            """
            if hasattr(s, 'eta'):
                eta = np.asarray(s.eta, dtype=float)
                y_u_t = np.asarray(s.avg_uniform_time_fraction, dtype=float)
                y_u_s = np.asarray(s.uniform_over_series_uniform, dtype=float)
                alpha = (
                    float(s.alpha) if getattr(s, 'alpha', None) is not None else np.nan
                )
            else:
                eta = np.asarray(s['eta'], dtype=float)
                y_u_t = np.asarray(s['avg_uniform_time_fraction'], dtype=float)
                y_u_s = np.asarray(s['uniform_over_series_uniform'], dtype=float)
                alpha = float(s.get('alpha', np.nan))
            idx = np.argsort(eta)
            return eta[idx], y_u_t[idx], y_u_s[idx], alpha

        for i, sweep in enumerate(sweeps):
            ax = flat_axes[i]
            eta, y_u_t, y_u_s, alpha = _unpack(sweep)

            ax.plot(
                eta,
                y_u_t,
                marker='s',
                markersize=marker_size,
                lw=line_width,
                label='Average Uniform Band Coverage',
            )
            ax.plot(
                eta,
                y_u_s,
                marker='v',
                markersize=marker_size,
                lw=line_width,
                label='Uniform over Time Coverage',
            )

            if show_target and np.isfinite(alpha):
                ax.axhline(
                    1.0 - alpha,
                    linestyle='--',
                    linewidth=1.0,
                    label=r'$1-\alpha$',
                )

            if titles is not None and i < len(titles) and titles[i] is not None:
                ax.set_title(str(titles[i]), fontsize=title_size)

            ax.set_xlabel(r'$\eta$', fontsize=axis_labelsize)
            ax.set_ylabel('Coverage', fontsize=axis_labelsize)

            ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
            ax.tick_params(
                axis='both', which='minor', labelsize=max(tick_labelsize - 1, 8)
            )

            ax.set_xlim(float(np.nanmin(eta)), float(np.nanmax(eta)))
            ax.set_ylim(0.0, 1.05)
            ax.margins(x=0)

            ax.legend(loc='best', fontsize=legend_fontsize, frameon=True)
            ax.grid(True, linestyle='--', alpha=0.5)

        for j in range(K, cells):
            flat_axes[j].set_visible(False)

        if reduce_whitespace:
            fig.tight_layout(pad=tight_pad)
            fig.subplots_adjust(hspace=hspace, wspace=wspace)
        else:
            fig.tight_layout()

        self._save_figure(
            fig, save_path, dpi=dpi, save_kwargs=save_kwargs, default_suffix='.png'
        )

        if show:
            plt.show()

        return fig, axes
