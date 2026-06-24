from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401

from olbootstrap.experiments._base_experiment import ExperimentResults

from ._baseplotting import BasePlotter


def _eta_to_ess(eta_arr: np.ndarray) -> np.ndarray:
    """Convert EWMA smoothing rates to effective sample sizes.

    Args:
        eta_arr (np.ndarray): Array-like EWMA smoothing rates.

    Returns:
        np.ndarray: Effective sample sizes computed as `2 / eta`.
    """
    eta_arr = np.asarray(eta_arr, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        nu = 2.0 / eta_arr
    return nu


def _canonical_method_key(method: Any) -> str:
    """Normalize method labels used for style lookups.

    Args:
        method (Any): Method token extracted from a label or provided by a caller.

    Returns:
        str: Canonical method key for known bootstrap methods, otherwise the
            stripped input string.
    """
    key = str(method).strip()
    lower = key.lower()
    aliases = {
        'ours': 'ours',
        'iid': 'iid',
        'ws': 'WS',
        'block': 'block',
    }
    return aliases.get(lower, key)


def _method_color_defaults(
    method_keys: Sequence[str],
    cycle: Sequence[str],
    method_colors: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a method-to-color map with canonical bootstrap defaults.

    Args:
        method_keys (Sequence[str]): Canonical method keys to style.
        cycle (Sequence[str]): Matplotlib fallback color cycle.
        method_colors (Optional[Dict[str, str]], optional): Caller-provided color
            overrides keyed by method name. Defaults to None.

    Returns:
        Dict[str, str]: Mapping from method key to color.
    """
    method2color = {m: cycle[i % len(cycle)] for i, m in enumerate(method_keys)}
    method2color.update(
        {
            'ours': '#0C5DA5',
            'iid': '#00B945',
            'WS': 'orange',
            'block': '#B07AA1',
        }
    )
    if method_colors is not None:
        method2color.update(
            {_canonical_method_key(k): v for k, v in method_colors.items()}
        )
    return method2color


def _group_color_defaults(
    group_keys: Sequence[str],
    cycle: Sequence[str],
    group_colors: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build a group-to-color map with the plotting palette used for groups.

    Args:
        group_keys (Sequence[str]): Group keys to style.
        cycle (Sequence[str]): Matplotlib fallback color cycle.
        group_colors (Optional[Dict[str, str]], optional): Caller-provided color
            overrides keyed by group name. Defaults to None.

    Returns:
        Dict[str, str]: Mapping from group key to color.
    """
    palette = ['#0C5DA5', '#00B945', 'orange', '#B07AA1']
    fallback = list(cycle) if cycle else ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    colors = palette + [c for c in fallback if c not in palette]
    group2color = {g: colors[i % len(colors)] for i, g in enumerate(group_keys)}
    if group_colors is not None:
        group2color.update({str(k).strip(): v for k, v in group_colors.items()})
    return group2color


class BootstrapPlotter(BasePlotter):
    """Plotter class for bootstrap experiments."""

    def __init__(
        self,
        style: str = 'science',
        figsize: Tuple[float, float] = (12, 6),
        dpi: int = 150,
        rc_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the bootstrap plotter.

        Args:
            style (str, optional): Matplotlib style name or "science".
            figsize (Tuple[float, float], optional): Default figure size.
            dpi (int, optional): Default figure DPI.
            rc_overrides (Optional[Dict[str, Any]], optional): Optional rcParam
                overrides.
        """
        super().__init__(
            style=style, figsize=figsize, dpi=dpi, rc_overrides=rc_overrides
        )

    def plot_bootstrap(
        self,
        res: 'ExperimentResults',
        plot_series: bool = False,
        alpha: float = 0.02,
        center_label: str = 'point estimate',
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        *,
        save_path: Optional[Union[str, Path]] = None,
        dpi: Optional[int] = None,
        figsize: Optional[tuple] = None,
        save_kwargs: Optional[dict] = None,
        show: bool = True,
        legend_fontsize: int = 16,
        tick_labelsize: int = 16,
        axis_labelsize: int = 18,
        title_size: int = 18,
        marker_size: int = 6,
        line_width: float = 2.5,
        reduce_whitespace: bool = True,
        tight_pad: float = 0.6,
        hspace: float = 0.15,
        wspace: float = 0.12,
        show_pointwise_band: bool = True,
        show_uniform_band: bool = True,
        show_mu: bool = True,
    ):
        """Plot a single bootstrap experiment result.

        Args:
            res (ExperimentResults): Experiment result to plot.
            plot_series (bool, optional): If True, show the observed series.
            alpha (float, optional): Nominal significance level.
            center_label (str, optional): Label for the bootstrap center.
            ax (Optional[plt.Axes], optional): Existing axis to plot on.
            title (Optional[str], optional): Optional plot title.
            save_path (Optional[Union[str, Path]], optional): Optional output path.
            dpi (Optional[int], optional): Figure DPI override.
            figsize (Optional[tuple], optional): Figure size override.
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
            show_pointwise_band (bool, optional): If True, draw pointwise bands.
            show_uniform_band (bool, optional): If True, draw uniform bands.
            show_mu (bool, optional): If True, draw the true mean when available.

        Returns:
            tuple: `(fig, ax, save_path)` where `save_path` is the requested path.

        Raises:
            RuntimeError: If `res.times` is missing.
        """
        if res.times is None:
            raise RuntimeError('res.times required')

        if ax is None:
            use_figsize = self.figsize if figsize is None else figsize
            use_dpi = self.dpi if dpi is None else dpi
            fig, ax = plt.subplots(figsize=use_figsize, dpi=use_dpi)
        else:
            fig = ax.figure
            if figsize is not None:
                fig.set_size_inches(figsize)
            if dpi is not None:
                fig.set_dpi(dpi)

        self._plot_bootstrap_on_ax(
            ax,
            res,
            plot_series=plot_series,
            alpha=alpha,
            center_label=center_label,
            legend_fontsize=legend_fontsize,
            tick_labelsize=tick_labelsize,
            axis_labelsize=axis_labelsize,
            title=title,
            title_size=title_size,
            marker_size=marker_size,
            line_width=line_width,
            show_pointwise_band=show_pointwise_band,
            show_uniform_band=show_uniform_band,
            show_mu=show_mu,
        )

        ax.legend(loc='upper left', fontsize=legend_fontsize)

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

        return fig, ax, save_path

    def plot_bootstrap_grid(
        self,
        results: Sequence['ExperimentResults'],
        *,
        titles: Optional[Sequence[Optional[str]]] = None,
        plot_series: bool = False,
        alpha: float = 0.02,
        center_label: str = 'Point estimate',
        nrows: Optional[int] = None,
        ncols: Optional[int] = None,
        layout: Optional[Tuple[int, int]] = None,  # alternative to nrows/ncols
        sharex: bool = True,
        sharey: bool = False,
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        suptitle: Optional[str] = None,
        common_legend: bool = True,
        legend_loc: str = 'upper center',
        legend_ncol: Optional[int] = None,
        legend_bbox: Optional[tuple] = None,
        save_path: Optional[Union[str, Path]] = None,
        save_kwargs: Optional[dict] = None,
        show: bool = True,
        legend_fontsize: int = 14,
        tick_labelsize: int = 12,
        axis_labelsize: int = 14,
        title_size: int = 14,
        marker_size: int = 5,
        line_width: float = 1.3,
        reduce_whitespace: bool = True,
        tight_pad: float = 0.6,
        hspace: float = 0.25,
        wspace: float = 0.25,
        bottom_pad: float = 0.12,
        align_axes: bool = True,
        ypad_frac: float = 0.05,
        ylabel: Optional[str] = None,
        ylabel_on_first: bool = True,
        show_pointwise_band: bool = True,
        show_uniform_band: bool = True,
        show_mu: bool = True,
    ) -> tuple:
        """Plot multiple bootstrap experiment results in a grid.

        Args:
            results (Sequence[ExperimentResults]): Experiment results to plot.
            titles (Optional[Sequence[Optional[str]]], optional): Optional panel
                titles.
            plot_series (bool, optional): If True, show observed samples.
            alpha (float, optional): Nominal significance level.
            center_label (str, optional): Label for bootstrap center lines.
            nrows (Optional[int], optional): Number of subplot rows.
            ncols (Optional[int], optional): Number of subplot columns.
            layout (Optional[Tuple[int, int]], optional): Explicit `(rows, cols)`.
            sharex (bool, optional): If True, share x-axes.
            sharey (bool, optional): If True, share y-axes.
            figsize (Optional[Tuple[float, float]], optional): Figure size override.
            dpi (Optional[int], optional): Figure DPI override.
            suptitle (Optional[str], optional): Optional figure-level title.
            common_legend (bool, optional): If True, draw one legend for all axes.
            legend_loc (str, optional): Figure legend location.
            legend_ncol (Optional[int], optional): Number of legend columns.
            legend_bbox (Optional[tuple], optional): Figure legend anchor.
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
            bottom_pad (float, optional): Bottom margin for lower legends.
            align_axes (bool, optional): If True, align x/y limits.
            ypad_frac (float, optional): Fractional y-axis padding.
            ylabel (Optional[str], optional): Override y-axis label.
            ylabel_on_first (bool, optional): If True, show shared y-label only on
                the first panel.
            show_pointwise_band (bool, optional): If True, draw pointwise bands.
            show_uniform_band (bool, optional): If True, draw uniform bands.
            show_mu (bool, optional): If True, draw the true mean when available.

        Returns:
            tuple: `(fig, axes_array, saved_path_or_None)`.

        Raises:
            ValueError: If `results` is empty.
        """
        R = len(results)
        if R == 0:
            raise ValueError('results must be a non-empty sequence')

        if layout is not None:
            r, c = layout
        else:
            if nrows is None and ncols is None:
                r, c = 1, R
            elif nrows is None:
                c = int(ncols)
                r = int(np.ceil(R / c))
            elif ncols is None:
                r = int(nrows)
                c = int(np.ceil(R / r))
            else:
                r, c = int(nrows), int(ncols)

        use_figsize = (
            figsize
            if figsize is not None
            else (
                self.figsize
                if (r, c) == (1, 1)
                else (self.figsize[0] * c / 1.2, self.figsize[1] * r / 1.2)
            )
        )
        use_dpi = self.dpi if dpi is None else dpi

        fig, axs = plt.subplots(
            nrows=r,
            ncols=c,
            figsize=use_figsize,
            dpi=use_dpi,
            sharex=sharex,
            sharey=sharey,
        )
        axs = np.atleast_1d(axs).ravel()

        # plot each panel
        handles_all: List = []
        labels_all: List = []
        for i, res in enumerate(results):
            ax = axs[i]
            ttl = None if titles is None else (titles[i] if i < len(titles) else None)
            self._plot_bootstrap_on_ax(
                ax,
                res,
                plot_series=plot_series,
                alpha=alpha,
                center_label=center_label,
                legend_fontsize=legend_fontsize,
                tick_labelsize=tick_labelsize,
                axis_labelsize=axis_labelsize,
                title=ttl,
                title_size=title_size,
                marker_size=marker_size,
                line_width=line_width,
                show_pointwise_band=show_pointwise_band,
                show_uniform_band=show_uniform_band,
                show_mu=show_mu,
            )
            ho, lo = ax.get_legend_handles_labels()
            if ho:
                handles_all.extend(ho)
                labels_all.extend(lo)

        for j in range(R, r * c):
            axs[j].axis('off')

        if align_axes:
            global_tmin, global_tmax = np.inf, -np.inf
            global_ymin, global_ymax = np.inf, -np.inf

            def _upd(arr):
                """Update global y-limits from an array-like object.

                Args:
                    arr: Array-like y-values to include in the global bounds.

                Returns:
                    None
                """
                nonlocal global_ymin, global_ymax
                a = np.asarray(arr, dtype=float)
                if a.size:
                    m, M = np.nanmin(a), np.nanmax(a)
                    if np.isfinite(m):
                        global_ymin = min(global_ymin, m)
                    if np.isfinite(M):
                        global_ymax = max(global_ymax, M)

            for res in results:
                t = (
                    np.asarray(res.times, dtype=int)
                    if res.times is not None
                    else np.array([])
                )
                if t.size:
                    global_tmin = min(global_tmin, int(np.nanmin(t)))
                    global_tmax = max(global_tmax, int(np.nanmax(t)))

                if plot_series and (res.samples is not None) and t.size:
                    _upd(res.samples[t - 1])
                if res.bootstrap_means is not None:
                    _upd(res.bootstrap_means)
                if (res.lower_bounds is not None) and (res.upper_bounds is not None):
                    _upd(res.lower_bounds)
                    _upd(res.upper_bounds)
                if (res.uniform_lower is not None) and (res.uniform_upper is not None):
                    _upd(res.uniform_lower)
                    _upd(res.uniform_upper)
                if (res.mu_t is not None) and t.size:
                    _upd(res.mu_t[t - 1])
                if res.smoother_target is not None and t.size:
                    st = np.asarray(res.smoother_target, dtype=float)
                    if st.size >= t.max():
                        _upd(st[t - 1])

            if np.isfinite(global_tmin) and np.isfinite(global_tmax):
                for ax in axs[:R]:
                    ax.set_xlim(float(global_tmin), float(global_tmax))
                    ax.margins(x=0)
            if np.isfinite(global_ymin) and np.isfinite(global_ymax):
                span = global_ymax - global_ymin
                pad = ypad_frac * (span if span > 0 else (abs(global_ymax) or 1.0))
                for ax in axs[:R]:
                    ax.set_ylim(global_ymin - pad, global_ymax + pad)

            try:
                fig.align_xlabels(axs[:R])
                fig.align_ylabels(axs[:R])
            except Exception:
                pass

        if ylabel is not None:
            try:
                if ylabel_on_first:
                    first_ax = axs[0]
                    first_ax.set_ylabel(ylabel, fontsize=axis_labelsize)
                    for ax in axs[1:R]:
                        ax.set_ylabel('')
                        ax.tick_params(labelleft=True)
                else:
                    for ax in axs[:R]:
                        ax.set_ylabel(ylabel, fontsize=axis_labelsize)
            except Exception:
                for ax in axs[:R]:
                    ax.set_ylabel(ylabel, fontsize=axis_labelsize)

        if common_legend:
            from matplotlib.lines import Line2D

            label2handle = {}
            for h, l in zip(handles_all, labels_all, strict=False):
                label2handle[l] = h

            xt_label = r'$X_t$'
            if plot_series:
                if xt_label in label2handle:
                    h = label2handle[xt_label]
                    marker = getattr(h, 'get_marker', lambda: 'o')()
                    color = getattr(h, 'get_color', lambda: None)()
                    mfc = getattr(h, 'get_markerfacecolor', lambda: color)()
                    mec = getattr(h, 'get_markeredgecolor', lambda: 'white')()
                    mew = getattr(h, 'get_markeredgewidth', lambda: 0.25)()
                    alpha_pt = getattr(h, 'get_alpha', lambda: None)()
                    try:
                        if alpha_pt is None and hasattr(h, 'get_markerfacecolor'):
                            fc = h.get_markerfacecolor()
                            if hasattr(fc, '__len__') and len(fc) == 4:
                                alpha_pt = fc[3]
                    except Exception:
                        pass
                    if alpha_pt is None:
                        alpha_pt = 0.36

                    proxy_xt = Line2D(
                        [0],
                        [0],
                        linestyle='None',
                        marker=marker if marker not in (None, 'None') else 'o',
                        markersize=5.0,  # base size; markerscale will apply
                        markerfacecolor=mfc if mfc is not None else color or '0.10',
                        markeredgecolor=mec if mec is not None else 'white',
                        markeredgewidth=mew if mew is not None else 0.25,
                        color=color if color is not None else '0.10',
                        alpha=alpha_pt,
                    )
                    label2handle[xt_label] = proxy_xt
                else:
                    label2handle[xt_label] = Line2D(
                        [0],
                        [0],
                        linestyle='None',
                        marker='o',
                        markersize=5.0,
                        markerfacecolor='0.10',
                        markeredgecolor='white',
                        markeredgewidth=0.25,
                        color='0.10',
                        alpha=0.36,
                    )

            uniq_labels = list(label2handle.keys())
            uniq_handles = [label2handle[lo] for lo in uniq_labels]

            ncol = legend_ncol if legend_ncol is not None else len(uniq_labels)
            if legend_bbox is None and legend_loc.startswith('lower'):
                legend_bbox = (0.5, -0.03)

            fig.legend(
                uniq_handles,
                uniq_labels,
                loc=legend_loc,
                ncol=ncol,
                fontsize=legend_fontsize,
                frameon=False,
                bbox_to_anchor=legend_bbox,
                markerscale=2.8,  # makes small markers readable
                handlelength=1.0,
                handletextpad=0.5,
            )
        else:
            for ax in axs[:R]:
                ax.legend(loc='upper left', fontsize=legend_fontsize)

        if suptitle:
            fig.suptitle(suptitle, fontsize=max(title_size + 2, 14), y=0.99)

        if reduce_whitespace:
            fig.tight_layout(pad=tight_pad)
            top_arg = 0.90 if suptitle else None
            bottom_arg = (
                bottom_pad
                if (common_legend and legend_loc.startswith('lower'))
                else None
            )
            fig.subplots_adjust(
                hspace=hspace, wspace=wspace, top=top_arg, bottom=bottom_arg
            )
        else:
            fig.tight_layout()

        saved_path = self._save_figure(
            fig, save_path, dpi=dpi, save_kwargs=save_kwargs, default_suffix='.pdf'
        )

        if show:
            plt.show()

        return fig, axs.reshape(r, c), saved_path

    def plot_sweep_overlay_grid(
        self,
        panels,
        *,
        order: Optional[Sequence[str]] = None,
        layout: tuple = (2, 2),
        figsize: Optional[tuple] = None,
        dpi: Optional[int] = None,
        plot_both: bool = True,
        metric: str = 'uniform_time',
        show_target: bool = True,
        suptitle: Optional[str] = None,
        common_legend: bool = True,
        legend_loc: str = 'lower center',
        legend_ncol: Optional[int] = None,
        legend_bbox: Optional[tuple] = (0.5, -0.06),
        legend_fontsize: int = 12,
        tick_labelsize: int = 12,
        axis_labelsize: int = 14,
        title_size: int = 14,
        marker_size: int = 5,
        line_width: float = 1.4,
        hspace: float = 0.35,
        wspace: float = 0.12,
        bottom_pad: float = 0.12,
        save_path: Optional[Union[str, Path]] = None,
        save_kwargs: Optional[dict] = None,
        show: bool = True,
        label_left_only: bool = True,
        single_ylabel: Optional[str] = 'Coverage',
        legend_in_panel: bool = False,
        legend_outside_right: bool = False,
        legend_panel_index: int = -1,
        log_eta: bool = False,
        dashed_phi: Optional[Sequence[str]] = ('0.6',),
        style_by_method: bool = False,
        method_from_label: str = 'last_token',  # "last_token" or "regex"
        method_regex: str = r'(AR|WS|iid)$',
        method_order: Optional[Sequence[str]] = None,
        method_colors: Optional[Dict[str, str]] = None,
        color_by_group: bool = False,
        group_from_label: str = 'strip_phi',
        group_regex: str = r'trend\s*=\s*([0-9.eE+-]+)',
        group_colors: Optional[Dict[str, str]] = None,
        ess: bool = False,
    ):
        """Draw a grid where each subplot overlays multiple coverage sweeps.

        Args:
            panels: Mapping from panel title to sweep entries with `sweeps` and
                `labels`.
            order (Optional[Sequence[str]], optional): Panel order override.
            layout (tuple, optional): `(rows, cols)` subplot layout.
            figsize (Optional[tuple], optional): Figure size override.
            dpi (Optional[int], optional): Figure DPI override.
            plot_both (bool, optional): If True, plot both coverage metrics.
            metric (str, optional): Metric to plot when `plot_both=False`.
            show_target (bool, optional): If True, draw the nominal target line.
            suptitle (Optional[str], optional): Optional figure-level title.
            common_legend (bool, optional): If True, draw one common legend.
            legend_loc (str, optional): Figure legend location.
            legend_ncol (Optional[int], optional): Number of legend columns.
            legend_bbox (Optional[tuple], optional): Legend anchor.
            legend_fontsize (int, optional): Legend font size.
            tick_labelsize (int, optional): Tick label font size.
            axis_labelsize (int, optional): Axis label font size.
            title_size (int, optional): Title font size.
            marker_size (int, optional): Marker size.
            line_width (float, optional): Line width.
            hspace (float, optional): Subplot vertical spacing.
            wspace (float, optional): Subplot horizontal spacing.
            bottom_pad (float, optional): Bottom margin for lower legends.
            save_path (Optional[Union[str, Path]], optional): Optional output path.
            save_kwargs (Optional[dict], optional): Extra `fig.savefig` kwargs.
            show (bool, optional): If True, call `plt.show()`.
            label_left_only (bool, optional): If True, label only left panels.
            single_ylabel (Optional[str], optional): Shared y-axis label.
            legend_in_panel (bool, optional): If True, put common legend in a panel.
            legend_outside_right (bool, optional): If True, put common legend to
                the right of the grid.
            legend_panel_index (int, optional): Axis index for in-panel legend.
            log_eta (bool, optional): If True, use log-scaled x-axis.
            dashed_phi (Optional[Sequence[str]], optional): Phi values to dash.
            style_by_method (bool, optional): If True, use method-specific styles.
            method_from_label (str, optional): Method extraction mode.
            method_regex (str, optional): Regex used when `method_from_label="regex"`.
            method_order (Optional[Sequence[str]], optional): Method style order.
            method_colors (Optional[Dict[str, str]], optional): Method colors.
            color_by_group (bool, optional): If True, color curves by group.
            group_from_label (str, optional): Group extraction mode.
            group_regex (str, optional): Regex used when `group_from_label="regex"`.
            group_colors (Optional[Dict[str, str]], optional): Group color
                overrides.
            ess (bool, optional): If True, plot effective sample size instead of eta.

        Returns:
            tuple: `(fig, axes, saved_path)`.

        Raises:
            ValueError: If the panel structure is invalid or the layout is too small.
            RuntimeError: If no visible axis exists for an in-panel legend.
            IndexError: If `legend_panel_index` is out of range.
        """
        keys = list(panels.keys()) if order is None else list(order)
        r, c = layout
        if len(keys) > r * c:
            raise ValueError('More panels than grid cells.')

        use_figsize = self.figsize if figsize is None else figsize
        use_dpi = self.dpi if dpi is None else dpi
        fig, axes = plt.subplots(nrows=r, ncols=c, figsize=use_figsize, dpi=use_dpi)
        axes = np.atleast_1d(axes).ravel()

        base_markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

        handles_all: list = []
        labels_all: list = []

        def _label_is_dashed(label: str) -> bool:
            """Decide whether a label should be drawn with a dashed line.

            Args:
                label (str): Series label to inspect.

            Returns:
                bool: True if the label contains a dashed phi token.
            """
            if not dashed_phi:
                return False
            s = str(label)
            return any((f'\\phi={p}' in s) or (f'phi={p}' in s) for p in dashed_phi)

        def _extract_method(label: str) -> str:
            """Extract a method key from a plotted label.

            Args:
                label (str): Series label to parse.

            Returns:
                str: Extracted method key or the original label.
            """
            s = str(label).strip()
            if method_from_label == 'regex':
                m = re.search(method_regex, s)
                return _canonical_method_key(m.group(1) if m else s)
            parts = s.split()
            return _canonical_method_key(parts[-1] if parts else s)

        def _strip_phi(label: str) -> str:
            """Remove phi tokens from a label.

            Args:
                label (str): Label from which phi annotations are removed.

            Returns:
                str: Label with phi annotations stripped.
            """
            s = str(label)
            s = re.sub(r'\$\\phi\s*=\s*[^$]+\$', '', s)
            s = re.sub(r'\bphi\s*=\s*[0-9.eE+-]+\b', '', s)
            s = re.sub(r'\s{2,}', ' ', s).strip()
            s = s.strip(',;|')
            return s

        def _extract_group(label: str) -> str:
            """Extract a color-group key from a plotted label.

            Args:
                label (str): Series label to parse.

            Returns:
                str: Extracted group key or the original label.
            """
            s = str(label).strip()
            if group_from_label == 'regex':
                m = re.search(group_regex, s)
                return m.group(1) if m else s
            if group_from_label == 'identity':
                return s
            return _strip_phi(s)

        method2color = {}
        method2marker = {}
        if style_by_method:
            method_keys: List[str] = []
            for k in keys:
                entry = panels[k]
                if 'labels' not in entry:
                    continue
                for lab in entry['labels']:
                    m = _extract_method(lab)
                    if m not in method_keys:
                        method_keys.append(m)

            if method_order is not None:
                method_order_keys = [_canonical_method_key(m) for m in method_order]
                ordered = [m for m in method_order_keys if m in method_keys]
                leftovers = [m for m in method_keys if m not in ordered]
                method_keys = ordered + leftovers

            cycle = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [])
            if not cycle:
                cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

            method2color = _method_color_defaults(method_keys, cycle, method_colors)
            method2marker = {
                m: base_markers[i % len(base_markers)]
                for i, m in enumerate(method_keys)
            }

        group2color = {}
        if color_by_group:
            group_keys: List[str] = []
            for k in keys:
                entry = panels[k]
                for lab in entry.get('labels', []):
                    g = _extract_group(lab)
                    if g not in group_keys:
                        group_keys.append(g)

            cycle = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [])
            if not cycle:
                cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
            group2color = _group_color_defaults(group_keys, cycle, group_colors)

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
                alpha_ = float(getattr(s, 'alpha', np.nan))
            else:
                eta = np.asarray(s['eta'], dtype=float)
                y_u_t = np.asarray(s['avg_uniform_time_fraction'], dtype=float)
                y_u_s = np.asarray(s['uniform_over_series_uniform'], dtype=float)
                alpha_ = float(s.get('alpha', np.nan))
            idx = np.argsort(eta)
            return eta[idx], y_u_t[idx], y_u_s[idx], alpha_

        for i, key in enumerate(keys):
            ax = axes[i]
            entry = panels[key]
            if 'sweeps' not in entry or 'labels' not in entry:
                raise ValueError(
                    f"Panel '{key}' must contain 'sweeps' and 'labels' "
                    f'(use load_panels_from_paths).'
                )
            sweeps = entry['sweeps']
            labels = entry['labels']
            if len(sweeps) != len(labels):
                raise ValueError(f"Panel '{key}' has mismatched sweeps/labels lengths.")

            ax.set_title(str(key), fontsize=title_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)

            all_x_for_ticks: list[np.ndarray] = []

            for j, (sw, lab) in enumerate(zip(sweeps, labels, strict=False)):
                eta, y_u_t, y_u_s, alpha_ = _unpack(sw)

                x = _eta_to_ess(eta) if ess else eta
                all_x_for_ticks.append(np.asarray(x, dtype=float))

                dash_this = _label_is_dashed(lab)

                if color_by_group:
                    g = _extract_group(lab)
                    col = group2color.get(g, None)
                    if style_by_method:
                        m = _extract_method(lab)
                        mk = method2marker.get(m, base_markers[j % len(base_markers)])
                    else:
                        mk = base_markers[j % len(base_markers)]
                elif style_by_method:
                    m = _extract_method(lab)
                    mk = method2marker.get(m, base_markers[j % len(base_markers)])
                    col = method2color.get(m, None)
                else:
                    mk = base_markers[j % len(base_markers)]
                    col = None

                if plot_both:
                    (ln1,) = ax.plot(
                        x,
                        y_u_t,
                        marker=mk,
                        markersize=marker_size,
                        lw=line_width,
                        linestyle='--' if dash_this else '-',
                        color=col,
                        label=f'{lab}: avg-time',
                    )
                    (ln2,) = ax.plot(
                        x,
                        y_u_s,
                        marker=mk,
                        markersize=marker_size,
                        lw=line_width,
                        linestyle='--',
                        color=col,
                        label=f'{lab}: over-series',
                    )
                    handles_all.extend([ln1, ln2])
                    labels_all.extend([ln1.get_label(), ln2.get_label()])
                else:
                    if metric == 'uniform_time':
                        y = y_u_t
                    elif metric == 'uniform_series':
                        y = y_u_s
                    else:
                        y = y_u_t

                    (ln,) = ax.plot(
                        x,
                        y,
                        marker=mk,
                        markersize=marker_size,
                        lw=line_width,
                        linestyle='--' if dash_this else '-',
                        color=col,
                        label=f'{lab}',
                    )
                    handles_all.append(ln)
                    labels_all.append(ln.get_label())

                if show_target and np.isfinite(alpha_):
                    ax.axhline(
                        1.0 - alpha_,
                        linestyle=':',
                        linewidth=1.5,
                        color='gray',
                        alpha=0.9,
                        label=None,
                    )

            # x-axis scaling and ticks (log_eta acts as log-x)
            if log_eta:
                ax.set_xscale('log')
                ticks = (
                    np.unique(np.concatenate(all_x_for_ticks))
                    if all_x_for_ticks
                    else np.array([])
                )
                ticks = ticks[np.isfinite(ticks)]
                ticks = ticks[ticks > 0]
                if ticks.size:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f'{v:g}' for v in ticks])
            else:
                ax.margins(x=0)

            ax.set_xlabel(
                (r'$\nu_\eta$' if ess else r'$\eta$'), fontsize=axis_labelsize
            )
            ax.set_ylabel(
                single_ylabel if single_ylabel is not None else '',
                fontsize=axis_labelsize,
            )

            if not common_legend:
                ax.legend(loc='lower right', fontsize=legend_fontsize, frameon=True)

        for j in range(len(keys), r * c):
            axes[j].set_visible(False)

        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles_all, labels_all, strict=False):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)

        if common_legend and uniq_handles:
            if legend_outside_right:
                ncol = legend_ncol if legend_ncol is not None else 1
                right_legend_bbox = (0.84, 0.5)
                if legend_bbox is not None and legend_bbox != (0.5, -0.06):
                    right_legend_bbox = legend_bbox
                fig.legend(
                    uniq_handles,
                    uniq_labels,
                    loc='center left',
                    ncol=ncol,
                    fontsize=legend_fontsize,
                    frameon=False,
                    bbox_to_anchor=right_legend_bbox,
                )
            elif legend_in_panel:
                idx = legend_panel_index if legend_panel_index is not None else -1
                if idx < 0:
                    vis_idxs = [ii for ii, a in enumerate(axes) if a.get_visible()]
                    if not vis_idxs:
                        raise RuntimeError('No visible axes to place legend in.')
                    idx = vis_idxs[-1] if idx < 0 else idx
                if idx < 0 or idx >= len(axes):
                    raise IndexError('legend_panel_index out of range')
                ax_leg = axes[idx]
                ax_leg.legend(
                    uniq_handles,
                    uniq_labels,
                    loc='lower right',
                    fontsize=legend_fontsize,
                    frameon=True,
                )
            else:
                ncol = legend_ncol if legend_ncol is not None else len(uniq_labels)
                fig.legend(
                    uniq_handles,
                    uniq_labels,
                    loc=legend_loc,
                    ncol=ncol,
                    fontsize=legend_fontsize,
                    frameon=False,
                    bbox_to_anchor=legend_bbox,
                )

        # show y-label only in left-most panels if requested
        if label_left_only and single_ylabel is not None:
            left_idxs = {row * c for row in range(r)}
            for ii, ax in enumerate(axes[: r * c]):
                if not ax.get_visible():
                    continue
                if (ii in left_idxs) and (ii < len(keys)):
                    ax.set_ylabel(single_ylabel, fontsize=axis_labelsize)
                    ax.yaxis.set_label_position('left')
                    ax.yaxis.tick_left()
                    ax.tick_params(labelleft=True)
                else:
                    ax.set_ylabel('')
                    ax.tick_params(labelleft=False, left=False)

        if suptitle:
            fig.suptitle(suptitle, fontsize=max(title_size + 2, 14), y=0.995)

        fig.tight_layout(pad=0.6)
        bottom_arg = (
            bottom_pad
            if (
                common_legend
                and (not legend_in_panel)
                and (not legend_outside_right)
                and ('lower' in str(legend_loc).lower())
            )
            else None
        )
        fig.subplots_adjust(
            hspace=hspace,
            wspace=wspace,
            bottom=(bottom_arg if bottom_arg is not None else 0.06),
            right=(0.82 if legend_outside_right else None),
        )

        saved_path = self._save_figure(
            fig, save_path, dpi=dpi, save_kwargs=save_kwargs, default_suffix='.pdf'
        )

        if show:
            plt.show()

        return fig, axes.reshape(r, c), saved_path

    def plot_avg_uniform_width_grid(
        self,
        panels: Dict[str, Dict[str, Any]],
        *,
        order: Optional[Sequence[str]] = None,
        layout: Tuple[int, int] = (2, 2),
        figsize: Optional[Tuple[float, float]] = None,
        dpi: Optional[int] = None,
        marker_size: int = 6,
        line_width: float = 1.4,
        legend_fontsize: int = 12,
        tick_labelsize: int = 12,
        axis_labelsize: int = 14,
        title_size: int = 14,
        hspace: float = 0.35,
        wspace: float = 0.12,
        single_ylabel: str = 'Average interval width',
        label_left_only: bool = True,
        common_legend: bool = True,
        legend_loc: str = 'lower center',
        legend_inside_bottom_right: bool = False,
        legend_outside_right: bool = False,
        legend_ncol: Optional[int] = None,
        legend_bbox: Optional[tuple] = (0.5, -0.06),
        save_path: Optional[Path] = None,
        show: bool = True,
        log_eta: bool = False,
        log_y: bool = False,
        dashed_phi: Optional[Sequence[str]] = ('0.6',),
        style_by_method: bool = False,
        method_from_label: str = 'last_token',  # "last_token" or "regex"
        method_regex: str = r'(AR|WS|iid)$',
        method_order: Optional[Sequence[str]] = None,
        method_colors: Optional[Dict[str, str]] = None,
        ess: bool = False,
    ) -> Tuple['plt.Figure', np.ndarray]:
        r"""Plot a grid of average uniform band widths versus eta or ESS.

        Args:
            panels (Dict[str, Dict[str, Any]]): Panel mapping with `sweeps` and
                `labels`.
            order (Optional[Sequence[str]], optional): Panel order override.
            layout (Tuple[int, int], optional): `(rows, cols)` subplot layout.
            figsize (Optional[Tuple[float, float]], optional): Figure size override.
            dpi (Optional[int], optional): Figure DPI override.
            marker_size (int, optional): Marker size.
            line_width (float, optional): Line width.
            legend_fontsize (int, optional): Legend font size.
            tick_labelsize (int, optional): Tick label font size.
            axis_labelsize (int, optional): Axis label font size.
            title_size (int, optional): Title font size.
            hspace (float, optional): Subplot vertical spacing.
            wspace (float, optional): Subplot horizontal spacing.
            single_ylabel (str, optional): Shared y-axis label.
            label_left_only (bool, optional): If True, label only left panels.
            common_legend (bool, optional): If True, draw one common legend.
            legend_loc (str, optional): Figure legend location.
            legend_inside_bottom_right (bool, optional): If True, place the legend
                in the bottom-right panel.
            legend_outside_right (bool, optional): If True, place the legend to
                the right of the grid.
            legend_ncol (Optional[int], optional): Number of legend columns.
            legend_bbox (Optional[tuple], optional): Legend anchor.
            save_path (Optional[Path], optional): Optional output path.
            show (bool, optional): If True, call `plt.show()`.
            log_eta (bool, optional): If True, use log-scaled x-axis.
            log_y (bool, optional): If True, use log-scaled y-axis.
            dashed_phi (Optional[Sequence[str]], optional): Phi values to dash.
            style_by_method (bool, optional): If True, use method-specific styles.
            method_from_label (str, optional): Method extraction mode.
            method_regex (str, optional): Regex used when `method_from_label="regex"`.
            method_order (Optional[Sequence[str]], optional): Method style order.
            method_colors (Optional[Dict[str, str]], optional): Method colors.
            ess (bool, optional): If True, plot effective sample size instead of eta.

        Returns:
            Tuple[plt.Figure, np.ndarray]: Figure and axes array.

        Raises:
            ValueError: If the panel structure is invalid, the layout is too small,
                or log scaling receives invalid values.
            RuntimeError: If no compatible width metric is found in a sweep.
        """
        keys = list(panels.keys()) if order is None else list(order)
        r, c = layout
        if len(keys) > r * c:
            raise ValueError('More panels than grid cells.')

        use_figsize = self.figsize if figsize is None else figsize
        use_dpi = self.dpi if dpi is None else dpi
        fig, axes = plt.subplots(nrows=r, ncols=c, figsize=use_figsize, dpi=use_dpi)
        axes = np.atleast_1d(axes).ravel()

        base_markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

        handles_all: list = []
        labels_all: list = []

        def _label_is_dashed(label: str) -> bool:
            """Decide whether a label should be drawn with a dashed line.

            Args:
                label (str): Series label to inspect.

            Returns:
                bool: True if the label contains a dashed phi token.
            """
            if not dashed_phi:
                return False
            s = str(label)
            return any((f'\\phi={p}' in s) or (f'phi={p}' in s) for p in dashed_phi)

        def _extract_method(label: str) -> str:
            """Extract a method key from a plotted label.

            Args:
                label (str): Series label to parse.

            Returns:
                str: Extracted method key or the original label.
            """
            s = str(label).strip()
            if method_from_label == 'regex':
                m = re.search(method_regex, s)
                return _canonical_method_key(m.group(1) if m else s)
            parts = s.split()
            return _canonical_method_key(parts[-1] if parts else s)

        method2color = {}
        method2marker = {}
        if style_by_method:
            method_keys: List[str] = []
            for k in keys:
                entry = panels[k]
                if 'labels' not in entry:
                    continue
                for lab in entry['labels']:
                    m = _extract_method(lab)
                    if m not in method_keys:
                        method_keys.append(m)

            if method_order is not None:
                method_order_keys = [_canonical_method_key(m) for m in method_order]
                ordered = [m for m in method_order_keys if m in method_keys]
                leftovers = [m for m in method_keys if m not in ordered]
                method_keys = ordered + leftovers

            cycle = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [])
            if not cycle:
                cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

            method2color = _method_color_defaults(method_keys, cycle, method_colors)
            method2marker = {
                m: base_markers[i % len(base_markers)]
                for i, m in enumerate(method_keys)
            }

        for i, key in enumerate(keys):
            ax = axes[i]
            entry = panels[key]
            if 'sweeps' not in entry or 'labels' not in entry:
                raise ValueError(
                    f"Panel '{key}' must contain 'sweeps' and 'labels'. "
                    'Use load_panels_from_paths(...) to build `panels` or provide that structure.'
                )
            sweeps = entry['sweeps']
            labels = entry['labels']
            if len(sweeps) != len(labels):
                raise ValueError(f"Panel '{key}' has mismatched sweeps/labels lengths.")

            ax.set_title(str(key), fontsize=title_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
            ax.grid(True, linestyle='--', alpha=0.35)

            x_all: list[float] = []
            y_all: list[float] = []

            for j, (sw, lab) in enumerate(zip(sweeps, labels)):
                if hasattr(sw, 'eta'):
                    eta = np.asarray(sw.eta, dtype=float)
                    if hasattr(sw, 'avg_uniform_mean_width'):
                        y = np.asarray(sw.avg_uniform_mean_width, dtype=float)
                    elif hasattr(sw, 'avg_uniform_width'):
                        y = np.asarray(sw.avg_uniform_width, dtype=float)
                    elif hasattr(sw, 'series_uniform_mean_width'):
                        arr = np.asarray(sw.series_uniform_mean_width, dtype=float)
                        y = np.full_like(eta, float(np.nanmean(arr)), dtype=float)
                    else:
                        y = None
                else:
                    eta = np.asarray(sw['eta'], dtype=float)
                    if 'avg_uniform_mean_width' in sw:
                        y = np.asarray(sw['avg_uniform_mean_width'], dtype=float)
                    elif 'avg_uniform_width' in sw:
                        y = np.asarray(sw['avg_uniform_width'], dtype=float)
                    elif 'series_uniform_mean_width' in sw:
                        arr = np.asarray(sw['series_uniform_mean_width'], dtype=float)
                        y = np.full_like(eta, float(np.nanmean(arr)), dtype=float)
                    else:
                        y = None

                if y is None:
                    raise RuntimeError(
                        'Could not find avg_uniform_mean_width (or a compatible fallback) for sweep '
                        f"'{lab}' in panel '{key}'."
                    )

                idx = np.argsort(eta)
                eta = eta[idx]
                y = y[idx]

                x = _eta_to_ess(eta) if ess else eta

                ls = '--' if _label_is_dashed(lab) else '-'

                if style_by_method:
                    m = _extract_method(lab)
                    mk = method2marker.get(m, base_markers[j % len(base_markers)])
                    col = method2color.get(m, None)
                else:
                    mk = base_markers[j % len(base_markers)]
                    col = None

                (ln,) = ax.plot(
                    x,
                    y,
                    marker=mk,
                    markersize=marker_size,
                    lw=line_width,
                    linestyle=ls,
                    color=col,
                    label=str(lab),
                )
                handles_all.append(ln)
                labels_all.append(ln.get_label())

                if log_eta:
                    x_all.extend(np.asarray(x, dtype=float).tolist())
                if log_y:
                    y_all.extend(np.asarray(y, dtype=float).tolist())

            ax.set_xlabel(
                (r'$\nu_\eta$' if ess else r'$\eta$'), fontsize=axis_labelsize
            )
            ax.set_ylabel(single_ylabel, fontsize=axis_labelsize)

            if log_eta:
                ax.set_xscale('log')
                ticks = np.unique(np.asarray(x_all, dtype=float))
                ticks = ticks[np.isfinite(ticks)]
                ticks = ticks[ticks > 0]
                if ticks.size:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f'{v:g}' for v in ticks])
                    ax.set_xlim(ticks.min(), ticks.max())
                    ax.minorticks_off()
            else:
                ax.margins(x=0)

            if log_y:
                y_arr = np.asarray(y_all, dtype=float)
                if np.any(~np.isfinite(y_arr)):
                    raise ValueError(
                        f"log_y=True but non-finite y-values found in panel '{key}'."
                    )
                if np.any(y_arr <= 0):
                    raise ValueError(
                        f"log_y=True but y-values <= 0 found in panel '{key}'. "
                        'Log scale requires strictly positive y.'
                    )
                ax.set_yscale('log')

        for j in range(len(keys), r * c):
            axes[j].set_visible(False)

        # dedupe legend entries
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles_all, labels_all, strict=False):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)

        if common_legend and uniq_handles:
            if legend_outside_right:
                right_legend_bbox = (0.84, 0.5)
                if legend_bbox is not None and legend_bbox != (0.5, -0.06):
                    right_legend_bbox = legend_bbox
                fig.legend(
                    uniq_handles,
                    uniq_labels,
                    loc='center left',
                    ncol=1 if legend_ncol is None else legend_ncol,
                    fontsize=legend_fontsize,
                    frameon=False,
                    bbox_to_anchor=right_legend_bbox,
                )
            elif legend_inside_bottom_right or legend_loc == 'inplot-br':
                br_idx = (r - 1) * c + (c - 1)
                ax_legend = None
                if 0 <= br_idx < len(axes) and axes[br_idx].get_visible():
                    ax_legend = axes[br_idx]
                else:
                    vis_idxs = [ii for ii, a in enumerate(axes) if a.get_visible()]
                    if vis_idxs:
                        ax_legend = axes[vis_idxs[-1]]
                if ax_legend is None:
                    fig.legend(
                        uniq_handles,
                        uniq_labels,
                        loc='lower center',
                        ncol=len(uniq_labels) if legend_ncol is None else legend_ncol,
                        fontsize=legend_fontsize,
                        frameon=False,
                        bbox_to_anchor=legend_bbox,
                    )
                else:
                    ax_legend.legend(
                        uniq_handles,
                        uniq_labels,
                        loc='lower right',
                        ncol=1 if legend_ncol is None else legend_ncol,
                        fontsize=legend_fontsize,
                        frameon=True,
                        handlelength=1.2,
                        handletextpad=0.6,
                        labelspacing=0.4,
                        borderaxespad=0.6,
                    )
            else:
                fig.legend(
                    uniq_handles,
                    uniq_labels,
                    loc=legend_loc,
                    ncol=len(uniq_labels) if legend_ncol is None else legend_ncol,
                    fontsize=legend_fontsize,
                    frameon=False,
                    bbox_to_anchor=legend_bbox,
                )
        else:
            vis_idxs = [ii for ii, a in enumerate(axes) if a.get_visible()]
            for idx in vis_idxs:
                axes[idx].legend(
                    loc='lower right', fontsize=legend_fontsize, frameon=True
                )

        if label_left_only:
            left_idxs = {row * c for row in range(r)}
            for ii, ax in enumerate(axes[: r * c]):
                if not ax.get_visible():
                    continue
                if ii in left_idxs:
                    ax.set_ylabel(single_ylabel, fontsize=axis_labelsize)
                    ax.yaxis.set_label_position('left')
                    ax.yaxis.tick_left()
                    ax.tick_params(labelleft=True)
                else:
                    ax.set_ylabel('')
                    ax.tick_params(labelleft=False, left=False)

        fig.tight_layout(pad=0.6)
        fig.subplots_adjust(
            hspace=hspace,
            wspace=wspace,
            bottom=0.08,
            right=(0.82 if legend_outside_right else None),
        )

        self._save_figure(fig, save_path, default_suffix='.pdf')

        if show:
            plt.show()

        return fig, axes.reshape(r, c)

    def plot_power_overlay_grid(
        self,
        panels,
        *,
        order: Optional[Sequence[str]] = None,
        layout: tuple = (2, 2),
        figsize: Optional[tuple] = None,
        dpi: Optional[int] = None,
        metric: str = 'power',
        show_target: bool = True,
        target_level: Optional[float] = None,
        show_burn: bool = True,
        suptitle: Optional[str] = None,
        common_legend: bool = True,
        legend_loc: str = 'lower center',
        legend_ncol: Optional[int] = None,
        legend_bbox: Optional[tuple] = (0.5, -0.06),
        legend_fontsize: int = 12,
        tick_labelsize: int = 12,
        axis_labelsize: int = 14,
        title_size: int = 14,
        line_width: float = 1.4,
        hspace: float = 0.35,
        wspace: float = 0.12,
        bottom_pad: float = 0.12,
        save_path: Optional[Union[str, Path]] = None,
        save_kwargs: Optional[dict] = None,
        show: bool = True,
        label_left_only: bool = True,
        single_ylabel: Optional[str] = 'Power',
        legend_in_panel: bool = False,
        legend_outside_right: bool = False,
        legend_panel_index: int = -1,
        log_time: bool = False,
        dashed_phi: Optional[Sequence[str]] = ('0.6',),
        style_by_method: bool = False,
        method_from_label: str = 'last_token',
        method_regex: str = r'(AR|WS|iid)$',
        method_order: Optional[Sequence[str]] = None,
        color_by_group: bool = False,
        group_from_label: str = 'strip_phi',
        group_regex: str = r'trend\s*=\s*([0-9.eE+-]+)',
        group_colors: Optional[Dict[str, str]] = None,
    ):
        """Draw a grid where each subplot overlays multiple *power-vs-time* curves.

        Args:
            panels: Mapping from panel title to run entries with `runs` and `labels`.
            order (Optional[Sequence[str]], optional): Panel order override.
            layout (tuple, optional): `(rows, cols)` subplot layout.
            figsize (Optional[tuple], optional): Figure size override.
            dpi (Optional[int], optional): Figure DPI override.
            metric (str, optional): "power" for reject-by-time probability or
                "survival" for one minus power.
            show_target (bool, optional): If True, draw target rejection level.
            target_level (Optional[float], optional): Explicit target level.
            show_burn (bool, optional): If True, draw the burn-in/warmup line.
            suptitle (Optional[str], optional): Optional figure-level title.
            common_legend (bool, optional): If True, draw one common legend.
            legend_loc (str, optional): Figure legend location.
            legend_ncol (Optional[int], optional): Number of legend columns.
            legend_bbox (Optional[tuple], optional): Legend anchor.
            legend_fontsize (int, optional): Legend font size.
            tick_labelsize (int, optional): Tick label font size.
            axis_labelsize (int, optional): Axis label font size.
            title_size (int, optional): Title font size.
            line_width (float, optional): Line width.
            hspace (float, optional): Subplot vertical spacing.
            wspace (float, optional): Subplot horizontal spacing.
            bottom_pad (float, optional): Bottom margin for lower legends.
            save_path (Optional[Union[str, Path]], optional): Optional output path.
            save_kwargs (Optional[dict], optional): Extra `fig.savefig` kwargs.
            show (bool, optional): If True, call `plt.show()`.
            label_left_only (bool, optional): If True, label only left panels.
            single_ylabel (Optional[str], optional): Shared y-axis label.
            legend_in_panel (bool, optional): If True, place legend in a panel.
            legend_outside_right (bool, optional): If True, place legend to the
                right of the grid.
            legend_panel_index (int, optional): Axis index for in-panel legend.
            log_time (bool, optional): If True, use log-scaled time axis.
            dashed_phi (Optional[Sequence[str]], optional): Phi values to dash.
            style_by_method (bool, optional): If True, use method-specific markers.
            method_from_label (str, optional): Method extraction mode.
            method_regex (str, optional): Regex used when `method_from_label="regex"`.
            method_order (Optional[Sequence[str]], optional): Method style order.
            color_by_group (bool, optional): If True, color curves by group.
            group_from_label (str, optional): Group extraction mode.
            group_regex (str, optional): Regex used when `group_from_label="regex"`.
            group_colors (Optional[Dict[str, str]], optional): Group color
                overrides.

        Returns:
            tuple: `(fig, axes, saved_path)`.

        Raises:
            ValueError: If the panel structure is invalid or the layout is too small.
            RuntimeError: If no visible axis exists for an in-panel legend.
        """
        keys = list(panels.keys()) if order is None else list(order)
        r, c = layout
        if len(keys) > r * c:
            raise ValueError('More panels than grid cells.')

        use_figsize = self.figsize if figsize is None else figsize
        use_dpi = self.dpi if dpi is None else dpi
        fig, axes = plt.subplots(nrows=r, ncols=c, figsize=use_figsize, dpi=use_dpi)
        axes = np.atleast_1d(axes).ravel()

        handles_all: list = []
        labels_all: list = []

        def _label_is_dashed(label: str) -> bool:
            """Decide whether a label should be drawn with a dashed line.

            Args:
                label (str): Series label to inspect.

            Returns:
                bool: True if the label contains a dashed phi token.
            """
            if not dashed_phi:
                return False
            s = str(label)
            return any((f'\\phi={p}' in s) or (f'phi={p}' in s) for p in dashed_phi)

        def _extract_method(label: str) -> str:
            """Extract a method key from a plotted label.

            Args:
                label (str): Series label to parse.

            Returns:
                str: Extracted method key or the original label.
            """
            s = str(label).strip()
            if method_from_label == 'regex':
                m = re.search(method_regex, s)
                return _canonical_method_key(m.group(1) if m else s)
            parts = s.split()
            return _canonical_method_key(parts[-1] if parts else s)

        def _strip_phi(label: str) -> str:
            """Remove phi tokens from a label.

            Args:
                label (str): Label from which phi annotations are removed.

            Returns:
                str: Label with phi annotations stripped.
            """
            s = str(label)
            s = re.sub(r'\$\\phi\s*=\s*[^$]+\$', '', s)
            s = re.sub(r'\bphi\s*=\s*[0-9.eE+-]+\b', '', s)
            s = re.sub(r'\s{2,}', ' ', s).strip()
            s = s.strip(',;|')
            return s

        def _extract_group(label: str) -> str:
            """Extract a color-group key from a plotted label.

            Args:
                label (str): Series label to parse.

            Returns:
                str: Extracted group key or the original label.
            """
            s = str(label).strip()
            if group_from_label == 'regex':
                m = re.search(group_regex, s)
                return m.group(1) if m else s
            if group_from_label == 'identity':
                return s
            # default: strip phi from label
            return _strip_phi(s)

        base_markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

        method2color = {}
        method2marker = {}
        if style_by_method:
            method_keys: List[str] = []
            for k in keys:
                entry = panels[k]
                for lab in entry.get('labels', []):
                    m = _extract_method(lab)
                    if m not in method_keys:
                        method_keys.append(m)

            if method_order is not None:
                method_order_keys = [_canonical_method_key(m) for m in method_order]
                ordered = [m for m in method_order_keys if m in method_keys]
                leftovers = [m for m in method_keys if m not in ordered]
                method_keys = ordered + leftovers

            cycle = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [])
            if not cycle:
                cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

            method2color = _method_color_defaults(method_keys, cycle)
            method2marker = {
                m: base_markers[i % len(base_markers)]
                for i, m in enumerate(method_keys)
            }

        group2color = {}
        if color_by_group:
            group_keys: List[str] = []
            for k in keys:
                entry = panels[k]
                for lab in entry.get('labels', []):
                    g = _extract_group(lab)
                    if g not in group_keys:
                        group_keys.append(g)

            cycle = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [])
            if not cycle:
                cycle = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
            group2color = _group_color_defaults(group_keys, cycle, group_colors)

        def _unpack_run(run):
            """Extract arrays and metadata from a power run.

            Args:
                run: Power-run object or mapping.

            Returns:
                tuple[np.ndarray, np.ndarray, Optional[int], float]: Times, power
                    curve, effective burn-in, and alpha.
            """
            if hasattr(run, 'times'):
                t = np.asarray(run.times, dtype=int)
                pc = np.asarray(run.power_curve, dtype=float)
                be = getattr(run, 'burn_eff', None)
                a = getattr(run, 'alpha', np.nan)
            else:
                t = np.asarray(run['times'], dtype=int)
                pc = np.asarray(run['power_curve'], dtype=float)
                be = run.get('burn_eff', None)
                a = float(run.get('alpha', np.nan))
            T = min(t.size, pc.size)
            return t[:T], pc[:T], be, a

        for i, key in enumerate(keys):
            ax = axes[i]
            entry = panels[key]
            if 'runs' not in entry or 'labels' not in entry:
                raise ValueError(f"Panel '{key}' must contain 'runs' and 'labels'.")

            runs = entry['runs']
            labels = entry['labels']
            if len(runs) != len(labels):
                raise ValueError(f"Panel '{key}' has mismatched runs/labels lengths.")

            ax.set_title(str(key), fontsize=title_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, linestyle='--', alpha=0.5)

            burn_eff_panel = None
            alpha_panel = np.nan

            for j, (run, lab) in enumerate(zip(runs, labels)):
                t, pc, be, a = _unpack_run(run)
                if burn_eff_panel is None and be is not None:
                    burn_eff_panel = int(be)
                if np.isfinite(a) and not np.isfinite(alpha_panel):
                    alpha_panel = float(a)

                y = pc if metric == 'power' else (1.0 - pc)
                ls = '--' if _label_is_dashed(lab) else '-'

                col = None
                if color_by_group:
                    g = _extract_group(lab)
                    col = group2color.get(g, None)
                elif style_by_method:
                    m = _extract_method(lab)
                    col = method2color.get(m, None)

                if style_by_method:
                    m = _extract_method(lab)
                    mk = method2marker.get(m, base_markers[j % len(base_markers)])
                    (ln,) = ax.plot(
                        t,
                        y,
                        lw=line_width,
                        linestyle=ls,
                        color=col,
                        marker=mk,
                        markersize=3.5,
                        markevery=max(1, len(t) // 12),
                        label=str(lab),
                    )
                else:
                    (ln,) = ax.plot(
                        t,
                        y,
                        lw=line_width,
                        linestyle=ls,
                        color=col,
                        label=str(lab),
                    )

                handles_all.append(ln)
                labels_all.append(ln.get_label())

            if show_burn and burn_eff_panel is not None:
                ax.axvline(
                    burn_eff_panel,
                    linestyle='--',
                    linewidth=1.2,
                    color='gray',
                    alpha=0.85,
                )

            if show_target:
                lvl = target_level
                if lvl is None and np.isfinite(alpha_panel):
                    lvl = float(alpha_panel)
                if lvl is not None and np.isfinite(lvl):
                    ax.axhline(
                        lvl,
                        linestyle=':',
                        linewidth=1.5,
                        color='gray',
                        alpha=0.9,
                    )

            if log_time:
                ax.set_xscale('log')
            else:
                ax.margins(x=0)

            ax.set_xlabel('time $t$', fontsize=axis_labelsize)
            if single_ylabel is not None:
                ax.set_ylabel(single_ylabel, fontsize=axis_labelsize)

            if not common_legend:
                ax.legend(loc='lower right', fontsize=legend_fontsize, frameon=True)

        for j in range(len(keys), r * c):
            axes[j].set_visible(False)

        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles_all, labels_all, strict=False):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)

        if common_legend and uniq_handles:
            if legend_outside_right:
                ncol = legend_ncol if legend_ncol is not None else 1
                right_legend_bbox = (0.84, 0.5)
                if legend_bbox is not None and legend_bbox != (0.5, -0.06):
                    right_legend_bbox = legend_bbox
                fig.legend(
                    uniq_handles,
                    uniq_labels,
                    loc='center left',
                    ncol=ncol,
                    fontsize=legend_fontsize,
                    frameon=False,
                    bbox_to_anchor=right_legend_bbox,
                )
            elif legend_in_panel:
                idx = legend_panel_index if legend_panel_index is not None else -1
                if idx < 0:
                    vis_idxs = [ii for ii, a in enumerate(axes) if a.get_visible()]
                    if not vis_idxs:
                        raise RuntimeError('No visible axes to place legend in.')
                    idx = vis_idxs[-1]
                ax_leg = axes[idx]
                ax_leg.legend(
                    uniq_handles,
                    uniq_labels,
                    loc='lower right',
                    fontsize=legend_fontsize,
                    frameon=True,
                )
            else:
                ncol = legend_ncol if legend_ncol is not None else len(uniq_labels)
                fig.legend(
                    uniq_handles,
                    uniq_labels,
                    loc=legend_loc,
                    ncol=ncol,
                    fontsize=legend_fontsize,
                    frameon=False,
                    bbox_to_anchor=legend_bbox,
                )

        if label_left_only and single_ylabel is not None:
            left_idxs = {row * c for row in range(r)}
            for ii, ax in enumerate(axes[: r * c]):
                if not ax.get_visible():
                    continue
                if (ii in left_idxs) and (ii < len(keys)):
                    ax.set_ylabel(single_ylabel, fontsize=axis_labelsize)
                    ax.yaxis.set_label_position('left')
                    ax.yaxis.tick_left()
                    ax.tick_params(labelleft=True)
                else:
                    ax.set_ylabel('')
                    ax.tick_params(labelleft=False, left=False)

        if suptitle:
            fig.suptitle(suptitle, fontsize=max(title_size + 2, 14), y=0.995)

        fig.tight_layout(pad=0.6)
        bottom_arg = (
            bottom_pad
            if (
                common_legend
                and (not legend_in_panel)
                and (not legend_outside_right)
                and ('lower' in str(legend_loc).lower())
            )
            else None
        )
        fig.subplots_adjust(
            hspace=hspace,
            wspace=wspace,
            bottom=(bottom_arg if bottom_arg is not None else 0.06),
            right=(0.82 if legend_outside_right else None),
        )

        saved_path = self._save_figure(
            fig, save_path, dpi=dpi, save_kwargs=save_kwargs, default_suffix='.pdf'
        )

        if show:
            plt.show()

        return fig, axes.reshape(r, c), saved_path

    def plot_runtime_overlay_grid(
        self,
        runtime_df,
        *,
        order: Optional[Sequence[str]] = None,
        layout: tuple = (1, 3),
        figsize: Optional[tuple] = None,
        dpi: Optional[int] = None,
        panel_col: str = 'panel',
        method_col: str = 'method',
        eta_col: str = 'eta',
        runtime_col: str = 'runtime_sec',
        summary_col: str = 'runtime_median_sec',
        qlow_col: str = 'runtime_q25_sec',
        qhigh_col: str = 'runtime_q75_sec',
        method_order: Optional[Sequence[str]] = ('ours', 'block'),
        method_display: Optional[Dict[str, str]] = None,
        show_error_band: bool = True,
        suptitle: Optional[str] = None,
        common_legend: bool = True,
        legend_loc: str = 'lower center',
        legend_ncol: Optional[int] = None,
        legend_bbox: Optional[tuple] = (0.5, -0.06),
        legend_fontsize: int = 12,
        tick_labelsize: int = 12,
        axis_labelsize: int = 14,
        title_size: int = 14,
        marker_size: int = 5,
        line_width: float = 1.4,
        hspace: float = 0.35,
        wspace: float = 0.12,
        bottom_pad: float = 0.12,
        save_path: Optional[Union[str, Path]] = None,
        save_kwargs: Optional[dict] = None,
        show: bool = True,
        label_left_only: bool = True,
        single_ylabel: Optional[str] = 'Runtime (seconds)',
        log_eta: bool = False,
        log_runtime: bool = True,
        ess: bool = False,
    ):
        """Plot runtime comparisons in a grid of panels.

        The input can be either:
        1) a raw runtime dataframe with repeated measurements and a column
        `runtime_sec`, or
        2) an already summarized dataframe containing `summary_col`.

        If the dataframe contains only raw runtimes, it is summarized internally.

        Args:
            runtime_df: Runtime dataframe to plot.
            order (Optional[Sequence[str]], optional): Panel order override.
            layout (tuple, optional): `(rows, cols)` subplot layout.
            figsize (Optional[tuple], optional): Figure size override.
            dpi (Optional[int], optional): Figure DPI override.
            panel_col (str, optional): Column containing panel labels.
            method_col (str, optional): Column containing method labels.
            eta_col (str, optional): Column containing eta values.
            runtime_col (str, optional): Raw runtime column.
            summary_col (str, optional): Summary runtime column.
            qlow_col (str, optional): Lower quantile column for error bands.
            qhigh_col (str, optional): Upper quantile column for error bands.
            method_order (Optional[Sequence[str]], optional): Method plotting order.
            method_display (Optional[Dict[str, str]], optional): Method display names.
            show_error_band (bool, optional): If True, draw runtime error bands.
            suptitle (Optional[str], optional): Optional figure-level title.
            common_legend (bool, optional): If True, draw one common legend.
            legend_loc (str, optional): Figure legend location.
            legend_ncol (Optional[int], optional): Number of legend columns.
            legend_bbox (Optional[tuple], optional): Legend anchor.
            legend_fontsize (int, optional): Legend font size.
            tick_labelsize (int, optional): Tick label font size.
            axis_labelsize (int, optional): Axis label font size.
            title_size (int, optional): Title font size.
            marker_size (int, optional): Marker size.
            line_width (float, optional): Line width.
            hspace (float, optional): Subplot vertical spacing.
            wspace (float, optional): Subplot horizontal spacing.
            bottom_pad (float, optional): Bottom margin for lower legends.
            save_path (Optional[Union[str, Path]], optional): Optional output path.
            save_kwargs (Optional[dict], optional): Extra `fig.savefig` kwargs.
            show (bool, optional): If True, call `plt.show()`.
            label_left_only (bool, optional): If True, label only left panels.
            single_ylabel (Optional[str], optional): Shared y-axis label.
            log_eta (bool, optional): If True, use log-scaled x-axis.
            log_runtime (bool, optional): If True, use log-scaled runtime axis.
            ess (bool, optional): If True, plot effective sample size instead of eta.

        Returns:
            tuple: `(fig, axes, saved_path)`.

        Raises:
            ValueError: If the runtime dataframe is empty or the layout is too small.
        """
        if runtime_df is None or len(runtime_df) == 0:
            raise ValueError('runtime_df must be a non-empty dataframe.')

        df = runtime_df.copy()

        # If only raw runtimes are present, summarize first
        if summary_col not in df.columns:
            df = summarize_runtime_results(
                df,
                panel_col=panel_col,
                method_col=method_col,
                eta_col=eta_col,
                runtime_col=runtime_col,
            )

        keys = list(df[panel_col].drop_duplicates()) if order is None else list(order)
        r, c = layout
        if len(keys) > r * c:
            raise ValueError('More panels than grid cells.')

        use_figsize = self.figsize if figsize is None else figsize
        use_dpi = self.dpi if dpi is None else dpi
        fig, axes = plt.subplots(nrows=r, ncols=c, figsize=use_figsize, dpi=use_dpi)
        axes = np.atleast_1d(axes).ravel()

        base_markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

        # consistent colors by method
        cycle = mpl.rcParams['axes.prop_cycle'].by_key().get('color', [])
        if not cycle:
            cycle = ['C0', 'C1', 'C2', 'C3']

        method_keys = list(df[method_col].drop_duplicates())
        if method_order is not None:
            ordered = [m for m in method_order if m in method_keys]
            leftovers = [m for m in method_keys if m not in ordered]
            method_keys = ordered + leftovers

        method2color = {m: cycle[i % len(cycle)] for i, m in enumerate(method_keys)}
        method2marker = {
            m: base_markers[i % len(base_markers)] for i, m in enumerate(method_keys)
        }

        handles_all = []
        labels_all = []

        for i, key in enumerate(keys):
            ax = axes[i]
            sub = df.loc[df[panel_col] == key].copy()

            if sub.empty:
                ax.set_visible(False)
                continue

            ax.set_title(str(key), fontsize=title_size)
            ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
            ax.grid(True, linestyle='--', alpha=0.5)

            all_x_for_ticks = []

            methods_here = list(sub[method_col].drop_duplicates())
            if method_order is not None:
                ordered = [m for m in method_order if m in methods_here]
                leftovers = [m for m in methods_here if m not in ordered]
                methods_here = ordered + leftovers

            for j, method in enumerate(methods_here):
                dd = sub.loc[sub[method_col] == method].copy()
                dd = dd.sort_values(eta_col)

                eta = dd[eta_col].to_numpy(dtype=float)
                x = _eta_to_ess(eta) if ess else eta
                y = dd[summary_col].to_numpy(dtype=float)

                all_x_for_ticks.append(np.asarray(x, dtype=float))

                col = method2color.get(method, None)
                mk = method2marker.get(method, base_markers[j % len(base_markers)])
                label = (
                    method_display.get(method, method)
                    if method_display is not None
                    else str(method)
                )

                (ln,) = ax.plot(
                    x,
                    y,
                    marker=mk,
                    markersize=marker_size,
                    lw=line_width,
                    color=col,
                    label=label,
                )
                handles_all.append(ln)
                labels_all.append(ln.get_label())

                if (
                    show_error_band
                    and (qlow_col in dd.columns)
                    and (qhigh_col in dd.columns)
                ):
                    lo = dd[qlow_col].to_numpy(dtype=float)
                    hi = dd[qhigh_col].to_numpy(dtype=float)
                    band = ax.fill_between(
                        x,
                        lo,
                        hi,
                        alpha=0.18,
                        color=col,
                        linewidth=0.0,
                    )
                    try:
                        band.set_edgecolor('none')
                    except Exception:
                        pass

            if log_eta:
                ax.set_xscale('log')
                ticks = (
                    np.unique(np.concatenate(all_x_for_ticks))
                    if all_x_for_ticks
                    else np.array([])
                )
                ticks = ticks[np.isfinite(ticks)]
                ticks = ticks[ticks > 0]
                if ticks.size:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f'{v:g}' for v in ticks])

            if log_runtime:
                # only possible if all positive
                positive_y = sub[summary_col].to_numpy(dtype=float)
                if np.all(np.isfinite(positive_y)) and np.all(positive_y > 0):
                    ax.set_yscale('log')

            ax.set_xlabel(
                (r'$\nu_\eta$' if ess else r'$\eta$'), fontsize=axis_labelsize
            )
            ax.set_ylabel(
                single_ylabel if single_ylabel is not None else '',
                fontsize=axis_labelsize,
            )

            if not log_eta:
                ax.margins(x=0)

            if not common_legend:
                ax.legend(loc='best', fontsize=legend_fontsize, frameon=True)

        for j in range(len(keys), r * c):
            axes[j].set_visible(False)

        # common legend
        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles_all, labels_all):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)

        if common_legend and uniq_handles:
            ncol = legend_ncol if legend_ncol is not None else len(uniq_labels)
            fig.legend(
                uniq_handles,
                uniq_labels,
                loc=legend_loc,
                ncol=ncol,
                fontsize=legend_fontsize,
                frameon=False,
                bbox_to_anchor=legend_bbox,
            )

        if label_left_only and single_ylabel is not None:
            left_idxs = {row * c for row in range(r)}
            for ii, ax in enumerate(axes[: r * c]):
                if not ax.get_visible():
                    continue
                if ii in left_idxs and ii < len(keys):
                    ax.set_ylabel(single_ylabel, fontsize=axis_labelsize)
                    ax.yaxis.set_label_position('left')
                    ax.yaxis.tick_left()
                    ax.tick_params(labelleft=True)
                else:
                    ax.set_ylabel('')
                    ax.tick_params(labelleft=False, left=False)

        if suptitle:
            fig.suptitle(suptitle, fontsize=max(title_size + 2, 14), y=0.995)

        fig.tight_layout(pad=0.6)
        bottom_arg = (
            bottom_pad
            if (common_legend and ('lower' in str(legend_loc).lower()))
            else None
        )
        fig.subplots_adjust(
            hspace=hspace,
            wspace=wspace,
            bottom=(bottom_arg if bottom_arg is not None else 0.06),
        )

        saved_path = self._save_figure(
            fig, save_path, dpi=dpi, save_kwargs=save_kwargs, default_suffix='.pdf'
        )

        if show:
            plt.show()

        return fig, axes.reshape(r, c), saved_path
