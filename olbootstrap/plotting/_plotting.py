from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401

from olbootstrap.experiments._base_experiment import ExperimentResults

from ._baseplotting import BasePlotter


class BootstrapPlotter(BasePlotter):
    """Plotter specialized for bootstrap experiments.

    Currently this class inherits all style handling from BasePlotter. Add
    custom plotting helpers here (e.g. plot_sweep, plot_sweep_overlay_grid).
    """

    def __init__(
        self,
        style: str = 'science',
        figsize: Tuple[float, float] = (12, 6),
        dpi: int = 150,
        rc_overrides: Optional[Dict[str, Any]] = None,
    ):
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
        if res.times is None:
            raise RuntimeError('res.times required')

        # Figure/axes setup
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

        # draw
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

        # layout / save / show
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
        """Plot multiple ExperimentResults in a grid (e.g., 1x3). Returns (fig, axes_array, saved_path_or_None)."""
        R = len(results)
        if R == 0:
            raise ValueError('results must be a non-empty sequence')

        # determine layout
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

        # hide any unused axes (if grid bigger than number of results)
        for j in range(R, r * c):
            axs[j].axis('off')

        if align_axes:
            global_tmin, global_tmax = np.inf, -np.inf
            global_ymin, global_ymax = np.inf, -np.inf

            def _upd(arr):
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
                        ax.set_ylabel('')  # remove y-label text only
                        ax.tick_params(labelleft=True)  # keep tick labels visible
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
                    # Clone visual style from the real plotted handle
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
                        alpha_pt = 0.36  # fallback to your plotting alpha

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
                    # Fallback if $X_t$ wasn't collected at all — use your plotting style
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

        # suptitle / layout
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
        legend_panel_index: int = -1,
        log_eta: bool = False,
    ):
        """Draw a grid where each subplot overlays multiple sweeps.

        panels: dict title -> {"sweeps": [SweepResults,...], "labels": [str,...]}
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

        # helper to read sweep arrays
        def _unpack(s):
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

            # collect all eta values for ticks (when log_eta=True)
            all_eta_for_ticks: list[np.ndarray] = []

            for j, (sw, lab) in enumerate(zip(sweeps, labels)):
                eta, y_u_t, y_u_s, alpha_ = _unpack(sw)
                all_eta_for_ticks.append(np.asarray(eta, dtype=float))
                mk = base_markers[j % len(base_markers)]

                if plot_both:
                    (ln1,) = ax.plot(
                        eta,
                        y_u_t,
                        marker=mk,
                        markersize=marker_size,
                        lw=line_width,
                        linestyle='-',
                        label=f'{lab}: avg-time',
                    )
                    (ln2,) = ax.plot(
                        eta,
                        y_u_s,
                        marker=mk,
                        markersize=marker_size,
                        lw=line_width,
                        linestyle='--',
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
                        eta,
                        y,
                        marker=mk,
                        markersize=marker_size,
                        lw=line_width,
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

            # x-axis scaling and ticks
            if log_eta:
                ax.set_xscale('log')
                ticks = (
                    np.unique(np.concatenate(all_eta_for_ticks))
                    if all_eta_for_ticks
                    else np.array([])
                )
                ticks = ticks[ticks > 0]
                if ticks.size:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f'{v:g}' for v in ticks])
            else:
                ax.margins(x=0)

            ax.set_xlabel(r'$\eta$', fontsize=axis_labelsize)
            ax.set_ylabel(
                single_ylabel if single_ylabel is not None else '',
                fontsize=axis_labelsize,
            )

            if not common_legend:
                ax.legend(loc='lower right', fontsize=legend_fontsize, frameon=True)

        # hide unused axes if any
        for j in range(len(keys), r * c):
            axes[j].set_visible(False)

        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles_all, labels_all):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)

        if common_legend and uniq_handles:
            if legend_in_panel:
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
                and ('lower' in str(legend_loc).lower())
            )
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
        legend_inset: Optional[str] = None,
        legend_inside_bottom_right: bool = False,
        legend_ncol: Optional[int] = None,
        legend_bbox: Optional[tuple] = (0.5, -0.06),
        save_path: Optional[Path] = None,
        show: bool = True,
        log_eta: bool = False,
    ) -> Tuple['plt.Figure', np.ndarray]:
        """Plot grid of average uniform band widths vs eta.

        panels maps panel_title -> {"sweeps": [SweepResults,...], "labels": [str,...]}.
        If a sweep lacks `avg_uniform_mean_width`, an error is raised.

        Args:
            log_eta: If True, use a log scale on the eta axis but keep the tick
                labels as the original (linear) eta values.
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

            eta_all: list[float] = []

            for j, (sw, lab) in enumerate(zip(sweeps, labels)):
                # extract eta and width series
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
                    elif 'series_uniform_mean_width' in sw:
                        arr = np.asarray(sw['series_uniform_mean_width'], dtype=float)
                        y = np.full_like(eta, float(np.nanmean(arr)), dtype=float)
                    else:
                        y = None

                if y is None:
                    raise RuntimeError(
                        'Could not find avg_uniform_mean_width for sweep '
                        f"'{lab}' in panel '{key}'. Ensure your SweepResults includes "
                        "'avg_uniform_mean_width' aligned with eta."
                    )

                # sort by eta to ensure monotone x for plotting
                idx = np.argsort(eta)
                eta = eta[idx]
                y = y[idx]

                mk = base_markers[j % len(base_markers)]
                (ln,) = ax.plot(
                    eta,
                    y,
                    marker=mk,
                    markersize=marker_size,
                    lw=line_width,
                    label=str(lab),
                )
                handles_all.append(ln)
                labels_all.append(ln.get_label())

                if log_eta:
                    eta_all.extend(eta.tolist())

            ax.set_xlabel(r'$\eta$', fontsize=axis_labelsize)
            ax.set_ylabel(single_ylabel, fontsize=axis_labelsize)

            if log_eta:
                ax.set_xscale('log')
                ticks = np.unique(np.asarray(eta_all, dtype=float))
                ticks = ticks[ticks > 0]
                if ticks.size:
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f'{v:g}' for v in ticks])
                    ax.set_xlim(ticks.min(), ticks.max())
                    ax.minorticks_off()
            else:
                ax.margins(x=0)

        # hide unused axes
        for j in range(len(keys), r * c):
            axes[j].set_visible(False)

        seen = set()
        uniq_handles = []
        uniq_labels = []
        for h, l in zip(handles_all, labels_all):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)

        if common_legend and uniq_handles:
            if legend_inside_bottom_right or legend_loc == 'inplot-br':
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
        fig.subplots_adjust(hspace=hspace, wspace=wspace, bottom=0.08)

        self._save_figure(fig, save_path, default_suffix='.pdf')

        if show:
            plt.show()

        return fig, axes.reshape(r, c)
