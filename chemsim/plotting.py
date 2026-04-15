"""
plotting.py – Matplotlib-based 2D visualization.

All functions return matplotlib Figure objects so callers can save,
embed in notebooks, or further customize.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    import matplotlib.figure
    from chemsim.simulator import SimulationResult


# ─── Publication style defaults ───────────────────────────────────────────────

_STYLE = {
    "font.family":       "serif",
    "font.size":         12,
    "axes.labelsize":    13,
    "axes.titlesize":    14,
    "legend.fontsize":   11,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "lines.linewidth":   2.0,
    "figure.dpi":        100,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
}

# Colorblind-friendly palette (10 colors)
_COLORS = [
    "#0072B2",  # blue
    "#E69F00",  # amber
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
    "#999999",  # gray
    "#AA4499",  # purple
]


def _apply_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)


# ─── Trajectory plot ──────────────────────────────────────────────────────────

def plot_trajectories(
    result: "SimulationResult",
    species: Optional[Sequence[str]] = None,
    title: str = "Species Concentrations",
    figsize: Tuple[float, float] = (10, 5),
    log_scale: bool = False,
    normalize: bool = False,
    show_events: bool = True,
    save_path: Optional[str] = None,
    ax=None,
) -> "matplotlib.figure.Figure":
    """
    Plot species concentration trajectories over time.

    Parameters
    ----------
    result : SimulationResult
    species : list of str, optional
        Species to plot (default: all).
    title : str
        Plot title.
    figsize : tuple
    log_scale : bool
        Use log y-axis.
    normalize : bool
        Normalize each species by its maximum value.
    show_events : bool
        Highlight time points where concentration is near zero.
    save_path : str, optional
        Save figure to this path (300 DPI).
    ax : matplotlib Axes, optional
        Plot into an existing axes.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    with mpl.rc_context(_STYLE):
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        _apply_style(ax)

        plot_names = list(species) if species else result.species_names
        t = result.time

        for i, name in enumerate(plot_names):
            y = result.species(name)
            if normalize:
                ymax = y.max()
                y = y / ymax if ymax > 1e-15 else y
            color = _COLORS[i % len(_COLORS)]
            ax.plot(t, y, color=color, label=name, linewidth=2.0)

        ax.set_xlabel("Time")
        y_label = "Concentration (normalized)" if normalize else "Concentration"
        ax.set_ylabel(y_label)
        ax.set_title(title)

        if log_scale:
            ax.set_yscale("log")

        ax.set_xlim(t[0], t[-1])

        if len(plot_names) <= 12:
            ax.legend(loc="best", framealpha=0.9)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ─── Multi-panel plot ─────────────────────────────────────────────────────────

def plot_trajectories_panel(
    result: "SimulationResult",
    species: Optional[Sequence[str]] = None,
    title: str = "Species Concentrations",
    ncols: int = 2,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot each species in its own subplot panel.

    Useful for networks where species have very different concentration scales.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    plot_names = list(species) if species else result.species_names
    n = len(plot_names)
    nrows = (n + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 3.5 * nrows)

    with mpl.rc_context(_STYLE):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        t = result.time
        for i, name in enumerate(plot_names):
            ax = axes_flat[i]
            _apply_style(ax)
            y = result.species(name)
            ax.plot(t, y, color=_COLORS[i % len(_COLORS)], linewidth=2.0)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.set_xlim(t[0], t[-1])

        # Hide unused subplots
        for j in range(n, len(axes_flat)):
            axes_flat[j].set_visible(False)

        fig.suptitle(title, fontsize=14, y=1.02)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ─── Phase portrait ───────────────────────────────────────────────────────────

def plot_phase_portrait(
    result: "SimulationResult",
    x_species: str,
    y_species: str,
    color_by_time: bool = True,
    show_quiver: bool = False,
    figsize: Tuple[float, float] = (6, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot a 2-D phase portrait of x_species vs y_species.

    Parameters
    ----------
    result : SimulationResult
    x_species, y_species : str
        Species to use as X and Y axes.
    color_by_time : bool
        Color the trajectory by simulation time.
    show_quiver : bool
        Overlay velocity arrows.
    figsize : tuple
    title : str, optional
    save_path : str, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from chemsim.analysis import phase_portrait as _pp

    data = _pp(result, x_species, y_species, velocity=show_quiver)
    x    = data["x"]
    y    = data["y"]
    t    = data["t"]

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(ax)

        if color_by_time:
            sc = ax.scatter(x, y, c=t, cmap="viridis", s=3, zorder=2)
            cbar = fig.colorbar(sc, ax=ax, label="Time")
        else:
            ax.plot(x, y, color=_COLORS[0], linewidth=1.5)

        # Mark start and end
        ax.plot(x[0],  y[0],  "go", ms=8, label="Start", zorder=5)
        ax.plot(x[-1], y[-1], "rs", ms=8, label="End",   zorder=5)

        if show_quiver and "dx" in data:
            step = max(1, len(x) // 30)
            ax.quiver(x[::step], y[::step],
                      data["dx"][::step], data["dy"][::step],
                      alpha=0.5, scale_units="xy", angles="xy", scale=None)

        ax.set_xlabel(x_species)
        ax.set_ylabel(y_species)
        ax.set_title(title or f"Phase Portrait: {x_species} vs {y_species}")
        ax.legend(loc="best", fontsize=10)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ─── Parameter sweep plot ─────────────────────────────────────────────────────

def plot_parameter_sweep(
    results: List["SimulationResult"],
    species_name: str,
    parameter_values: Sequence[float],
    parameter_label: str = "Parameter",
    figsize: Tuple[float, float] = (10, 5),
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Plot trajectory families from a parameter sweep, one trace per value.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.cm as cm

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(ax)

        cmap = cm.get_cmap("plasma", len(results))
        for i, (res, val) in enumerate(zip(results, parameter_values)):
            y = res.species(species_name)
            ax.plot(res.time, y, color=cmap(i), label=f"{parameter_label}={val:.3g}",
                    linewidth=1.5, alpha=0.85)

        ax.set_xlabel("Time")
        ax.set_ylabel(f"[{species_name}]")
        ax.set_title(title or f"Parameter Sweep – {species_name}")

        if len(results) <= 10:
            ax.legend(loc="best", fontsize=9)
        else:
            # Colorbar instead of legend
            sm = plt.cm.ScalarMappable(
                cmap="plasma",
                norm=plt.Normalize(min(parameter_values), max(parameter_values))
            )
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label=parameter_label)

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


# ─── Steady-state bar plot ────────────────────────────────────────────────────

def plot_steady_state(
    steady_state: Dict[str, float],
    title: str = "Steady-State Concentrations",
    figsize: Tuple[float, float] = (8, 4),
    log_scale: bool = False,
    save_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """Bar plot of detected steady-state concentrations."""
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    names  = list(steady_state.keys())
    values = list(steady_state.values())

    with mpl.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        _apply_style(ax)

        bars = ax.bar(names, values,
                      color=[_COLORS[i % len(_COLORS)] for i in range(len(names))],
                      edgecolor="white", linewidth=0.8, alpha=0.9)

        ax.set_xlabel("Species")
        ax.set_ylabel("Concentration")
        ax.set_title(title)

        if log_scale:
            ax.set_yscale("log")

        # Value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.3g}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=10,
            )

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
