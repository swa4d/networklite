"""
transportsim/sweep.py  –  Pressure sweep and pump curve visualisation (v2).

All graphing that is physically part of TransportSim lives here:
  - plot_pressure_sweep       : ΔP vs flow rate (system resistance curve)
  - plot_pump_operating_curve : system curve + pump power overlay
  - plot_flow_regime_map      : Re vs flow rate, coloured by regime
  - plot_fanning_vs_darcy     : comparison of friction factor formulations
  - plot_pressure_breakdown   : bar chart of ΔP components for one operating point

All functions return a base64-encoded PNG string (data:image/png;base64,...)
and close the figure after encoding.
"""

from __future__ import annotations

import io
import base64
import sys
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_ts_dir = os.path.dirname(os.path.abspath(__file__))
if _ts_dir not in sys.path:
    sys.path.insert(0, _ts_dir)

try:
    import _transportsim_core as _core
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False

# ── Theme ─────────────────────────────────────────────────────────────────────

BG       = "#0F1117"
SURFACE  = "#1A1D27"
AMBER    = "#f0a030"
TEAL     = "#18d8b0"
BLUE     = "#4a90f5"
PINK     = "#d050c8"
RED      = "#e84040"
GREEN    = "#40c878"
TEXT_DIM = "#AAAAAA"
TEXT     = "#CCCCCC"

REGIME_COLORS = {
    "laminar"      : RED,
    "transitional" : AMBER,
    "turbulent"    : TEAL,
}


def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _styled_fig(nrows=1, ncols=1, figsize=(10, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=BG)
    ax_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]
    for ax in ax_flat:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color("#EEEEEE")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
    return fig, axes


# ── 1. System resistance / pressure sweep ────────────────────────────────────

def plot_pressure_sweep(
    pipe_name    : str,
    geom,               # PipelineSpec-like
    fluid,              # FluidProperties-like
    q_min_Ls     : float = 0.05,
    q_max_Ls     : float = 5.0,
    n_points     : int   = 60,
    current_flow_m3s: Optional[float] = None,
    pump_model   = None,  # PumpModel (optional, to mark operating point)
    node_P_inlet_Pa : float = 101325.0,
    node_P_outlet_Pa: float = 101325.0,
) -> str:
    """
    System resistance curve: ΔP (kPa) vs flow rate (L/s).

    Colours line segments by flow regime. Marks the current operating
    point if current_flow_m3s is provided.
    """
    q_min = q_min_Ls / 1000.0
    q_max = q_max_Ls / 1000.0

    if _HAS_CORE:
        g = _core.PipelineGeometry()
        g.length           = geom.length
        g.diameter         = geom.diameter
        g.roughness        = geom.roughness
        g.elevation_change = geom.elevation_change
        g.n_fittings_K     = int(geom.n_fittings_K)

        f = _core.FluidProps()
        f.density   = fluid.density
        f.viscosity = fluid.viscosity

        sweep = _core.pump_curve_sweep(
            g, q_min, q_max, n_points, f,
            node_P_inlet_Pa, node_P_outlet_Pa, 0.75
        )
        qs_Ls  = [q * 1000 for q in sweep.flow_rates]
        dps    = list(sweep.pressure_drops_kPa)
        res_ns = list(sweep.reynolds_numbers)
    else:
        # Python fallback
        qs     = np.linspace(q_min, q_max, n_points)
        qs_Ls  = (qs * 1000).tolist()
        dps    = []
        res_ns = []
        import math
        A = math.pi * (geom.diameter / 2) ** 2
        for Q in qs:
            v    = Q / A if A > 0 else 0.0
            Re   = fluid.density * v * geom.diameter / fluid.viscosity
            eps_D = geom.roughness / geom.diameter
            if Re < 2300:
                fr = 64.0 / max(Re, 1.0)
            else:
                fr = 0.02
                for _ in range(20):
                    rhs = -2.0 * math.log10(eps_D / 3.7 + 2.51 / (max(Re, 1.0) * math.sqrt(fr)))
                    fr_new = 1.0 / rhs ** 2
                    if abs(fr_new - fr) < 1e-8:
                        break
                    fr = fr_new
            dP = fr * (geom.length / geom.diameter) * 0.5 * fluid.density * v ** 2
            dps.append(dP / 1000.0)
            res_ns.append(Re)

    # Colour by regime
    fig, ax = _styled_fig(figsize=(9, 5))
    ax = ax if not isinstance(ax, np.ndarray) else ax.flat[0]

    prev_regime = None
    seg_xs, seg_ys = [], []

    def _flush(regime):
        if len(seg_xs) >= 2:
            col = REGIME_COLORS.get(regime, TEAL)
            ax.plot(seg_xs, seg_ys, color=col, lw=2)

    for i, (Q_Ls, dP, Re) in enumerate(zip(qs_Ls, dps, res_ns)):
        if Re < 2300:
            reg = "laminar"
        elif Re < 4000:
            reg = "transitional"
        else:
            reg = "turbulent"

        if reg != prev_regime and prev_regime is not None:
            seg_xs.append(Q_Ls)
            seg_ys.append(dP)
            _flush(prev_regime)
            seg_xs, seg_ys = [Q_Ls], [dP]
        else:
            seg_xs.append(Q_Ls)
            seg_ys.append(dP)
        prev_regime = reg

    _flush(prev_regime or "turbulent")

    # Legend patches
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color=RED,   lw=2, label="Laminar"),
        Line2D([0], [0], color=AMBER, lw=2, label="Transitional"),
        Line2D([0], [0], color=TEAL,  lw=2, label="Turbulent"),
    ]
    ax.legend(handles=legend_els, facecolor=SURFACE, edgecolor="#333344",
              labelcolor=TEXT, fontsize=9, loc="upper left")

    # Operating point
    if current_flow_m3s is not None and len(dps) > 0:
        idx = min(range(len(qs_Ls)),
                  key=lambda i: abs(qs_Ls[i] - current_flow_m3s * 1000))
        ax.scatter([qs_Ls[idx]], [dps[idx]], color=AMBER, s=80, zorder=5,
                   label="Operating point")
        ax.axvline(qs_Ls[idx], color=AMBER, lw=0.8, linestyle="--", alpha=0.5)
        ax.axhline(dps[idx],   color=AMBER, lw=0.8, linestyle="--", alpha=0.5)
        ax.annotate(
            f" Q={current_flow_m3s*1000:.2f} L/s\n ΔP={dps[idx]:.2f} kPa",
            xy=(qs_Ls[idx], dps[idx]),
            xytext=(qs_Ls[idx] + 0.05 * (max(qs_Ls) - min(qs_Ls)), dps[idx]),
            color=AMBER, fontsize=9, fontfamily="monospace",
        )

    ax.set_xlabel("Volumetric Flow Rate  (L/s)")
    ax.set_ylabel("Pipe ΔP  (kPa)")
    ax.set_title(f"System Resistance Curve — {pipe_name}", pad=10)
    ax.grid(True, color="#1e2535", lw=0.5)

    roughness_m = getattr(geom, "roughness", 4.6e-5)
    rough_flag = "⚠ roughness unset (steel default)" if roughness_m == 4.6e-5 else ""
    density_flag = "⚠ density unset (water default)" if fluid.density == 1000.0 else ""
    notes = "  |  ".join(x for x in [rough_flag, density_flag] if x)
    if notes:
        fig.text(0.01, 0.01, notes, color=AMBER, fontsize=8, fontfamily="monospace")

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 2. Pump operating curve (system + power) ─────────────────────────────────

def plot_pump_operating_curve(
    pipe_name        : str,
    geom,
    fluid,
    efficiency       : float = 0.75,
    q_min_Ls         : float = 0.05,
    q_max_Ls         : float = 5.0,
    n_points         : int   = 60,
    current_flow_m3s : Optional[float] = None,
    node_P_inlet_Pa  : float = 101325.0,
    node_P_outlet_Pa : float = 101325.0,
) -> str:
    """
    Two-panel plot: system resistance curve (top) and pump power (bottom).
    Both share the x-axis (flow rate L/s).
    """
    q_min = q_min_Ls / 1000.0
    q_max = q_max_Ls / 1000.0

    if _HAS_CORE:
        g = _core.PipelineGeometry()
        g.length           = geom.length
        g.diameter         = geom.diameter
        g.roughness        = geom.roughness
        g.elevation_change = geom.elevation_change
        g.n_fittings_K     = int(geom.n_fittings_K)
        f = _core.FluidProps()
        f.density   = fluid.density
        f.viscosity = fluid.viscosity

        sweep = _core.pump_curve_sweep(
            g, q_min, q_max, n_points, f,
            node_P_inlet_Pa, node_P_outlet_Pa, efficiency
        )
        qs_Ls    = [q * 1000 for q in sweep.flow_rates]
        dps      = list(sweep.pump_delta_p_kPa)
        powers   = list(sweep.pump_power_kW)
        turbulent= list(sweep.flow_is_turbulent)
    else:
        qs_Ls  = list(np.linspace(q_min_Ls, q_max_Ls, n_points))
        dps    = [0.0] * n_points
        powers = [0.0] * n_points
        turbulent = [True] * n_points

    fig = plt.figure(figsize=(9, 7), facecolor=BG)
    gs  = gridspec.GridSpec(2, 1, hspace=0.35, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    for ax in [ax1, ax2]:
        ax.set_facecolor(SURFACE)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color("#EEEEEE")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333344")
        ax.grid(True, color="#1e2535", lw=0.5)

    ax1.plot(qs_Ls, dps, color=TEAL, lw=2, label="Pump ΔP required")
    ax2.plot(qs_Ls, powers, color=AMBER, lw=2, label="Shaft power")

    if current_flow_m3s is not None:
        for ax, vals, col, unit in [
            (ax1, dps, PINK, " kPa"),
            (ax2, powers, PINK, " kW"),
        ]:
            idx = min(range(len(qs_Ls)),
                      key=lambda i: abs(qs_Ls[i] - current_flow_m3s * 1000))
            ax.scatter([qs_Ls[idx]], [vals[idx]], color=PINK, s=80, zorder=5)
            ax.axvline(qs_Ls[idx], color=PINK, lw=0.8, linestyle="--", alpha=0.5)

    ax1.set_ylabel("Pump ΔP  (kPa)")
    ax1.set_title(f"Pump Operating Curve — {pipe_name}  (η={efficiency:.0%})", pad=8)
    ax2.set_xlabel("Volumetric Flow Rate  (L/s)")
    ax2.set_ylabel("Shaft Power  (kW)")

    for ax in [ax1, ax2]:
        ax.legend(facecolor=SURFACE, edgecolor="#333344", labelcolor=TEXT,
                  fontsize=9, loc="upper left")

    return _fig_to_b64(fig)


# ── 3. Flow regime map ───────────────────────────────────────────────────────

def plot_flow_regime_map(
    pipe_name : str,
    geom,
    fluid,
    q_min_Ls  : float = 0.01,
    q_max_Ls  : float = 5.0,
    n_points  : int   = 80,
) -> str:
    """
    Reynolds number vs flow rate, with regime band shading.
    """
    qs_Ls = np.linspace(q_min_Ls, q_max_Ls, n_points)
    qs    = qs_Ls / 1000.0
    import math
    A  = math.pi * (geom.diameter / 2) ** 2
    Re = [fluid.density * (Q / A) * geom.diameter / fluid.viscosity
          if A > 0 else 0.0 for Q in qs]

    fig, ax = _styled_fig(figsize=(9, 4))
    ax = ax if not isinstance(ax, np.ndarray) else ax.flat[0]

    ax.plot(qs_Ls, Re, color=TEAL, lw=2, label="Reynolds number")
    ax.axhspan(0,    2300, facecolor=RED,   alpha=0.12, label="Laminar (Re < 2300)")
    ax.axhspan(2300, 4000, facecolor=AMBER, alpha=0.12, label="Transitional (2300–4000)")
    ax.axhspan(4000, max(max(Re) * 1.1, 10000), facecolor=TEAL, alpha=0.07,
               label="Turbulent (Re ≥ 4000)")
    ax.axhline(2300, color=RED,   lw=0.8, linestyle="--")
    ax.axhline(4000, color=AMBER, lw=0.8, linestyle="--")

    ax.set_xlabel("Volumetric Flow Rate  (L/s)")
    ax.set_ylabel("Reynolds Number")
    ax.set_title(f"Flow Regime Map — {pipe_name}", pad=8)
    ax.legend(facecolor=SURFACE, edgecolor="#333344", labelcolor=TEXT,
              fontsize=9, loc="lower right")
    ax.grid(True, color="#1e2535", lw=0.5)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 4. Fanning vs Darcy comparison ──────────────────────────────────────────

def plot_fanning_vs_darcy(
    pipe_name : str,
    geom,
    fluid,
    q_min_Ls  : float = 0.05,
    q_max_Ls  : float = 5.0,
    n_points  : int   = 60,
) -> str:
    """
    Plot Darcy f and Fanning f (= Darcy/4) over a range of flow rates.

    Educational: shows engineers the relationship between the two
    commonly used friction factor conventions.
    """
    qs_Ls = np.linspace(q_min_Ls, q_max_Ls, n_points)
    qs    = qs_Ls / 1000.0
    import math
    A     = math.pi * (geom.diameter / 2) ** 2
    eps_D = geom.roughness / geom.diameter

    darcy_fs   = []
    fanning_fs = []

    for Q in qs:
        v    = Q / A if A > 0 else 0.0
        Re   = fluid.density * v * geom.diameter / fluid.viscosity
        Re   = max(Re, 1.0)
        if Re < 2300:
            f = 64.0 / Re
        else:
            f = 0.02
            for _ in range(25):
                rhs = -2.0 * math.log10(eps_D / 3.7 + 2.51 / (Re * math.sqrt(f)))
                f_new = 1.0 / rhs ** 2
                if abs(f_new - f) < 1e-8:
                    break
                f = f_new
        darcy_fs.append(f)
        fanning_fs.append(f / 4.0)

    fig, ax = _styled_fig(figsize=(9, 4))
    ax = ax if not isinstance(ax, np.ndarray) else ax.flat[0]

    ax.plot(qs_Ls, darcy_fs,   color=TEAL,  lw=2, label="Darcy-Weisbach  f_D")
    ax.plot(qs_Ls, fanning_fs, color=AMBER, lw=2, label="Fanning  f_F = f_D / 4",
            linestyle="--")

    ax.set_xlabel("Volumetric Flow Rate  (L/s)")
    ax.set_ylabel("Friction Factor")
    ax.set_title(f"Darcy vs Fanning Friction Factor — {pipe_name}", pad=8)
    ax.legend(facecolor=SURFACE, edgecolor="#333344", labelcolor=TEXT,
              fontsize=9, loc="upper right")
    ax.grid(True, color="#1e2535", lw=0.5)

    note = ("Note: ΔP = f_D·(L/D)·(ρv²/2) = 4·f_F·(L/D)·(ρv²/2)  "
            "— both give the same total pressure drop.")
    fig.text(0.01, 0.01, note, color=TEXT_DIM, fontsize=8, fontfamily="monospace")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 5. Pressure breakdown bar chart ─────────────────────────────────────────

def plot_pressure_breakdown(
    pipe_name     : str,
    pump_state,   # PumpState
    hydraulic_result = None,  # optional HydraulicResult-like for component detail
) -> str:
    """
    Stacked bar: friction ΔP | minor losses | gravity | node correction.
    Displays pump ΔP required and shaft power clearly.
    """
    if pump_state is None:
        fig, ax = _styled_fig(figsize=(6, 3))
        ax = ax if not isinstance(ax, np.ndarray) else ax.flat[0]
        ax.text(0.5, 0.5, "No pump data — run simulation first.",
                ha="center", va="center", color=TEXT_DIM,
                fontfamily="monospace", fontsize=11, transform=ax.transAxes)
        ax.set_title(f"Pressure Breakdown — {pipe_name}", pad=8)
        return _fig_to_b64(fig)

    # Values from pump state
    total_kPa = pump_state.required_delta_p_kPa
    pipe_kPa  = pump_state.pressure_drop_kPa   # pipe losses only

    fig, axes = _styled_fig(1, 2, figsize=(9, 4))
    ax1, ax2 = axes

    # Bar 1: ΔP breakdown
    categories = ["Pipe losses", "Node ΔP\ncorrection"]
    node_corr  = max(total_kPa - pipe_kPa, 0.0)
    values     = [pipe_kPa, node_corr]
    colors_bar = [TEAL, BLUE]
    bars = ax1.bar(categories, values, color=colors_bar, width=0.45,
                   edgecolor="#333344")
    for bar, val in zip(bars, values):
        if val > 0.001:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * total_kPa,
                     f"{val:.2f}", ha="center", va="bottom", color=TEXT,
                     fontsize=10, fontfamily="monospace")
    ax1.set_ylabel("ΔP  (kPa)")
    ax1.set_title(f"Pump ΔP = {total_kPa:.2f} kPa", pad=6)

    # Bar 2: Power / efficiency
    input_power = pump_state.shaft_power_kW / pump_state.efficiency if pump_state.efficiency > 0 else 0
    hydraulic_p = pump_state.shaft_power_kW
    heat_loss   = input_power - hydraulic_p
    p_cats  = ["Hydraulic\npower", "Heat loss\n(1-η)"]
    p_vals  = [hydraulic_p, heat_loss]
    p_cols  = [AMBER, PINK]
    pbars   = ax2.bar(p_cats, p_vals, color=p_cols, width=0.45, edgecolor="#333344")
    for bar, val in zip(pbars, p_vals):
        if val > 0.0001:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(input_power, 0.001),
                     f"{val:.3f}", ha="center", va="bottom", color=TEXT,
                     fontsize=10, fontfamily="monospace")
    ax2.set_ylabel("Power  (kW)")
    ax2.set_title(f"Shaft Power = {hydraulic_p:.4f} kW  (η={pump_state.efficiency:.0%})", pad=6)

    for ax in [ax1, ax2]:
        ax.grid(True, axis="y", color="#1e2535", lw=0.5)

    fig.suptitle(f"Pressure & Power Breakdown — {pipe_name}",
                 color="#EEEEEE", fontsize=11, y=1.02)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 6. Power vs time chart ───────────────────────────────────────────────────

def plot_pump_power_over_time(
    pipe_time_series_dict: dict,  # {pipe_name: PumpTimeSeries}
) -> str:
    """
    Multi-pipe pump power vs simulation time.
    """
    PALETTE = [AMBER, TEAL, BLUE, PINK, GREEN, RED]

    fig, ax = _styled_fig(figsize=(10, 4))
    ax = ax if not isinstance(ax, np.ndarray) else ax.flat[0]

    for i, (name, ts) in enumerate(pipe_time_series_dict.items()):
        col = PALETTE[i % len(PALETTE)]
        ax.plot(ts.times, ts.power_kW, color=col, lw=2,
                label=f"{name}  (mean {ts.mean_power_kW:.3f} kW)")

    ax.set_xlabel("Simulation Time  (s)")
    ax.set_ylabel("Pump Shaft Power  (kW)")
    ax.set_title("Pump Power Demand vs Time", pad=8)
    ax.legend(facecolor=SURFACE, edgecolor="#333344", labelcolor=TEXT,
              fontsize=9, loc="upper right")
    ax.grid(True, color="#1e2535", lw=0.5)

    fig.tight_layout()
    return _fig_to_b64(fig)
