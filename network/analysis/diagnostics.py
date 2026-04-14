from __future__ import annotations
import sys as _sys
for _p in ["/mnt/project", "/home/claude/networklite"]:
    if _p not in _sys.path: _sys.path.insert(0, _p)
del _sys
"""
network/analysis/diagnostics.py  –  Plant-level analysis and diagnostics.

Generates matplotlib figures for the dashboard: concentration trajectories,
material balances, Pareto fronts, temperature profiles, pipeline curves.
"""


import io
import base64
import sys, os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

_net_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _net_dir not in sys.path:
    sys.path.insert(0, _net_dir)

from network.plant import PlantNetwork, NetworkSimulationResult
from network.optimizer.multi_objective import OptimizationResult


# ── Colour palette ────────────────────────────────────────────────────────────

PALETTE = [
    "#4C9BE8", "#E85C4C", "#4CE87A", "#E8C34C", "#C34CE8",
    "#4CE8D4", "#E8844C", "#7A4CE8", "#E84CAF", "#84E84C",
]

def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def _styled_fig(nrows=1, ncols=1, figsize=(10, 5)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor="#0F1117")
    for ax in (np.array(axes).flatten() if nrows * ncols > 1 else [axes]):
        ax.set_facecolor("#1A1D27")
        ax.tick_params(colors="#AAAAAA", labelsize=9)
        ax.xaxis.label.set_color("#CCCCCC")
        ax.yaxis.label.set_color("#CCCCCC")
        ax.title.set_color("#EEEEEE")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333344")
    return fig, axes


# ── 1. Concentration trajectories ─────────────────────────────────────────────

def plot_reactor_trajectories(
    result: NetworkSimulationResult,
    reactor_names: Optional[List[str]] = None,
) -> Dict[str, str]:
    """
    Plot concentration vs time for each reactor.

    Returns dict: reactor_name → base64 PNG.
    """
    plots: Dict[str, str] = {}
    names = reactor_names or list(result.reactor_results.keys())

    for rname in names:
        rr = result.reactor_results.get(rname)
        if rr is None:
            continue
        # Skip SourceSink nodes — they have constant composition, no trajectory plot needed
        if not hasattr(rr, 'temperature_profile'):
            continue

        n_sp = len(rr.species_names)
        fig, axes = _styled_fig(1, 2, figsize=(12, 4))
        ax_conc, ax_T = axes

        # Concentration trajectories
        for i, sp in enumerate(rr.species_names):
            col = PALETTE[i % len(PALETTE)]
            ax_conc.plot(rr.time, rr.concentrations[:, i],
                         color=col, lw=1.8, label=sp)

        ax_conc.set_xlabel("Time (s)")
        ax_conc.set_ylabel("Concentration (M)")
        ax_conc.set_title(f"{rname} — Species Concentrations")
        ax_conc.legend(fontsize=8, framealpha=0.3,
                       facecolor="#1A1D27", labelcolor="#CCCCCC")
        ax_conc.grid(alpha=0.15)

        # Temperature profile
        ax_T.plot(rr.time, rr.temperature_profile - 273.15,
                  color="#E8C34C", lw=2)
        ax_T.set_xlabel("Time (s)")
        ax_T.set_ylabel("Temperature (°C)")
        ax_T.set_title(f"{rname} — Temperature Profile")
        ax_T.grid(alpha=0.15)

        fig.tight_layout()
        plots[rname] = _fig_to_b64(fig)

    return plots


# ── 2. Material balance bar chart ─────────────────────────────────────────────

def plot_material_balance(result: NetworkSimulationResult) -> str:
    """Stacked bar chart of inlet vs outlet molar flows per species."""
    bal = result.material_balance
    if not bal:
        return ""

    species = list(bal.keys())
    inflows  = [bal[s]["in_mol_s"]  * 1000.0 for s in species]   # mmol/s
    outflows = [bal[s]["out_mol_s"] * 1000.0 for s in species]

    x = np.arange(len(species))
    w = 0.35

    fig, ax = _styled_fig(1, 1, figsize=(max(7, len(species) * 1.2), 5))
    ax = ax

    bars_in  = ax.bar(x - w/2, inflows,  w, label="Inlet",  color="#4C9BE8", alpha=0.9)
    bars_out = ax.bar(x + w/2, outflows, w, label="Outlet", color="#4CE87A", alpha=0.9)

    # Closure annotation
    for i, sp in enumerate(species):
        cl = bal[sp]["closure_pct"]
        color = "#4CE87A" if abs(cl - 100) < 5 else "#E85C4C"
        ax.text(i, max(inflows[i], outflows[i]) * 1.05,
                f"{cl:.0f}%", ha="center", va="bottom",
                fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(species, rotation=30, ha="right")
    ax.set_ylabel("Molar Flow (mmol/s)")
    ax.set_title("Plant Material Balance — Inlet vs Outlet")
    ax.legend(framealpha=0.3, facecolor="#1A1D27", labelcolor="#CCCCCC")
    ax.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 3. Conversion heatmap ─────────────────────────────────────────────────────

def plot_conversion_heatmap(result: NetworkSimulationResult) -> str:
    """Heatmap: reactors × species conversion."""
    reactors = list(result.reactor_results.keys())
    if not reactors:
        return ""

    # Collect all species
    all_species: List[str] = []
    for rr in result.reactor_results.values():
        for sp in rr.conversion:
            if sp not in all_species:
                all_species.append(sp)

    if not all_species:
        return ""

    matrix = np.zeros((len(reactors), len(all_species)))
    for i, rname in enumerate(reactors):
        rr = result.reactor_results[rname]
        for j, sp in enumerate(all_species):
            matrix[i, j] = rr.conversion.get(sp, 0.0) * 100.0

    fig, ax = _styled_fig(1, 1, figsize=(max(6, len(all_species) * 1.0),
                                         max(3, len(reactors) * 0.8)))

    cmap = LinearSegmentedColormap.from_list(
        "chem", ["#1A1D27", "#4C9BE8", "#4CE87A"]
    )
    im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    plt.colorbar(im, ax=ax, label="Conversion (%)")

    ax.set_xticks(range(len(all_species)))
    ax.set_yticks(range(len(reactors)))
    ax.set_xticklabels(all_species, rotation=45, ha="right")
    ax.set_yticklabels(reactors)
    ax.set_title("Species Conversion by Reactor (%)")

    # Annotate cells
    for i in range(len(reactors)):
        for j in range(len(all_species)):
            v = matrix[i, j]
            ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                    fontsize=9,
                    color="white" if v < 60 else "#0F1117")

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 4. Pipeline pressure profile ──────────────────────────────────────────────

def plot_pipeline_pressures(result: NetworkSimulationResult) -> str:
    """Bar chart of pressure drops across each pipeline."""
    diags = result.connection_diags
    if not diags:
        return ""

    names = list(diags.keys())
    dP    = [diags[n].pressure_drop_kPa for n in names]
    needs = [diags[n].needs_compressor   for n in names]

    colors = ["#E85C4C" if n else "#4C9BE8" for n in needs]

    fig, axes = _styled_fig(1, 2, figsize=(12, 4))
    ax_dP, ax_Re = axes

    bars = ax_dP.bar(names, dP, color=colors, alpha=0.85)
    ax_dP.set_xlabel("Pipeline")
    ax_dP.set_ylabel("Pressure Drop (kPa)")
    ax_dP.set_title("Pipeline Pressure Drops")
    ax_dP.tick_params(axis="x", rotation=30)
    ax_dP.grid(axis="y", alpha=0.15)

    # Add compressor annotations
    for i, (bar, nc) in enumerate(zip(bars, needs)):
        if nc:
            ax_dP.text(bar.get_x() + bar.get_width() / 2,
                       bar.get_height() * 1.02, "⚠ comp",
                       ha="center", va="bottom", fontsize=8, color="#E8C34C")

    # Reynolds number chart
    Re_vals = [diags[n].reynolds_number for n in names]
    bar_colors_Re = []
    for n in names:
        r = diags[n].flow_regime
        bar_colors_Re.append(
            "#4C9BE8" if r == "turbulent" else
            "#E8C34C" if r == "transitional" else "#E85C4C"
        )

    ax_Re.bar(names, Re_vals, color=bar_colors_Re, alpha=0.85)
    ax_Re.axhline(4000,  color="#E8C34C", lw=1, linestyle="--", label="Turbulent (4000)")
    ax_Re.axhline(2300,  color="#E85C4C", lw=1, linestyle="--", label="Laminar (2300)")
    ax_Re.set_xlabel("Pipeline")
    ax_Re.set_ylabel("Reynolds Number")
    ax_Re.set_title("Flow Regime by Pipeline")
    ax_Re.tick_params(axis="x", rotation=30)
    ax_Re.legend(fontsize=8, framealpha=0.3, facecolor="#1A1D27", labelcolor="#CCCCCC")
    ax_Re.grid(axis="y", alpha=0.15)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 5. Pipeline flow sweep curve ──────────────────────────────────────────────

def plot_pipeline_sweep(plant: PlantNetwork, connection_name: str) -> str:
    """Pressure-drop vs flow rate curve for a pipeline."""
    conn = plant.get_connection(connection_name)
    sweep = conn.flow_sweep(
        q_min=1e-6,
        q_max=max(conn.pipeline.flow_rate_m3s * 5, 1e-3),
        n_points=60,
    )

    Q_Ls  = [q * 1000 for q in sweep["flow_rates"]]
    dP_kPa = [dp / 1000 for dp in sweep["pressure_drops"]]
    Re     = sweep["reynolds_numbers"]
    nc     = sweep["needs_compressor"]

    fig, axes = _styled_fig(1, 2, figsize=(12, 4))
    ax_dP, ax_Re = axes

    # Colour segments by compressor need
    for i in range(len(Q_Ls) - 1):
        col = "#E85C4C" if nc[i] else "#4C9BE8"
        ax_dP.plot(Q_Ls[i:i+2], dP_kPa[i:i+2], color=col, lw=2)

    current_q = conn.pipeline.flow_rate_m3s * 1000
    current_dp = None
    if sweep["flow_rates"]:
        idx = min(range(len(sweep["flow_rates"])),
                  key=lambda j: abs(sweep["flow_rates"][j] - conn.pipeline.flow_rate_m3s))
        current_dp = sweep["pressure_drops"][idx] / 1000

    if current_dp is not None:
        ax_dP.axvline(current_q, color="#E8C34C", lw=1.5, linestyle="--",
                      label=f"Current Q = {current_q:.2f} L/s")
        ax_dP.axhline(current_dp, color="#E8C34C", lw=1, linestyle=":",
                      alpha=0.6)

    ax_dP.set_xlabel("Flow Rate (L/s)")
    ax_dP.set_ylabel("Pressure Drop (kPa)")
    ax_dP.set_title(f"{connection_name} — ΔP vs Flow Rate")
    ax_dP.legend(fontsize=8, framealpha=0.3, facecolor="#1A1D27", labelcolor="#CCCCCC")
    ax_dP.grid(alpha=0.15)

    # Reynolds plot
    for i in range(len(Q_Ls) - 1):
        col = "#4C9BE8" if Re[i] > 4000 else ("#E8C34C" if Re[i] > 2300 else "#E85C4C")
        ax_Re.plot(Q_Ls[i:i+2], [Re[i], Re[i+1]], color=col, lw=2)

    ax_Re.axhline(4000, color="#E8C34C", lw=1, linestyle="--")
    ax_Re.axhline(2300, color="#E85C4C", lw=1, linestyle="--")
    ax_Re.set_xlabel("Flow Rate (L/s)")
    ax_Re.set_ylabel("Reynolds Number")
    ax_Re.set_title(f"{connection_name} — Reynolds Number vs Flow Rate")
    ax_Re.grid(alpha=0.15)

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 6. Pareto front ───────────────────────────────────────────────────────────

def plot_pareto_front(opt_result: OptimizationResult) -> str:
    """
    Plot the Pareto front for 2-objective problems.
    For 3+ objectives, shows parallel coordinates.
    """
    if opt_result.pareto_F is None or len(opt_result.pareto_F) == 0:
        return ""

    F    = opt_result.pareto_F
    objs = opt_result.objectives
    n_obj = F.shape[1]

    if n_obj == 2:
        fig, ax = _styled_fig(1, 1, figsize=(8, 6))

        scatter = ax.scatter(F[:, 0], F[:, 1],
                             c=np.linalg.norm(F, axis=1),
                             cmap="plasma", s=50, alpha=0.85, edgecolors="#444")
        plt.colorbar(scatter, ax=ax, label="Distance from origin")

        # Highlight best (optimal_objectives)
        if opt_result.optimal_objectives:
            ax.scatter(
                [opt_result.optimal_objectives[0]],
                [opt_result.optimal_objectives[1]],
                color="#E8C34C", s=150, zorder=5, marker="*",
                label="Best (weighted)"
            )
            ax.legend(fontsize=9, framealpha=0.3,
                      facecolor="#1A1D27", labelcolor="#CCCCCC")

        ax.set_xlabel(objs[0])
        ax.set_ylabel(objs[1])
        ax.set_title(f"Pareto Front ({opt_result.method})")
        ax.grid(alpha=0.15)

    elif n_obj >= 3:
        # Parallel coordinates
        fig, ax = _styled_fig(1, 1, figsize=(10, 5))
        x_ticks = np.arange(n_obj)

        # Normalise F to [0, 1]
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_norm = (F - F_min) / np.where(F_max > F_min, F_max - F_min, 1.0)

        cmap = plt.get_cmap("plasma")
        scores = F_norm.mean(axis=1)
        for i, row in enumerate(F_norm):
            col = cmap(scores[i])
            ax.plot(x_ticks, row, color=col, lw=0.9, alpha=0.6)

        ax.set_xticks(x_ticks)
        ax.set_xticklabels(objs, rotation=30, ha="right")
        ax.set_ylabel("Normalised objective value")
        ax.set_title(f"Pareto Front — Parallel Coordinates ({opt_result.method})")
        ax.grid(alpha=0.15)
    else:
        return ""

    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 7. Optimisation convergence ───────────────────────────────────────────────

def plot_convergence(opt_result: OptimizationResult) -> str:
    """Plot optimizer convergence history."""
    hist = opt_result.convergence_history
    if not hist:
        return ""

    fig, ax = _styled_fig(1, 1, figsize=(9, 4))
    ax.plot(hist, color="#4C9BE8", lw=1.5)
    ax.set_xlabel("Evaluation #")
    ax.set_ylabel("Objective value (internal)")
    ax.set_title(f"Optimisation Convergence — {opt_result.method}")
    ax.grid(alpha=0.15)
    # Rolling minimum
    running_min = [min(hist[:i+1]) for i in range(len(hist))]
    ax.plot(running_min, color="#4CE87A", lw=1.5, linestyle="--", label="Running min")
    ax.legend(fontsize=9, framealpha=0.3, facecolor="#1A1D27", labelcolor="#CCCCCC")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── 8. Network topology graph ─────────────────────────────────────────────────

def plot_network_topology(plant: PlantNetwork,
                          result: Optional[NetworkSimulationResult] = None) -> str:
    """
    Simple reactor network graph using matplotlib arrows.
    """
    topo = plant.topology()
    nodes = topo["nodes"]
    edges = topo["edges"]

    if not nodes:
        return ""

    fig, ax = _styled_fig(1, 1, figsize=(max(8, len(nodes) * 2.5), 5))

    # Layout: reactors evenly spaced horizontally
    n = len(nodes)
    positions: Dict[str, Tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        x = i / max(n - 1, 1) * 8
        y = 0.5 + 0.3 * np.sin(i * np.pi / max(n - 1, 1))
        positions[node["id"]] = (x, y)

    # Draw edges
    for edge in edges:
        if edge["source"] in positions and edge["target"] in positions:
            x0, y0 = positions[edge["source"]]
            x1, y1 = positions[edge["target"]]
            color = "#E85C4C" if edge.get("needs_compressor") else "#4C9BE8"
            ax.annotate(
                "",
                xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2, mutation_scale=15),
            )
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dP_text = ""
            if edge.get("pressure_drop_kPa") is not None:
                dP_text = f"\n{edge['pressure_drop_kPa']:.1f} kPa"
            ax.text(mx, my + 0.08,
                    f"{edge['flow_Ls']:.2f} L/s{dP_text}",
                    ha="center", fontsize=7, color="#AAAAAA")

    # Draw nodes
    for node in nodes:
        x, y = positions[node["id"]]
        conv_ok = node.get("converged")
        face = "#2A3A5A" if conv_ok else ("#4A2A2A" if conv_ok is False else "#2A2A3A")
        circle = plt.Circle((x, y), 0.22, color=face, ec="#4C9BE8", lw=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y + 0.05, node["label"], ha="center", va="center",
                fontsize=9, color="white", fontweight="bold", zorder=4)
        ax.text(x, y - 0.1,
                f"V={node['volume_L']}L\nτ={node['tau_s']:.0f}s",
                ha="center", va="center", fontsize=6.5, color="#AAAAAA", zorder=4)

    ax.set_xlim(-0.8, 8.8)
    ax.set_ylim(-0.5, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Plant Network Topology")
    fig.tight_layout()
    return _fig_to_b64(fig)


# ── All-in-one dashboard payload ──────────────────────────────────────────────

def generate_all_plots(
    plant : PlantNetwork,
    result: NetworkSimulationResult,
    opt_result: Optional[OptimizationResult] = None,
) -> dict:
    """
    Generate all analysis plots for the dashboard.
    Returns a dict of name → base64 PNG string.
    """
    plots: dict = {}

    try:
        plots["topology"] = plot_network_topology(plant, result)
    except Exception as e:
        plots["topology"] = ""

    try:
        traj = plot_reactor_trajectories(result)
        plots.update({f"trajectory_{k}": v for k, v in traj.items()})
    except Exception:
        pass

    try:
        plots["material_balance"] = plot_material_balance(result)
    except Exception:
        plots["material_balance"] = ""

    try:
        plots["conversion_heatmap"] = plot_conversion_heatmap(result)
    except Exception:
        plots["conversion_heatmap"] = ""

    try:
        plots["pipeline_pressures"] = plot_pipeline_pressures(result)
    except Exception:
        plots["pipeline_pressures"] = ""

    for conn_name in plant.connections:
        try:
            plots[f"sweep_{conn_name}"] = plot_pipeline_sweep(plant, conn_name)
        except Exception:
            pass

    if opt_result is not None:
        try:
            plots["pareto_front"] = plot_pareto_front(opt_result)
        except Exception:
            plots["pareto_front"] = ""
        try:
            plots["convergence"] = plot_convergence(opt_result)
        except Exception:
            plots["convergence"] = ""

    return plots
