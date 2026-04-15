"""
analysis.py – Post-simulation statistical analysis.

All functions operate on SimulationResult objects and return plain
Python dicts or numpy arrays (no plotting here; see plotting.py).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
from scipy import signal, optimize

if TYPE_CHECKING:
    from chemsim.simulator import SimulationResult


# ─── Steady-state analysis ────────────────────────────────────────────────────

def steady_state_analysis(
    result: "SimulationResult",
    window: float = 0.05,
    tol: float = 1e-3,
) -> Optional[Dict[str, float]]:
    """
    Detect whether the system has reached steady state.

    Steady state is declared when, over the last `window` fraction of the
    simulation time, the relative change in all species concentrations stays
    below `tol`.

    Parameters
    ----------
    result : SimulationResult
    window : float
        Fraction of total time to examine (default: last 5%).
    tol : float
        Maximum allowed relative variation (default: 0.1%).

    Returns
    -------
    dict {species_name: steady_state_value} if steady state detected,
    None otherwise.
    """
    t      = result.time
    conc   = result.concentrations
    t_span = t[-1] - t[0]
    t_cut  = t[-1] - window * t_span

    mask = t >= t_cut
    if mask.sum() < 3:
        return None  # not enough points in the window

    conc_window = conc[mask]
    means  = conc_window.mean(axis=0)
    ranges = conc_window.max(axis=0) - conc_window.min(axis=0)

    rel_var = np.where(means > 1e-15, ranges / means, ranges)

    if np.all(rel_var < tol):
        return {n: float(means[i]) for i, n in enumerate(result.species_names)}
    return None


# ─── Peak detection ───────────────────────────────────────────────────────────

def peak_detection(
    result: "SimulationResult",
    species_name: str,
    prominence: float = 0.01,
    distance: Optional[int] = None,
    rel_height: float = 0.5,
) -> Dict:
    """
    Find peaks and troughs in a species concentration trajectory.

    Parameters
    ----------
    result : SimulationResult
    species_name : str
    prominence : float
        Minimum peak prominence (relative to trajectory range).
    distance : int, optional
        Minimum samples between peaks.
    rel_height : float
        Relative height for width calculation (0–1).

    Returns
    -------
    dict with keys:
        peak_times, peak_values, trough_times, trough_values,
        mean_period, mean_amplitude, n_peaks, n_troughs
    """
    y  = result.species(species_name)
    t  = result.time
    dy = y.max() - y.min()

    if dy < 1e-15:
        return {
            "peak_times": np.array([]), "peak_values": np.array([]),
            "trough_times": np.array([]), "trough_values": np.array([]),
            "mean_period": None, "mean_amplitude": None,
            "n_peaks": 0, "n_troughs": 0,
        }

    abs_prominence = prominence * dy

    pk_idx, pk_props = signal.find_peaks(
        y, prominence=abs_prominence, distance=distance
    )
    tr_idx, tr_props = signal.find_peaks(
        -y, prominence=abs_prominence, distance=distance
    )

    peak_times  = t[pk_idx]
    peak_vals   = y[pk_idx]
    trough_times = t[tr_idx]
    trough_vals  = y[tr_idx]

    # Period estimate
    mean_period = None
    if len(peak_times) >= 2:
        periods = np.diff(peak_times)
        mean_period = float(np.mean(periods))

    # Amplitude estimate
    mean_amplitude = None
    if len(peak_vals) > 0 and len(trough_vals) > 0:
        mean_amplitude = float(np.mean(peak_vals) - np.mean(trough_vals))

    return {
        "peak_times":     peak_times,
        "peak_values":    peak_vals,
        "trough_times":   trough_times,
        "trough_values":  trough_vals,
        "mean_period":    mean_period,
        "mean_amplitude": mean_amplitude,
        "n_peaks":        len(pk_idx),
        "n_troughs":      len(tr_idx),
    }


# ─── Reaction completion time ─────────────────────────────────────────────────

def reaction_completion_time(
    result: "SimulationResult",
    species_name: str,
    threshold: float = 0.99,
    direction: str = "decrease",
) -> Optional[float]:
    """
    Find the time at which a species reaches a completion threshold.

    Parameters
    ----------
    result : SimulationResult
    species_name : str
    threshold : float
        Fraction of total change (0–1). Default 0.99 = 99% complete.
    direction : str
        'decrease' (species being consumed) or 'increase' (species being produced).

    Returns
    -------
    float or None – time at which threshold is crossed.
    """
    y  = result.species(species_name)
    t  = result.time

    y0   = y[0]
    yend = y[-1]
    dy   = yend - y0

    if abs(dy) < 1e-15:
        return None

    if direction == "decrease":
        target = y0 - threshold * abs(dy)
        crossings = np.where(y <= target)[0]
    else:
        target = y0 + threshold * abs(dy)
        crossings = np.where(y >= target)[0]

    if len(crossings) == 0:
        return None

    return float(t[crossings[0]])


# ─── Phase portrait data ──────────────────────────────────────────────────────

def phase_portrait(
    result: "SimulationResult",
    x_species: str,
    y_species: str,
    velocity: bool = True,
) -> Dict:
    """
    Extract phase portrait data for two species.

    Parameters
    ----------
    result : SimulationResult
    x_species, y_species : str
        Names of the two species to use as axes.
    velocity : bool
        If True, also compute approximate velocity (dx/dt, dy/dt) for quiver plots.

    Returns
    -------
    dict with keys: x, y, dx, dy (if velocity=True), t
    """
    x = result.species(x_species)
    y_arr = result.species(y_species)
    t = result.time

    out = {"x": x, "y": y_arr, "t": t,
           "x_label": x_species, "y_label": y_species}

    if velocity:
        dt = np.diff(t)
        dt = np.where(dt < 1e-15, 1e-15, dt)  # avoid division by zero
        dx = np.diff(x) / dt
        dy = np.diff(y_arr) / dt
        out["dx"] = np.append(dx, dx[-1])
        out["dy"] = np.append(dy, dy[-1])

    return out


# ─── Sensitivity analysis ─────────────────────────────────────────────────────

def compute_sensitivity_index(
    results: List["SimulationResult"],
    species_name: str,
    parameter_values: Sequence[float],
    metric: str = "final",
) -> np.ndarray:
    """
    Compute a normalized sensitivity index for a parameter sweep.

    Parameters
    ----------
    results : list of SimulationResult
        One result per parameter value.
    species_name : str
        Species to measure.
    parameter_values : sequence of float
        The swept parameter values.
    metric : str
        'final' = final concentration, 'max' = peak, 'auc' = area under curve.

    Returns
    -------
    np.ndarray of sensitivity (dY/dp · p/Y) at each parameter value.
    """
    p  = np.array(parameter_values, dtype=float)

    def get_metric(res: "SimulationResult") -> float:
        y = res.species(species_name)
        if metric == "final":
            return float(y[-1])
        elif metric == "max":
            return float(y.max())
        elif metric == "auc":
            return float(np.trapezoid(y, res.time))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use 'final', 'max', or 'auc'.")

    Y = np.array([get_metric(r) for r in results])

    # Normalized sensitivity: S = (dY/dp) * (p / Y)
    # Use central finite differences
    dY = np.gradient(Y, p)
    Y_safe = np.where(np.abs(Y) > 1e-15, Y, 1e-15)
    S = dY * p / Y_safe

    return S


# ─── Time series statistics ────────────────────────────────────────────────────

def time_series_stats(
    result: "SimulationResult",
    species_name: str,
) -> Dict:
    """
    Compute descriptive statistics for a single species trajectory.

    Returns
    -------
    dict with: mean, std, min, max, range, initial, final, cv (coeff of variation)
    """
    y = result.species(species_name)
    mean = float(y.mean())
    std  = float(y.std())
    return {
        "mean":    mean,
        "std":     std,
        "min":     float(y.min()),
        "max":     float(y.max()),
        "range":   float(y.max() - y.min()),
        "initial": float(y[0]),
        "final":   float(y[-1]),
        "cv":      std / mean if mean > 1e-15 else float("nan"),
    }
