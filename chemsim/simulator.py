"""
simulator.py – Simulator and SimulationResult.

Simulator is intentionally thin: it validates the network, packs data,
calls the C++ core, and returns a SimulationResult. All logic lives
in the domain model (network.py) or the analysis/plotting modules.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from chemsim.network import ReactionNetwork

# ─────────────────────────────────────────────────────────────────────────────
# Default solver parameters
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PARAMS = {
    "t_start":               0.0,
    "t_end":                 100.0,
    "abs_tol":               1e-9,
    "rel_tol":               1e-6,
    "max_step":              1.0,
    "initial_step":          1e-4,
    "max_steps":             500_000,
    "output_interval":       0.1,
    "method":                "bdf",      # "bdf", "adams", "rk4"
    "use_analytical_jacobian": True,
}


# ─────────────────────────────────────────────────────────────────────────────
# SimulationResult
# ─────────────────────────────────────────────────────────────────────────────

class SimulationResult:
    """
    Container for a completed simulation trajectory.

    Attributes
    ----------
    time : np.ndarray, shape (N,)
        Output time points.
    concentrations : np.ndarray, shape (N, M)
        Concentrations of each species at each time point.
    species_names : list of str
        Ordered species names (columns of `concentrations`).
    n_steps : int
        Number of internal integrator steps taken.
    n_rhs_evals : int
        Number of ODE right-hand side evaluations.
    n_jac_evals : int
        Number of Jacobian evaluations.
    converged : bool
        Whether the integrator converged successfully.
    message : str
        Solver status message.
    conservation_laws : list of dict
        Detected conserved moieties.
    wall_time_s : float
        Wall-clock time for the simulation in seconds.
    """

    def __init__(
        self,
        time: np.ndarray,
        concentrations: np.ndarray,
        species_names: List[str],
        n_steps: int = 0,
        n_rhs_evals: int = 0,
        n_jac_evals: int = 0,
        converged: bool = True,
        message: str = "OK",
        conservation_laws: Optional[List[dict]] = None,
        wall_time_s: float = 0.0,
        network: Optional[ReactionNetwork] = None,
    ) -> None:
        self.time            = np.asarray(time, dtype=np.float64)
        self.concentrations  = np.asarray(concentrations, dtype=np.float64)
        self.species_names   = list(species_names)
        self.n_steps         = int(n_steps)
        self.n_rhs_evals     = int(n_rhs_evals)
        self.n_jac_evals     = int(n_jac_evals)
        self.converged       = bool(converged)
        self.message         = str(message)
        self.conservation_laws = conservation_laws or []
        self.wall_time_s     = float(wall_time_s)
        self._network        = network

        if not self.converged:
            warnings.warn(
                f"Simulation did not converge: {self.message}",
                RuntimeWarning,
                stacklevel=3,
            )

    # ── Convenient accessors ──────────────────────────────────────────────────

    def species(self, name: str) -> np.ndarray:
        """Return the concentration trajectory for a species by name."""
        try:
            idx = self.species_names.index(name)
        except ValueError:
            raise KeyError(f"Unknown species '{name}'. Available: {self.species_names}")
        return self.concentrations[:, idx]

    def final_state(self) -> Dict[str, float]:
        """Return {species_name: final_concentration} dict."""
        return {n: float(self.concentrations[-1, i])
                for i, n in enumerate(self.species_names)}

    def at_time(self, t: float) -> Dict[str, float]:
        """
        Interpolate concentrations at arbitrary time *t*.
        Uses linear interpolation between recorded output points.
        """
        if t < self.time[0] or t > self.time[-1]:
            raise ValueError(f"t={t} is outside simulation range [{self.time[0]}, {self.time[-1]}]")
        idx = np.searchsorted(self.time, t)
        if idx == 0:
            row = self.concentrations[0]
        elif idx >= len(self.time):
            row = self.concentrations[-1]
        else:
            alpha = (t - self.time[idx - 1]) / (self.time[idx] - self.time[idx - 1])
            row = (1 - alpha) * self.concentrations[idx - 1] + alpha * self.concentrations[idx]
        return {n: float(row[i]) for i, n in enumerate(self.species_names)}

    def to_dataframe(self):
        """Return a pandas DataFrame with columns [time] + species_names."""
        import pandas as pd
        data = {"time": self.time}
        for i, n in enumerate(self.species_names):
            data[n] = self.concentrations[:, i]
        return pd.DataFrame(data)

    # ── Plotting ──────────────────────────────────────────────────────────────

    def plot(
        self,
        species: Optional[Sequence[str]] = None,
        title: Optional[str] = None,
        figsize: tuple = (10, 5),
        **kwargs,
    ):
        """
        Quick plot of trajectories. Returns the matplotlib Figure.

        Parameters
        ----------
        species : list of str, optional
            Which species to plot (default: all).
        title : str, optional
            Figure title (default: network name).
        figsize : tuple
            Matplotlib figsize.
        **kwargs
            Passed to ``plot_trajectories``.
        """
        from chemsim.plotting import plot_trajectories
        return plot_trajectories(
            self,
            species=species,
            title=title or (self._network.name if self._network else "ChemSim"),
            figsize=figsize,
            **kwargs,
        )

    def phase_portrait(self, x_species: str, y_species: str, **kwargs):
        """Plot a 2-D phase portrait of x_species vs y_species."""
        from chemsim.plotting import plot_phase_portrait
        return plot_phase_portrait(self, x_species, y_species, **kwargs)

    def steady_state(self, window: float = 0.05, tol: float = 1e-3) -> Optional[Dict[str, float]]:
        """Detect steady state concentrations. Returns None if not reached."""
        from chemsim.analysis import steady_state_analysis
        return steady_state_analysis(self, window=window, tol=tol)

    def peaks(self, species_name: str, **kwargs) -> dict:
        """Detect peaks/troughs in a species trajectory."""
        from chemsim.analysis import peak_detection
        return peak_detection(self, species_name, **kwargs)

    # ── Rendering ─────────────────────────────────────────────────────────

    def render(self):
        from chemsim.renderer import ParticleRenderer
        rnd = ParticleRenderer(result= self)
        rnd.run()

    # ── Serialization ─────────────────────────────────────────────────────────

    def save(self, path: str, fmt: str = "hdf5") -> None:
        """
        Save result to disk.

        Parameters
        ----------
        path : str
            Output file path.
        fmt : str
            'hdf5' (default) or 'csv'.
        """
        if fmt == "hdf5":
            from chemsim.writer import save_hdf5
            save_hdf5(self, path)
        elif fmt == "csv":
            from chemsim.writer import save_csv
            save_csv(self, path)
        else:
            raise ValueError(f"Unknown format '{fmt}'. Use 'hdf5' or 'csv'.")

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"SimulationResult("
            f"{len(self.species_names)} species, "
            f"{len(self.time)} timepoints, "
            f"t=[{self.time[0]:.3g}, {self.time[-1]:.3g}], "
            f"converged={self.converged})"
        )

    def stats(self) -> str:
        """Return a formatted summary of solver statistics."""
        lines = [
            "── Simulation Statistics ──────────────────────",
            f"  Converged     : {self.converged}  ({self.message})",
            f"  Wall time     : {self.wall_time_s:.3f} s",
            f"  Output points : {len(self.time)}",
            f"  Solver steps  : {self.n_steps:,}",
            f"  RHS evals     : {self.n_rhs_evals:,}",
            f"  Jacobian evals: {self.n_jac_evals:,}",
        ]
        if self.conservation_laws:
            lines.append(f"  Conserved moieties: {len(self.conservation_laws)}")
        return "\n".join(lines)



# ─────────────────────────────────────────────────────────────────────────────
# Pure-Python scipy fallback (used when _chemsim_core C++ extension is absent)
# ─────────────────────────────────────────────────────────────────────────────

def _run_scipy(network, params: dict, callback=None) -> dict:
    """
    Solve the reaction ODE using scipy.integrate.solve_ivp (BDF).

    Produces the same output dict structure as _chemsim_core.run_simulation
    so SimulationResult can be constructed identically.

    Rate law: r = k(T) · ∏ [C_i]^ν_i
    Arrhenius: k(T) = A·exp(-Ea/(R·T))  when Ea>0 and A>0
               k(T) = k_ref·exp(-Ea/R·(1/T - 1/298.15))  when Ea>0 but A=0
    """
    import numpy as np
    from scipy.integrate import solve_ivp

    R_GAS    = 8.314        # J/(mol·K)
    species  = list(network.species_names)
    n_sp     = len(species)
    sp_idx   = {sp: i for i, sp in enumerate(species)}
    y0       = network.initial_conditions_array().copy()
    reactions = network.reactions_as_core_dicts()
    T_spec   = network.temperature_spec
    t0       = params["t_start"]
    t1       = params["t_end"]

    def _T_at(t: float) -> float:
        if isinstance(T_spec, (int, float)):
            return float(T_spec)
        kind = T_spec.get("type", "constant")
        if kind == "constant":
            return float(T_spec.get("T", 298.15))
        if kind == "step":
            for seg in T_spec.get("segments", []):
                if seg[0] <= t < seg[1]:
                    return float(seg[2])
            return float(T_spec.get("T_default", 298.15))
        if kind == "ramp":
            for seg in T_spec.get("segments", []):
                t0s, t1s, T0s, T1s = float(seg[0]), float(seg[1]), float(seg[2]), float(seg[3])
                if t0s <= t <= t1s:
                    frac = (t - t0s) / max(t1s - t0s, 1e-12)
                    return T0s + frac * (T1s - T0s)
            return float(T_spec.get("T_default", 298.15))
        return 298.15

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        T    = _T_at(t)
        dydt = np.zeros(n_sp)
        y_safe = np.maximum(y, 0.0)   # prevent negative concentrations in rate calc

        for rxn in reactions:
            k_base = float(rxn.get("rate", 0.0))
            Ea     = float(rxn.get("activation_energy", 0.0))
            A_pre  = float(rxn.get("pre_exponential", 0.0))

            # Effective rate constant at temperature T
            if Ea > 0:
                if A_pre > 0:
                    k = A_pre * np.exp(-Ea / (R_GAS * T))
                else:
                    # Arrhenius correction from reference k at 298.15 K
                    k = k_base * np.exp(-Ea / R_GAS * (1.0 / T - 1.0 / 298.15))
            else:
                k = k_base

            # Power-law: r = k * ∏ C_i^stoich_i
            r_stoich = rxn.get("reactant_stoich") or [1.0] * len(rxn.get("reactants", []))
            p_stoich = rxn.get("product_stoich")  or [1.0] * len(rxn.get("products", []))

            rate = k
            for sp, s in zip(rxn.get("reactants", []), r_stoich):
                idx = sp_idx.get(sp)
                if idx is not None:
                    rate *= y_safe[idx] ** float(s)

            # Accumulate: reactants consumed, products formed
            for sp, s in zip(rxn.get("reactants", []), r_stoich):
                idx = sp_idx.get(sp)
                if idx is not None:
                    dydt[idx] -= float(s) * rate
            for sp, s in zip(rxn.get("products", []), p_stoich):
                idx = sp_idx.get(sp)
                if idx is not None:
                    dydt[idx] += float(s) * rate

        return dydt

    # Build t_eval from output_interval
    dt      = float(params.get("output_interval", (t1 - t0) / 200))
    t_eval  = np.arange(t0, t1 + dt * 0.5, dt)
    t_eval  = np.clip(t_eval, t0, t1)

    max_step = float(params.get("max_step", np.inf))
    rtol     = float(params.get("rel_tol", 1e-6))
    atol     = float(params.get("abs_tol", 1e-9))

    sol = solve_ivp(
        rhs, [t0, t1], y0,
        method="BDF",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
        dense_output=False,
    )

    # Clip negatives (physically impossible concentrations)
    conc = np.maximum(sol.y.T, 0.0)   # shape (N_times, N_species)

    return {
        "time"            : sol.t,
        "concentrations"  : conc,
        "species_names"   : species,
        "n_steps"         : int(sol.t.shape[0]),
        "n_rhs_evals"     : int(sol.nfev),
        "n_jac_evals"     : 0,
        "converged"       : bool(sol.success),
        "message"         : sol.message,
        "conservation_laws": [],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class Simulator:
    """
    Orchestrates a simulation run.

    Validates the network, packs data into NumPy arrays, calls the C++
    solver core, and returns a ``SimulationResult``.

    Parameters
    ----------
    network : ReactionNetwork
        The reaction network to simulate.

    Examples
    --------
    >>> sim = Simulator(network)
    >>> result = sim.run(t_end=500.0, method="bdf", output_interval=0.5)
    >>> result.plot()
    """

    def __init__(self, network: ReactionNetwork) -> None:
        if not isinstance(network, ReactionNetwork):
            raise TypeError("network must be a ReactionNetwork instance")
        self.network = network

    def run(
        self,
        t_end: float = 100.0,
        t_start: float = 0.0,
        method: str = "bdf",
        output_interval: float = 0.1,
        abs_tol: float = 1e-9,
        rel_tol: float = 1e-6,
        max_step: float = 1.0,
        initial_step: float = 1e-4,
        max_steps: int = 500_000,
        use_analytical_jacobian: bool = True,
        progress: bool = False,
        progress_callback: Optional[Callable[[float, float], None]] = None,
    ) -> SimulationResult:
        """
        Run the simulation and return a SimulationResult.

        Parameters
        ----------
        t_end : float
            End time for the simulation.
        t_start : float
            Start time (default 0.0).
        method : str
            Integrator: 'bdf' (stiff, recommended), 'adams' (non-stiff), 'rk4' (fallback).
        output_interval : float
            Time between recorded output points.
        abs_tol : float
            Absolute tolerance for adaptive steppers.
        rel_tol : float
            Relative tolerance for adaptive steppers.
        max_step : float
            Maximum allowed timestep.
        initial_step : float
            Initial step size hint.
        max_steps : int
            Maximum number of internal steps before aborting.
        use_analytical_jacobian : bool
            Use analytical Jacobian (recommended). Set False to use finite differences.
        progress : bool
            Print a simple progress bar to stdout.
        progress_callback : callable, optional
            Called as ``f(t_current, t_end)`` periodically during integration.

        Returns
        -------
        SimulationResult
        """
        # Validate before we touch any solver
        self.network.validate()

        solver_params = {
            "t_start":               float(t_start),
            "t_end":                 float(t_end),
            "abs_tol":               float(abs_tol),
            "rel_tol":               float(rel_tol),
            "max_step":              float(max_step),
            "initial_step":          float(initial_step),
            "max_steps":             int(max_steps),
            "output_interval":       float(output_interval),
            "method":                method,
            "use_analytical_jacobian": bool(use_analytical_jacobian),
        }

        cb: Optional[Callable] = progress_callback
        if progress and cb is None:
            cb = _make_progress_printer(t_end)

        # Try C++ core first; fall back to pure-Python scipy solver
        try:
            from chemsim._chemsim_core import run_simulation as _run
            _use_cpp = True
        except ImportError:
            _use_cpp = False

        t0_wall = time.perf_counter()

        if _use_cpp:
            raw = _run(
                species_names       = self.network.species_names,
                reactions           = self.network.reactions_as_core_dicts(),
                initial_conditions  = self.network.initial_conditions_array(),
                temperature_spec    = self.network.temperature_spec,
                solver_params       = solver_params,
                progress_callback   = cb,
            )
        else:
            raw = _run_scipy(self.network, solver_params, cb)

        wall_time = time.perf_counter() - t0_wall
        if progress:
            print()

        return SimulationResult(
            time             = raw["time"],
            concentrations   = raw["concentrations"],
            species_names    = raw["species_names"],
            n_steps          = raw.get("n_steps", 0),
            n_rhs_evals      = raw.get("n_rhs_evals", 0),
            n_jac_evals      = raw.get("n_jac_evals", 0),
            converged        = raw.get("converged", True),
            message          = raw.get("message", "OK"),
            conservation_laws= raw.get("conservation_laws", []),
            wall_time_s      = wall_time,
            network          = self.network,
        )

    def parameter_sweep(
        self,
        parameter: str,
        values: Sequence[float],
        t_end: float = 100.0,
        **run_kwargs,
    ) -> List[SimulationResult]:
        """
        Run multiple simulations sweeping a single rate constant.

        Parameters
        ----------
        parameter : str
            Species name to sweep initial concentration of, OR
            an int index to select a reaction's rate constant.
        values : sequence of float
            Values to sweep.
        t_end : float
            End time for each run.

        Returns
        -------
        list of SimulationResult, one per value.
        """
        results = []
        original_net = self.network.copy()

        for val in values:
            # Try as species initial condition first
            if parameter in self.network.species_names:
                self.network.set_initial(parameter, val)
            else:
                raise ValueError(
                    f"'{parameter}' is not a known species name. "
                    "For reaction rate sweeps, modify network.reactions directly."
                )
            result = self.run(t_end=t_end, **run_kwargs)
            results.append(result)

        # Restore original
        self.network = original_net
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Progress helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_progress_printer(t_end: float) -> Callable[[float, float], None]:
    """Returns a callback that prints a simple ASCII progress bar."""
    width = 40
    last_pct = [-1]

    def callback(t: float, _t_end: float) -> None:
        pct = int(100 * t / t_end)
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        filled = int(width * pct / 100)
        bar = "█" * filled + "░" * (width - filled)
        print(f"\r  [{bar}] {pct:3d}%  t={t:.2f}", end="", flush=True)

    return callback
