"""
network/reactor/cstr.py  –  CSTR reactor built on ChemSim

A Continuous Stirred-Tank Reactor that wraps ChemSim's Simulator to
provide steady-state and dynamic simulation with:
  - Full ChemSim reaction network (any number of species/reactions)
  - Temperature gradients / Arrhenius kinetics
  - Inlet feed streams with arbitrary compositions
  - Residence time control (τ = V/Q)
  - Heat balance (energy equation with jacket/feed cooling)
"""

from __future__ import annotations

import copy
import sys, os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── ChemSim import ────────────────────────────────────────────────────────────
_project_dir = "/mnt/project"
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

try:
    from chemsim.network import ReactionNetwork, Species, Reaction
    from chemsim.simulator import Simulator
    _HAS_CHEMSIM = True
except ImportError:
    _HAS_CHEMSIM = False
    ReactionNetwork = None
    Simulator = None

# ── Constants ─────────────────────────────────────────────────────────────────

R_GAS = 8.314   # J/(mol·K)

# ── Optional Gillespie SSA ────────────────────────────────────────────────────
try:
    import sys as _sys2, os as _os2
    # Try the directory that contains THIS file first, then /mnt/project as fallback
    _gillespie_candidates = [
        _os2.path.dirname(_os2.path.abspath(__file__)),   # same dir as cstr.py
        _os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__))),  # parent
        "/mnt/project",
    ]
    for _gp in _gillespie_candidates:
        if _gp not in _sys2.path:
            _sys2.path.insert(0, _gp)
    from gillespie import GillespieSimulator
    _HAS_GILLESPIE = True
except ImportError:
    _HAS_GILLESPIE = False
    GillespieSimulator = None


# ── Feed stream ───────────────────────────────────────────────────────────────

@dataclass
class FeedStream:
    """
    Inlet feed stream for a CSTR.

    Parameters
    ----------
    name : str
        Identifier for this feed (e.g. "primary", "recycle").
    compositions : dict
        Species name → concentration in M (mol/L).
    flow_rate_m3s : float
        Volumetric flow rate in m³/s.
    temperature_K : float
        Feed temperature in Kelvin.
    """
    name          : str
    compositions  : Dict[str, float] = field(default_factory=dict)
    flow_rate_m3s : float = 1e-4
    temperature_K : float = 298.15

    @property
    def flow_rate_Ls(self) -> float:
        return self.flow_rate_m3s * 1000.0

    def to_dict(self) -> dict:
        return {
            "name"          : self.name,
            "compositions"  : dict(self.compositions),
            "flow_rate_m3s" : self.flow_rate_m3s,
            "temperature_K" : self.temperature_K,
        }


# ── Temperature profile ───────────────────────────────────────────────────────

@dataclass
class TemperatureGradient:
    """
    Time-varying or ramp temperature profile for a CSTR.

    Supports:
    - Isothermal:  constant temperature
    - Linear ramp: from T_start to T_end over t_ramp seconds
    - Step:        step change at t_step seconds
    - Custom:      user-supplied (time, temperature) arrays

    Parameters
    ----------
    mode : str
        'isothermal' | 'ramp' | 'step' | 'custom'
    T_initial_K : float
        Initial / constant temperature.
    T_final_K : float
        Final temperature (for ramp/step modes).
    t_ramp : float
        Duration of ramp in simulation time units.
    t_step : float
        Time of step change.
    custom_times : list of float
        For 'custom' mode: time breakpoints.
    custom_temps : list of float
        For 'custom' mode: temperature values.
    """
    mode         : str            = "isothermal"
    T_initial_K  : float          = 298.15
    T_final_K    : float          = 298.15
    t_ramp       : float          = 100.0
    t_step       : float          = 50.0
    custom_times : List[float]    = field(default_factory=list)
    custom_temps : List[float]    = field(default_factory=list)

    def temperature_at(self, t: float) -> float:
        """Return temperature at simulation time t."""
        if self.mode == "isothermal":
            return self.T_initial_K
        elif self.mode == "ramp":
            frac = min(1.0, t / max(self.t_ramp, 1e-12))
            return self.T_initial_K + frac * (self.T_final_K - self.T_initial_K)
        elif self.mode == "step":
            return self.T_final_K if t >= self.t_step else self.T_initial_K
        elif self.mode == "custom":
            if len(self.custom_times) < 2:
                return self.T_initial_K
            return float(np.interp(t, self.custom_times, self.custom_temps))
        return self.T_initial_K

    def to_dict(self) -> dict:
        return {
            "mode"        : self.mode,
            "T_initial_K" : self.T_initial_K,
            "T_final_K"   : self.T_final_K,
            "t_ramp"      : self.t_ramp,
            "t_step"      : self.t_step,
            "custom_times": self.custom_times,
            "custom_temps": self.custom_temps,
        }


# ── CSTR Result ───────────────────────────────────────────────────────────────

@dataclass
class CSTRResult:
    """
    Result from a CSTR simulation run.

    Attributes
    ----------
    reactor_name : str
    time : np.ndarray, shape (N,)
    concentrations : np.ndarray, shape (N, M)
    species_names : list of str
    outlet_composition : dict
        Final concentrations at steady state.
    residence_time_s : float
    conversion : dict
        Fractional conversion per species.
    temperature_profile : np.ndarray, shape (N,)
    heat_generated_W : float
        Approximate heat of reaction (requires delta_H data).
    converged : bool
    issues : list of str
    """
    reactor_name        : str
    time                : np.ndarray
    concentrations      : np.ndarray
    species_names       : List[str]
    outlet_composition  : Dict[str, float]
    residence_time_s    : float
    conversion          : Dict[str, float]
    temperature_profile : np.ndarray
    heat_generated_W    : float     = 0.0
    converged           : bool      = True
    issues              : List[str] = field(default_factory=list)
    # ── V3.0 stochastic fields ─────────────────────────────────────────────────
    simulation_mode     : str       = "deterministic"  # "deterministic" | "stochastic"
    stochastic_data     : Optional[dict] = None        # GillespieResult.to_dict()

    def to_dict(self) -> dict:
        d = {
            "reactor_name"       : self.reactor_name,
            "time"               : self.time.tolist(),
            "concentrations"     : self.concentrations.tolist(),
            "species_names"      : self.species_names,
            "outlet_composition" : self.outlet_composition,
            "residence_time_s"   : self.residence_time_s,
            "conversion"         : self.conversion,
            "temperature_profile": self.temperature_profile.tolist(),
            "heat_generated_W"   : self.heat_generated_W,
            "converged"          : self.converged,
            "issues"             : self.issues,
            "simulation_mode"    : self.simulation_mode,
        }
        if self.stochastic_data is not None:
            d["stochastic_data"] = self.stochastic_data
        return d


# ── CSTR class ────────────────────────────────────────────────────────────────

class CSTR:
    """
    Continuous Stirred-Tank Reactor (CSTR).

    Uses ChemSim's ReactionNetwork and Simulator internally for
    chemical kinetics. Wraps the result with CSTR-specific mass
    balance equations (dilution terms) and heat balance.

    The effective ODE for species i in the CSTR is:
        dC_i/dt = (C_i_in - C_i) / τ + Σ_j ν_ij r_j(C, T)

    where τ = V / Q_total is the residence time.

    Parameters
    ----------
    name : str
        Unique reactor name.
    volume_L : float
        Reactor volume in litres.
    reaction_network : ReactionNetwork
        ChemSim reaction network for this reactor.
    temperature_gradient : TemperatureGradient, optional
        Temperature profile. Default = isothermal at 298.15 K.
    heat_transfer_coeff : float
        UA [W/K]: overall heat transfer coefficient × area for jacket.
    jacket_temperature_K : float
        Jacket temperature for cooling/heating.
    """

    def __init__(
        self,
        name               : str,
        volume_L           : float,
        reaction_network   : "ReactionNetwork",
        temperature_gradient: Optional[TemperatureGradient] = None,
        heat_transfer_coeff : float = 0.0,
        jacket_temperature_K: float = 298.15,
    ):
        self.name                = name
        self.volume_L            = volume_L
        self.reaction_network    = reaction_network
        self.temperature_gradient = temperature_gradient or TemperatureGradient()
        self.heat_transfer_coeff = heat_transfer_coeff
        self.jacket_temperature_K= jacket_temperature_K

        self._feeds   : List[FeedStream] = []
        self._result  : Optional[CSTRResult] = None

        # Outlet concentration (updated after simulation)
        self.outlet_composition : Dict[str, float] = {}

    # ── Feed management ───────────────────────────────────────────────────────

    def add_feed(self, feed: FeedStream) -> None:
        """Add an inlet feed stream."""
        self._feeds.append(feed)

    def remove_feed(self, name: str) -> None:
        self._feeds = [f for f in self._feeds if f.name != name]

    def set_feed_flow(self, name: str, flow_rate_m3s: float) -> None:
        for f in self._feeds:
            if f.name == name:
                f.flow_rate_m3s = flow_rate_m3s
                return
        raise KeyError(f"Feed '{name}' not found")

    def add_species_to_feed(
        self, feed_name: str, species: str, concentration: float
    ) -> None:
        """Add or update a species concentration in a named feed."""
        for f in self._feeds:
            if f.name == feed_name:
                f.compositions[species] = concentration
                return
        raise KeyError(f"Feed '{feed_name}' not found")

    # ── Network species management ────────────────────────────────────────────

    def add_species(self, name: str, initial: float = 0.0) -> None:
        """Add a species to the reaction network."""
        if _HAS_CHEMSIM:
            self.reaction_network.add_species(name, initial=initial)

    def add_reaction(
        self,
        reactants: List[str],
        products: List[str],
        rate: float,
        reactant_stoich: Optional[List[float]] = None,
        product_stoich: Optional[List[float]] = None,
        activation_energy: float = 0.0,
        pre_exponential: float = 0.0,
        label: str = "",
    ) -> None:
        """Add a reaction to the network."""
        if _HAS_CHEMSIM:
            kwargs = dict(
                rate=rate,
                activation_energy=activation_energy,
                pre_exponential=pre_exponential,
                label=label,
            )
            if reactant_stoich:
                kwargs["reactant_stoich"] = reactant_stoich
            if product_stoich:
                kwargs["product_stoich"] = product_stoich
            self.reaction_network.add_reaction(reactants, products, **kwargs)

    # ── Flow properties ───────────────────────────────────────────────────────

    @property
    def total_flow_m3s(self) -> float:
        return sum(f.flow_rate_m3s for f in self._feeds)

    @property
    def total_flow_Ls(self) -> float:
        return self.total_flow_m3s * 1000.0

    @property
    def residence_time_s(self) -> float:
        Q = self.total_flow_Ls  # L/s
        if Q <= 0:
            return float("inf")
        return (self.volume_L / Q)   # s

    @property
    def residence_time_min(self) -> float:
        return self.residence_time_s / 60.0

    # ── Mixed inlet composition ───────────────────────────────────────────────

    def _mixed_inlet(self) -> Dict[str, float]:
        """Flow-averaged inlet concentrations (M)."""
        Q_total = self.total_flow_Ls
        if Q_total <= 0:
            return {}
        mixed: Dict[str, float] = {}
        for feed in self._feeds:
            frac = feed.flow_rate_Ls / Q_total
            for sp, c in feed.compositions.items():
                mixed[sp] = mixed.get(sp, 0.0) + frac * c
        return mixed

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate(
        self,
        t_end       : float = 500.0,
        n_segments  : int   = 20,
        solver_params: Optional[dict] = None,
        mode        : str   = "deterministic",
        stochastic_params: Optional[dict] = None,
    ) -> CSTRResult:
        """
        Run CSTR dynamic simulation.

        Parameters
        ----------
        t_end : float
            Total simulation time.
        n_segments : int
            Number of temperature segments.
        solver_params : dict, optional
            Extra parameters for ChemSim Simulator.
        mode : str
            'deterministic' (default ODE solver) or 'stochastic' (Gillespie SSA).
        stochastic_params : dict, optional
            SSA-specific parameters: n_trajectories, omega, n_samples, tau_eps, seed.
        """
        if mode == "stochastic":
            return self.simulate_stochastic(
                t_end=t_end,
                n_segments=n_segments,
                **(stochastic_params or {}),
            )
        return self._simulate_deterministic(t_end=t_end, n_segments=n_segments, solver_params=solver_params)

    def _simulate_deterministic(
        self,
        t_end       : float = 500.0,
        n_segments  : int   = 20,
        solver_params: Optional[dict] = None,
    ) -> CSTRResult:
        """
        Deterministic ODE simulation (original ChemSim BDF path).
        """
        if not _HAS_CHEMSIM:
            raise RuntimeError("ChemSim is not installed. Cannot run CSTR simulation.")

        issues: List[str] = []
        τ = self.residence_time_s
        if τ == float("inf"):
            issues.append("No feed streams — residence time is infinite. Add a feed.")

        # Build initial conditions from mixed inlet
        inlet = self._mixed_inlet()
        net_copy = copy.deepcopy(self.reaction_network)

        # Helper: get species names from network (uses _species_order / species_names)
        def _get_species_names(net) -> List[str]:
            return list(net.species_names)

        def _get_species_initials(net) -> dict:
            return {name: net._species[name].initial for name in net._species_order}

        def _set_species_initial(net, name: str, val: float):
            if name in net._species:
                net._species[name].initial = float(val)

        # Ensure all inlet species are in the network
        existing = set(_get_species_names(net_copy))
        for sp, c in inlet.items():
            if sp not in existing:
                net_copy.add_species(sp, initial=c)

        # Set initial concentrations in network to inlet (CSTR startup)
        for sp_name in _get_species_names(net_copy):
            if sp_name in inlet:
                _set_species_initial(net_copy, sp_name, inlet[sp_name])

        dt_seg = t_end / max(n_segments, 1)
        t_all: List[np.ndarray] = []
        c_all: List[np.ndarray] = []
        T_all: List[float]      = []

        current_conc: Optional[np.ndarray] = None
        species_names = _get_species_names(net_copy)

        params = {
            "t_start"        : 0.0,
            "t_end"          : dt_seg,
            "abs_tol"        : 1e-8,
            "rel_tol"        : 1e-6,
            "max_step"       : dt_seg / 20.0,
            "method"         : "bdf",
            "output_interval": dt_seg / 50.0,
        }
        if solver_params:
            params.update(solver_params)

        t_offset = 0.0
        for seg in range(n_segments):
            t_mid = t_offset + dt_seg / 2.0
            T = self.temperature_gradient.temperature_at(t_mid)
            T_all.append(T)

            # Deep-copy network for this segment
            net_seg = copy.deepcopy(net_copy)

            # Update effective rate constants for temperature (Arrhenius)
            if abs(T - 298.15) > 0.5:
                for rxn in net_seg.reactions:
                    if rxn.activation_energy > 0:
                        k_ref = rxn.pre_exponential if rxn.pre_exponential > 0 else rxn.rate
                        rxn.rate = k_ref * np.exp(-rxn.activation_energy / (R_GAS * T))

            # Set initial concentrations from previous segment
            if current_conc is not None:
                for i, sp_name in enumerate(species_names):
                    _set_species_initial(net_seg, sp_name, float(current_conc[i]))

            params["t_start"] = 0.0
            params["t_end"]   = dt_seg

            sim = Simulator(net_seg)
            try:
                res = sim.run(**params)
            except Exception as e:
                issues.append(f"Segment {seg}: solver error — {e}")
                break

            # Apply CSTR dilution correction (exponential mixing):
            # C(t+dt) ≈ C_batch(dt) * exp(-dt/τ) + C_in * (1 - exp(-dt/τ))
            if τ < float("inf"):
                alpha = np.exp(-dt_seg / τ)
                c_inlet_arr = np.array([inlet.get(sp, 0.0) for sp in species_names])
                corrected = np.zeros_like(res.concentrations)
                for row_i, row in enumerate(res.concentrations):
                    corrected[row_i] = row * alpha + c_inlet_arr * (1.0 - alpha)
                res.concentrations[:] = corrected

            t_all.append(res.time + t_offset)
            c_all.append(res.concentrations)

            if res.converged and len(res.concentrations) > 0:
                current_conc = res.concentrations[-1].copy()
            else:
                issues.append(f"Segment {seg} did not converge.")
                break

            t_offset += dt_seg

        if not t_all:
            # Empty result
            time_arr = np.array([0.0])
            conc_arr = np.zeros((1, len(species_names)))
        else:
            time_arr = np.concatenate(t_all)
            conc_arr = np.vstack(c_all)

        # Final outlet composition
        outlet: Dict[str, float] = {}
        if len(conc_arr) > 0:
            final_row = conc_arr[-1]
            for i, name in enumerate(species_names):
                outlet[name] = float(final_row[i])

        self.outlet_composition = outlet

        # Conversion = (C_in - C_out) / C_in for each species in inlet
        conversion: Dict[str, float] = {}
        for sp, c_in in inlet.items():
            if c_in > 0 and sp in outlet:
                conversion[sp] = max(0.0, (c_in - outlet[sp]) / c_in)

        # Temperature profile array
        T_arr = np.array([
            self.temperature_gradient.temperature_at(t) for t in time_arr
        ])

        self._result = CSTRResult(
            reactor_name        = self.name,
            time                = time_arr,
            concentrations      = conc_arr,
            species_names       = species_names,
            outlet_composition  = outlet,
            residence_time_s    = τ,
            conversion          = conversion,
            temperature_profile = T_arr,
            converged           = len(issues) == 0,
            issues              = issues,
            simulation_mode     = "deterministic",
        )
        return self._result

    # ── Stochastic simulation (Gillespie SSA) ─────────────────────────────────

    def simulate_stochastic(
        self,
        t_end          : float = 500.0,
        n_segments     : int   = 10,
        n_trajectories : int   = 30,
        omega          : int   = 300,
        n_samples      : int   = 80,
        tau_eps        : float = 0.03,
        seed           : Optional[int] = None,
    ) -> CSTRResult:
        """
        Stochastic CSTR simulation via Gillespie SSA (τ-leaping with exact fallback).

        The deterministic CSTR dilution-correction is replaced by explicit
        inflow (∅→Xᵢ) and outflow (Xᵢ→∅) pseudo-reactions at rate D = Q/V.

        Parameters
        ----------
        t_end          : float   simulation end time
        n_segments     : int     temperature segments (Arrhenius piecewise)
        n_trajectories : int     number of independent SSA realisations
        omega          : int     system-size parameter Ω (noise-speed trade-off)
        n_samples      : int     evenly-spaced output time points
        tau_eps        : float   τ-leaping error bound ε (0.01–0.05)
        seed           : int     RNG seed for reproducibility

        Returns
        -------
        CSTRResult with simulation_mode='stochastic' and stochastic_data populated.
        The `concentrations` field holds the ensemble mean (for backward-compatible
        2-D canvas rendering); the full PDF is in stochastic_data.
        """
        if not _HAS_GILLESPIE:
            raise RuntimeError(
                "gillespie.py not found. Ensure it is in the Python path "
                "(same directory as cstr.py, the project root, or /mnt/project)."
            )

        τ      = self.residence_time_s
        issues : List[str] = []
        if τ == float("inf"):
            issues.append("No feed streams — residence time is infinite.")

        inlet        = self._mixed_inlet()

        # ── B1 FIX: start from reaction network species, then add any
        # feed-only species not yet tracked.  Without this, species
        # that arrive exclusively via _route_outlet_to_feeds (e.g. the
        # upstream reactor's outlet species) have zero inflow propensity
        # and are invisible to the Gillespie simulator.
        species_names = list(self.reaction_network.species_names)
        for sp in inlet:
            if sp not in species_names:
                species_names.append(sp)

        # Build reactions list from ReactionNetwork
        reactions_raw = []
        for rxn in self.reaction_network.reactions:
            reactions_raw.append({
                "reactants"        : list(rxn.reactants),
                "products"         : list(rxn.products),
                "reactant_stoich"  : list(rxn.reactant_stoich),
                "product_stoich"   : list(rxn.product_stoich),
                "rate"             : rxn.rate,
                "activation_energy": rxn.activation_energy,
                "pre_exponential"  : rxn.pre_exponential,
            })

        D = (self.total_flow_Ls / self.volume_L) if self.volume_L > 0 else 0.0

        # Initial concentrations: use inlet (CSTR start-up from feed composition)
        initial_conc = {sp: inlet.get(sp, 0.0) for sp in species_names}

        sim = GillespieSimulator(
            species_names  = species_names,
            reactions_raw  = reactions_raw,
            initial_conc   = initial_conc,
            inlet_conc     = inlet,
            dilution_rate  = D,
            temperature_fn = self.temperature_gradient.temperature_at,
            omega          = omega,
        )

        gr = sim.run(
            t_end          = t_end,
            n_trajectories = n_trajectories,
            n_samples      = n_samples,
            n_segments     = n_segments,
            tau_eps        = tau_eps,
            seed           = seed,
        )
        issues.extend(gr.issues)

        # Use ensemble mean as the deterministic-compatible trajectory
        time_arr = gr.time                    # (T,)
        mean_arr = gr.mean                    # (T, M)

        # Final outlet from mean at last time point
        outlet: Dict[str, float] = {
            sp: float(mean_arr[-1, i]) for i, sp in enumerate(species_names)
        }
        self.outlet_composition = outlet

        conversion: Dict[str, float] = {}
        for sp, c_in in inlet.items():
            if c_in > 0 and sp in outlet:
                conversion[sp] = max(0.0, (c_in - outlet[sp]) / c_in)

        T_arr = np.array([self.temperature_gradient.temperature_at(t) for t in time_arr])

        self._result = CSTRResult(
            reactor_name        = self.name,
            time                = time_arr,
            concentrations      = mean_arr,          # mean trajectory for 2-D canvas
            species_names       = species_names,
            outlet_composition  = outlet,
            residence_time_s    = τ,
            conversion          = conversion,
            temperature_profile = T_arr,
            converged           = gr.converged,
            issues              = issues,
            simulation_mode     = "stochastic",
            stochastic_data     = gr.to_dict(),
        )
        return self._result

    # ── Result access ─────────────────────────────────────────────────────────

    @property
    def result(self) -> Optional[CSTRResult]:
        return self._result

    @property
    def feeds(self) -> List[FeedStream]:
        return list(self._feeds)

    def to_dict(self) -> dict:
        return {
            "name"                : self.name,
            "volume_L"            : self.volume_L,
            "residence_time_s"    : self.residence_time_s,
            "total_flow_Ls"       : self.total_flow_Ls,
            "temperature_gradient": self.temperature_gradient.to_dict(),
            "heat_transfer_coeff" : self.heat_transfer_coeff,
            "jacket_temperature_K": self.jacket_temperature_K,
            "feeds"               : [f.to_dict() for f in self._feeds],
            "outlet_composition"  : self.outlet_composition,
            "species"             : list(self.reaction_network.species_names),
            "reactions"           : [r.equation for r in self.reaction_network.reactions],
        }

    def __repr__(self) -> str:
        return (f"CSTR('{self.name}', V={self.volume_L}L, "
                f"τ={self.residence_time_min:.1f}min, "
                f"Q={self.total_flow_Ls*1e3:.2f} mL/s)")