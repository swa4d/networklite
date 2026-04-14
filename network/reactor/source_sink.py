"""
network/reactor/source_sink.py  –  SourceSink node (v2).

A SourceSink is a first-class network node identical to a CSTR from the
solver's perspective, with two key differences:
  - volume = infinity  (never runs dry, no residence-time constraint)
  - pressure can be fixed by the user or left at P_ATM

Sources and sinks are treated identically by TransportSim's pressure solver
and by the mass-balance tool. The node_type field ("source" | "sink") is
purely a UI rendering hint.

This unifies all four previous special-cases:
  - Source nodes feeding into a CSTR
  - Sink nodes draining from a CSTR
  - Source/sink exclusion from mass balance (FIXED)
  - Source/sink exclusion from pressure sweep tools (FIXED)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_project_dir = "/mnt/project"
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

try:
    from chemsim.network import ReactionNetwork
    from chemsim.simulator import Simulator
    _HAS_CHEMSIM = True
except ImportError:
    _HAS_CHEMSIM = False
    ReactionNetwork = None
    Simulator = None

P_ATM = 101_325.0   # Pa


# ── Result stub (mirrors CSTRResult interface) ────────────────────────────────

@dataclass
class SourceSinkResult:
    """
    Simulation result for a SourceSink node.

    A SourceSink has no reactions, so concentrations are fixed by its
    configured species composition.
    """
    node_name          : str
    node_type          : str                    # "source" | "sink"
    species_names      : List[str]              = field(default_factory=list)
    concentrations     : np.ndarray             = field(default_factory=lambda: np.zeros((1, 0)))
    time               : np.ndarray             = field(default_factory=lambda: np.zeros(1))
    outlet_composition : Dict[str, float]       = field(default_factory=dict)
    pressure_Pa        : float                  = P_ATM
    pressure_kPa       : float                  = P_ATM / 1000
    flow_rate_m3s      : float                  = 0.0
    converged          : bool                   = True
    issues             : List[str]              = field(default_factory=list)

    @property
    def conversion(self) -> Dict[str, float]:
        return {}   # SourceSinks have no conversion

    def to_dict(self) -> dict:
        return {
            "node_name"         : self.node_name,
            "node_type"         : self.node_type,
            "species_names"     : self.species_names,
            "outlet_composition": {k: round(v, 6) for k, v in self.outlet_composition.items()},
            "pressure_kPa"      : round(self.pressure_kPa, 3),
            "flow_rate_Ls"      : round(self.flow_rate_m3s * 1000, 4),
            "converged"         : self.converged,
            "issues"            : self.issues,
            "concentrations"    : self.concentrations.tolist(),
            "time"              : self.time.tolist(),
        }


# ── SourceSink node ───────────────────────────────────────────────────────────

class SourceSink:
    """
    A source or sink node in the plant network.

    Behaves like a CSTR with infinite volume and no reactions.
    Pressure is constant (user-set or P_ATM default).

    Parameters
    ----------
    name        : str   unique node name
    node_type   : str   "source" or "sink"
    pressure_Pa : float operating pressure in Pa (default P_ATM)
    pressure_mode : str "fixed" (user-set) | "atm" (default)
    species     : dict  {species_name: concentration_M}  (source composition)
    flow_rate_m3s : float  default flow rate in m³/s
    """

    def __init__(
        self,
        name           : str,
        node_type      : str   = "source",
        pressure_Pa    : float = P_ATM,
        pressure_mode  : str   = "atm",
        species        : Optional[Dict[str, float]] = None,
        flow_rate_m3s  : float = 5e-4,
    ):
        if node_type not in ("source", "sink"):
            raise ValueError(f"node_type must be 'source' or 'sink', got '{node_type}'")
        self.name          = name
        self.node_type     = node_type
        self.pressure_Pa   = pressure_Pa
        self.pressure_mode = pressure_mode   # "fixed" | "atm"
        self.species       = dict(species) if species else {}
        self.flow_rate_m3s = flow_rate_m3s
        self.volume_L      = float("inf")    # infinite capacity
        self.result: Optional[SourceSinkResult] = None

        # Build a minimal ReactionNetwork so the CSTR interface stays compatible
        if _HAS_CHEMSIM and self.species:
            self._build_network()
        else:
            self.reaction_network = ReactionNetwork() if _HAS_CHEMSIM else None

    def _build_network(self):
        rn = ReactionNetwork()
        for sp, conc in self.species.items():
            rn.add_species(sp, initial=max(conc, 0.0))
        self.reaction_network = rn

    # ── Node interface (mirrors CSTR) ─────────────────────────────────────────

    @property
    def pressure_kPa(self) -> float:
        return self.pressure_Pa / 1000.0

    def set_pressure(self, pressure_Pa: float) -> None:
        self.pressure_Pa   = pressure_Pa
        self.pressure_mode = "fixed"

    def set_pressure_kPa(self, pressure_kPa: float) -> None:
        self.set_pressure(pressure_kPa * 1000.0)

    def add_species(self, name: str, concentration: float = 0.0) -> None:
        self.species[name] = concentration
        if self.reaction_network and name not in self.reaction_network.species_names:
            try:
                self.reaction_network.add_species(name, initial=concentration)
            except Exception:
                pass

    def set_species(self, species: Dict[str, float]) -> None:
        self.species = dict(species)
        self._build_network()

    @property
    def feeds(self):
        """Stub: SourceSinks have no feeds in the CSTR sense."""
        return []

    @property
    def total_flow_m3s(self) -> float:
        return self.flow_rate_m3s

    @property
    def total_flow_Ls(self) -> float:
        return self.flow_rate_m3s * 1000.0

    @property
    def outlet_composition(self) -> Dict[str, float]:
        return dict(self.species)

    @property
    def residence_time_s(self) -> float:
        return float("inf")

    # ── Simulate (no-op for SourceSink) ──────────────────────────────────────

    def simulate(
        self,
        t_end        : float = 300.0,
        n_segments   : int   = 10,
        solver_params: Optional[dict] = None,
    ) -> SourceSinkResult:
        """
        SourceSink simulation: returns fixed composition at all time points.
        No ODE integration needed — composition is constant.
        """
        times = np.linspace(0.0, t_end, max(n_segments, 2))
        n_sp  = len(self.species)
        concs = np.zeros((len(times), n_sp))
        names = list(self.species.keys())
        vals  = list(self.species.values())
        for j, v in enumerate(vals):
            concs[:, j] = v

        issues = []
        if self.pressure_mode == "atm":
            issues.append(
                f"Pressure not explicitly set — using atmospheric default "
                f"({P_ATM/1000:.1f} kPa). Set node pressure for accurate pump sizing."
            )
        if not self.species and self.node_type == "source":
            issues.append("No species defined on source node — no composition injected.")

        self.result = SourceSinkResult(
            node_name          = self.name,
            node_type          = self.node_type,
            species_names      = names,
            concentrations     = concs,
            time               = times,
            outlet_composition = dict(self.species),
            pressure_Pa        = self.pressure_Pa,
            pressure_kPa       = self.pressure_kPa,
            flow_rate_m3s      = self.flow_rate_m3s,
            converged          = True,
            issues             = issues,
        )
        return self.result

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name"          : self.name,
            "node_type"     : self.node_type,
            "volume_L"      : None,           # infinite — do not display
            "pressure_Pa"   : self.pressure_Pa,
            "pressure_kPa"  : round(self.pressure_kPa, 3),
            "pressure_mode" : self.pressure_mode,
            "species"       : self.species,
            "flow_rate_m3s" : self.flow_rate_m3s,
            "flow_rate_Ls"  : round(self.flow_rate_m3s * 1000, 4),
            "result"        : self.result.to_dict() if self.result else None,
        }

    def inspect(self) -> dict:
        """Extended dict for dashboard inspector panel."""
        d = self.to_dict()
        d["pressure_flag"] = (self.pressure_mode == "atm")
        d["species_flag"]  = (not self.species and self.node_type == "source")
        return d

    def __repr__(self) -> str:
        sp_str = ", ".join(f"{k}={v}" for k, v in self.species.items())
        return (f"SourceSink('{self.name}' [{self.node_type}], "
                f"P={self.pressure_kPa:.1f} kPa, species=[{sp_str}])")
