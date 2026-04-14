"""
transportsim/pipeline.py  –  Python pipeline API (v2).

High-level interface over the _transportsim_core C++ extension.
v2: Pipeline no longer manages inlet_pressure as an input parameter.
Instead, pressure is owned by nodes. The Pipeline's job is geometry
and fluid properties; the PumpModel handles ΔP solving.

All user-facing pressure values in kPa.
"""

from __future__ import annotations

import sys
import os
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

_ts_dir = os.path.dirname(os.path.abspath(__file__))
if _ts_dir not in sys.path:
    sys.path.insert(0, _ts_dir)

try:
    import _transportsim_core as _core
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False
    _core = None

# ── Constants ─────────────────────────────────────────────────────────────────

P_ATM           = 101_325.0
DENSITY_WATER   = 1000.0
VISCOSITY_WATER = 1.002e-3
ROUGHNESS_STEEL = 4.6e-5

_ROUGHNESS_UNSET_SENTINEL = ROUGHNESS_STEEL  # same value — flagged in UI only


@dataclass
class PipelineSpec:
    """
    Physical geometry of a pipeline segment.

    Parameters
    ----------
    length           : float  metres
    diameter         : float  inner diameter, metres (default 5 cm)
    roughness        : float  wall roughness, metres (default: commercial steel)
    elevation_change : float  Δz metres, positive = uphill
    n_fittings_K     : int    sum of K-values for minor losses
    roughness_user_set : bool  False = user never explicitly set roughness (UI warning)
    """
    length             : float = 10.0
    diameter           : float = 0.05
    roughness          : float = ROUGHNESS_STEEL
    elevation_change   : float = 0.0
    n_fittings_K       : int   = 2
    roughness_user_set : bool  = False


@dataclass
class FluidProperties:
    """
    Fluid physical properties.

    density_user_set / viscosity_user_set track whether the engineer
    has explicitly set these or whether we are using defaults.
    """
    density            : float = DENSITY_WATER
    viscosity          : float = VISCOSITY_WATER
    phase              : str   = "liquid"
    density_user_set   : bool  = False
    viscosity_user_set : bool  = False


@dataclass
class HydraulicState:
    """Complete hydraulic state of a pipeline (v2)."""
    pressure_drop_Pa    : float = 0.0
    pressure_drop_kPa   : float = 0.0
    outlet_pressure_Pa  : float = P_ATM
    velocity_m_s        : float = 0.0
    reynolds_number     : float = 0.0
    friction_factor     : float = 0.0
    fanning_factor      : float = 0.0
    head_loss_m         : float = 0.0
    minor_loss_Pa       : float = 0.0
    gravity_loss_Pa     : float = 0.0
    flow_regime         : str   = "turbulent"
    warning             : str   = ""
    issues              : List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class Pipeline:
    """
    A directed pipeline connecting two nodes.

    v2: inlet_pressure is not an input — it comes from the source node.
    PumpModel handles all ΔP solving. This class stores geometry/fluid
    and provides flow_sweep for sweep chart data.

    Parameters
    ----------
    name          : str   unique identifier, e.g. "R1→R2"
    source        : str   upstream node name
    target        : str   downstream node name
    spec          : PipelineSpec
    fluid         : FluidProperties
    flow_rate_m3s : float  user-specified volumetric flow rate
    """

    def __init__(
        self,
        name          : str,
        source        : str,
        target        : str,
        spec          : Optional[PipelineSpec]    = None,
        fluid         : Optional[FluidProperties] = None,
        flow_rate_m3s : float = 1e-4,
        # Legacy param — kept for backward-compat but not used in v2 solving
        inlet_pressure_Pa: float = P_ATM,
    ):
        self.name              = name
        self.source            = source
        self.target            = target
        self.spec              = spec  or PipelineSpec()
        self.fluid             = fluid or FluidProperties()
        self.flow_rate_m3s     = flow_rate_m3s
        self.inlet_pressure_Pa = inlet_pressure_Pa  # legacy; node owns this in v2
        self._state: Optional[HydraulicState] = None

    # ── Hydraulic compute (for diagnostics, not for pump solve) ───────────────

    def compute(self, inlet_pressure_Pa: float = None) -> HydraulicState:
        """
        Compute hydraulic state at the current flow rate.

        In v2 this is used for display purposes (velocity, Re, etc.).
        The pump solve (ΔP, power) is done by PumpModel.
        """
        P_in = inlet_pressure_Pa or self.inlet_pressure_Pa
        if _HAS_CORE:
            state = self._compute_cpp(P_in)
        else:
            state = self._compute_python(P_in)
        self._state = state
        return state

    def _compute_cpp(self, inlet_pressure_Pa: float) -> HydraulicState:
        geom           = _core.PipelineGeometry()
        geom.length            = self.spec.length
        geom.diameter          = self.spec.diameter
        geom.roughness         = self.spec.roughness
        geom.elevation_change  = self.spec.elevation_change
        geom.n_fittings_K      = int(self.spec.n_fittings_K)

        fluid           = _core.FluidProps()
        fluid.density   = self.fluid.density
        fluid.viscosity = self.fluid.viscosity
        fluid.phase     = self.fluid.phase

        cond                = _core.FlowConditions()
        cond.flow_rate_m3s  = self.flow_rate_m3s
        cond.inlet_pressure = inlet_pressure_Pa

        r = _core.compute_hydraulics(geom, cond, fluid)

        issues = []
        if not r.flow_is_turbulent:
            issues.append(
                f"Flow is {r.regime} (Re={r.reynolds_number:.0f}). "
                "Turbulent assumption may not hold — ΔP estimate is approximate."
            )
        if r.velocity_m_s > 5.0:
            issues.append(f"High velocity {r.velocity_m_s:.1f} m/s — check erosion limits.")
        if not self.spec.roughness_user_set:
            issues.append("Roughness not explicitly set — using commercial steel default (4.6×10⁻⁵ m).")
        if not self.fluid.density_user_set:
            issues.append("Fluid density not set — using water default (1000 kg/m³).")

        return HydraulicState(
            pressure_drop_Pa   = r.pressure_drop_Pa,
            pressure_drop_kPa  = r.pressure_drop_kPa,
            outlet_pressure_Pa = r.outlet_pressure_Pa,
            velocity_m_s       = r.velocity_m_s,
            reynolds_number    = r.reynolds_number,
            friction_factor    = r.friction_factor,
            fanning_factor     = r.fanning_factor,
            head_loss_m        = r.head_loss_m,
            minor_loss_Pa      = r.minor_loss_Pa,
            gravity_loss_Pa    = r.gravity_loss_Pa,
            flow_regime        = r.regime,
            warning            = r.warning,
            issues             = issues,
        )

    def _compute_python(self, inlet_pressure_Pa: float) -> HydraulicState:
        D  = self.spec.diameter
        L  = self.spec.length
        rho = self.fluid.density
        mu  = self.fluid.viscosity

        A   = np.pi * (D / 2) ** 2
        v   = self.flow_rate_m3s / A if A > 0 else 0.0
        Re  = rho * v * D / mu if mu > 0 else 0.0
        eps_D = self.spec.roughness / D

        if Re < 2300:
            f = 64.0 / max(Re, 1.0)
        else:
            f = 0.02
            for _ in range(30):
                rhs = -2.0 * np.log10(eps_D / 3.7 + 2.51 / (max(Re, 1.0) * np.sqrt(f)))
                f_new = 1.0 / rhs ** 2
                if abs(f_new - f) < 1e-8:
                    break
                f = f_new

        dP_f  = f * (L / D) * 0.5 * rho * v ** 2
        dP_m  = self.spec.n_fittings_K * 0.5 * rho * v ** 2
        dP_g  = rho * 9.80665 * self.spec.elevation_change
        dP    = dP_f + dP_m + dP_g
        P_out = inlet_pressure_Pa - dP
        regime = "laminar" if Re < 2300 else ("transitional" if Re < 4000 else "turbulent")

        issues = []
        if regime != "turbulent":
            issues.append(f"Flow is {regime} (Re={Re:.0f}).")
        if not self.spec.roughness_user_set:
            issues.append("Roughness not explicitly set — using commercial steel default.")
        if not self.fluid.density_user_set:
            issues.append("Fluid density not set — using water default (1000 kg/m³).")

        return HydraulicState(
            pressure_drop_Pa   = dP,
            pressure_drop_kPa  = dP / 1000.0,
            outlet_pressure_Pa = P_out,
            velocity_m_s       = v,
            reynolds_number    = Re,
            friction_factor    = f,
            fanning_factor     = f / 4.0,
            head_loss_m        = dP / (rho * 9.80665) if rho > 0 else 0.0,
            minor_loss_Pa      = dP_m,
            gravity_loss_Pa    = dP_g,
            flow_regime        = regime,
            warning            = "",
            issues             = issues,
        )

    # ── Flow sweep ────────────────────────────────────────────────────────────

    def flow_sweep(
        self,
        q_min: float = 1e-6,
        q_max: float = 1e-2,
        n_points: int = 60,
        node_P_inlet_Pa : float = P_ATM,
        node_P_outlet_Pa: float = P_ATM,
        efficiency: float = 0.75,
    ) -> Dict[str, list]:
        """
        Compute pump curve sweep: ΔP + power vs flow rate.

        Returns dict suitable for frontend charting.
        """
        if _HAS_CORE:
            geom           = _core.PipelineGeometry()
            geom.length    = self.spec.length
            geom.diameter  = self.spec.diameter
            geom.roughness = self.spec.roughness
            geom.elevation_change = self.spec.elevation_change
            geom.n_fittings_K    = int(self.spec.n_fittings_K)

            fluid           = _core.FluidProps()
            fluid.density   = self.fluid.density
            fluid.viscosity = self.fluid.viscosity

            sweep = _core.pump_curve_sweep(
                geom, q_min, q_max, n_points, fluid,
                node_P_inlet_Pa, node_P_outlet_Pa, efficiency
            )
            return {
                "flow_rates_Ls"      : [q * 1000 for q in sweep.flow_rates],
                "pressure_drops_kPa" : list(sweep.pressure_drops_kPa),
                "pump_delta_p_kPa"   : list(sweep.pump_delta_p_kPa),
                "pump_power_kW"      : list(sweep.pump_power_kW),
                "reynolds_numbers"   : list(sweep.reynolds_numbers),
                "friction_factors"   : list(sweep.friction_factors),
                "velocities_ms"      : list(sweep.velocities),
                "flow_is_turbulent"  : list(sweep.flow_is_turbulent),
            }
        else:
            qs = np.linspace(q_min, q_max, n_points)
            result = {
                "flow_rates_Ls": (qs * 1000).tolist(),
                "pressure_drops_kPa": [], "pump_delta_p_kPa": [],
                "pump_power_kW": [], "reynolds_numbers": [],
                "friction_factors": [], "velocities_ms": [],
                "flow_is_turbulent": [],
            }
            orig_q = self.flow_rate_m3s
            for q in qs:
                self.flow_rate_m3s = q
                s = self._compute_python(node_P_inlet_Pa)
                node_dP = max(node_P_outlet_Pa - node_P_inlet_Pa, 0.0)
                pump_dP = s.pressure_drop_Pa + node_dP
                power   = (q * pump_dP / (efficiency * 1000)) if pump_dP > 0 else 0.0
                result["pressure_drops_kPa"].append(s.pressure_drop_kPa)
                result["pump_delta_p_kPa"].append(pump_dP / 1000.0)
                result["pump_power_kW"].append(power)
                result["reynolds_numbers"].append(s.reynolds_number)
                result["friction_factors"].append(s.friction_factor)
                result["velocities_ms"].append(s.velocity_m_s)
                result["flow_is_turbulent"].append(s.flow_regime == "turbulent")
            self.flow_rate_m3s = orig_q
            return result

    # ── Properties / serialisation ────────────────────────────────────────────

    @property
    def state(self) -> Optional[HydraulicState]:
        return self._state

    def to_dict(self) -> dict:
        return {
            "name"            : self.name,
            "source"          : self.source,
            "target"          : self.target,
            "flow_rate_m3s"   : self.flow_rate_m3s,
            "flow_rate_Ls"    : round(self.flow_rate_m3s * 1000, 4),
            "spec"            : dataclasses.asdict(self.spec),
            "fluid"           : dataclasses.asdict(self.fluid),
            "state"           : self._state.to_dict() if self._state else None,
        }

    def __repr__(self) -> str:
        q_Ls = self.flow_rate_m3s * 1000.0
        return (f"Pipeline('{self.name}': {self.source}→{self.target}, "
                f"L={self.spec.length}m, D={self.spec.diameter*100:.1f}cm, "
                f"Q={q_Ls:.2f} L/s)")
