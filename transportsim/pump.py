"""
transportsim/pump.py  –  Pump model for TransportSim v2.

Every pipeline connection has an implicit pump at its inlet that
provides the pressure differential needed to move fluid at the
user-specified volumetric flow rate.

Key classes:
    PumpModel   – per-pipe pump with efficiency, ΔP, and power tracking
    PumpTimeSeries – power-demand integration over a simulation time span
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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

P_ATM = 101_325.0  # Pa


# ── PumpState ─────────────────────────────────────────────────────────────────

@dataclass
class PumpState:
    """
    Current solved state of a pump on a pipe.

    All pressures in kPa for user-facing output.
    """
    required_delta_p_kPa : float = 0.0
    required_delta_p_Pa  : float = 0.0
    shaft_power_kW       : float = 0.0
    flow_rate_m3s        : float = 0.0
    efficiency           : float = 0.75
    feasible             : bool  = True
    warning              : str   = ""
    regime               : str   = "turbulent"
    reynolds_number      : float = 0.0
    velocity_m_s         : float = 0.0
    friction_factor      : float = 0.0
    fanning_factor       : float = 0.0
    head_loss_m          : float = 0.0
    pressure_drop_kPa    : float = 0.0  # pipe ΔP only (excl. node correction)

    def to_dict(self) -> dict:
        return {
            "required_delta_p_kPa": round(self.required_delta_p_kPa, 3),
            "shaft_power_kW"      : round(self.shaft_power_kW, 4),
            "flow_rate_Ls"        : round(self.flow_rate_m3s * 1000, 4),
            "efficiency_pct"      : round(self.efficiency * 100, 1),
            "feasible"            : self.feasible,
            "warning"             : self.warning,
            "regime"              : self.regime,
            "reynolds_number"     : round(self.reynolds_number, 0),
            "velocity_m_s"        : round(self.velocity_m_s, 3),
            "friction_factor_darcy": round(self.friction_factor, 6),
            "friction_factor_fanning": round(self.fanning_factor, 6),
            "head_loss_m"         : round(self.head_loss_m, 3),
            "pipe_pressure_drop_kPa": round(self.pressure_drop_kPa, 3),
        }


# ── PumpTimeSeries ────────────────────────────────────────────────────────────

@dataclass
class PumpTimeSeries:
    """
    Power demand of a pump integrated over a simulation time span.

    When reactor simulation produces time-varying outlet concentrations,
    the fluid density can change, affecting pump power. This class holds
    the time-resolved power profile.
    """
    pipe_name    : str
    times        : np.ndarray         # seconds
    power_kW     : np.ndarray         # kW at each time
    flow_m3s     : float              # constant (per design)
    delta_p_kPa  : float              # constant ΔP (per auto-balance solve)

    @property
    def peak_power_kW(self) -> float:
        return float(np.max(self.power_kW)) if len(self.power_kW) else 0.0

    @property
    def mean_power_kW(self) -> float:
        return float(np.mean(self.power_kW)) if len(self.power_kW) else 0.0

    @property
    def energy_kWh(self) -> float:
        """Integrate power over time to get energy in kWh."""
        if len(self.times) < 2:
            return 0.0
        return float(np.trapz(self.power_kW, self.times)) / 3600.0

    def to_dict(self) -> dict:
        return {
            "pipe_name"   : self.pipe_name,
            "times"       : self.times.tolist(),
            "power_kW"    : self.power_kW.tolist(),
            "peak_kW"     : round(self.peak_power_kW, 4),
            "mean_kW"     : round(self.mean_power_kW, 4),
            "energy_kWh"  : round(self.energy_kWh, 6),
            "flow_Ls"     : round(self.flow_m3s * 1000, 4),
            "delta_p_kPa" : round(self.delta_p_kPa, 3),
        }


# ── PumpModel ─────────────────────────────────────────────────────────────────

class PumpModel:
    """
    Virtual pump at the inlet of a pipeline.

    In v2, every Connection has a PumpModel. The pump is sized by
    NetworkPressureSolver (auto-balance) to provide the ΔP needed to
    achieve the user-specified flow rate between two constant-pressure nodes.

    Parameters
    ----------
    pipe_name  : str
        Name of the owning pipeline.
    efficiency : float
        Isentropic/pump efficiency (0–1). Default 0.75. Per-pipe.
    """

    def __init__(self, pipe_name: str, efficiency: float = 0.75):
        if not 0 < efficiency <= 1.0:
            raise ValueError(f"Pump efficiency must be in (0, 1], got {efficiency}")
        self.pipe_name  = pipe_name
        self.efficiency = efficiency
        self._state: Optional[PumpState] = None

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(
        self,
        geom,           # PipelineSpec or dict
        flow_rate_m3s   : float,
        node_P_inlet_Pa : float = P_ATM,
        node_P_outlet_Pa: float = P_ATM,
        fluid           = None,  # FluidProperties or dict
    ) -> PumpState:
        """
        Run the auto-balance solve via TransportSim C++ core.

        Sets self._state and returns it.

        Parameters
        ----------
        geom             : PipelineSpec-like
        flow_rate_m3s    : float  target flow rate
        node_P_inlet_Pa  : float  source node operating pressure (Pa)
        node_P_outlet_Pa : float  target node operating pressure (Pa)
        fluid            : FluidProperties-like (optional)
        """
        if _HAS_CORE:
            return self._solve_cpp(geom, flow_rate_m3s, node_P_inlet_Pa, node_P_outlet_Pa, fluid)
        else:
            return self._solve_python(geom, flow_rate_m3s, node_P_inlet_Pa, node_P_outlet_Pa, fluid)

    def _solve_cpp(self, geom, flow_rate_m3s, node_P_inlet_Pa, node_P_outlet_Pa, fluid) -> PumpState:
        g = _core.PipelineGeometry()
        g.length           = geom.length
        g.diameter         = geom.diameter
        g.roughness        = geom.roughness
        g.elevation_change = geom.elevation_change
        g.n_fittings_K     = int(geom.n_fittings_K)

        f = _core.FluidProps()
        if fluid:
            f.density   = fluid.density
            f.viscosity = fluid.viscosity
            f.phase     = getattr(fluid, "phase", "liquid")

        sol = _core.solve_pump_delta_p(
            g, flow_rate_m3s, node_P_inlet_Pa, node_P_outlet_Pa, f, self.efficiency
        )

        # Also compute hydraulics for regime/Re info
        cond = _core.FlowConditions()
        cond.flow_rate_m3s  = flow_rate_m3s
        cond.inlet_pressure = node_P_inlet_Pa
        hyd = _core.compute_hydraulics(g, cond, f)

        regime_s = _core.classify_regime(hyd.reynolds_number)

        state = PumpState(
            required_delta_p_kPa = sol.required_delta_p_kPa,
            required_delta_p_Pa  = sol.required_delta_p_Pa,
            shaft_power_kW       = sol.shaft_power_kW,
            flow_rate_m3s        = flow_rate_m3s,
            efficiency           = self.efficiency,
            feasible             = sol.feasible,
            warning              = sol.message if not sol.feasible else hyd.warning,
            regime               = regime_s.regime,
            reynolds_number      = hyd.reynolds_number,
            velocity_m_s         = hyd.velocity_m_s,
            friction_factor      = hyd.friction_factor,
            fanning_factor       = hyd.fanning_factor,
            head_loss_m          = hyd.head_loss_m,
            pressure_drop_kPa    = hyd.pressure_drop_kPa,
        )
        self._state = state
        return state

    def _solve_python(self, geom, flow_rate_m3s, node_P_inlet_Pa, node_P_outlet_Pa, fluid) -> PumpState:
        """Pure-Python fallback when C++ core is not built."""
        import math

        D   = geom.diameter
        L   = geom.length
        rho = fluid.density if fluid else 1000.0
        mu  = fluid.viscosity if fluid else 1.002e-3

        A   = math.pi * (D / 2) ** 2
        v   = flow_rate_m3s / A if A > 0 else 0.0
        Re  = rho * v * D / mu if mu > 0 else 0.0
        eps_D = geom.roughness / D

        if Re < 2300:
            f = 64.0 / max(Re, 1.0)
            regime = "laminar"
        elif Re < 4000:
            f = 0.02
            regime = "transitional"
        else:
            f = 0.02
            for _ in range(30):
                rhs = -2.0 * math.log10(eps_D / 3.7 + 2.51 / (max(Re, 1.0) * math.sqrt(f)))
                f_new = 1.0 / rhs ** 2
                if abs(f_new - f) < 1e-8:
                    break
                f = f_new
            regime = "turbulent"

        dP_f  = f * (L / D) * 0.5 * rho * v ** 2
        dP_m  = geom.n_fittings_K * 0.5 * rho * v ** 2
        dP_g  = rho * 9.80665 * geom.elevation_change
        dP    = dP_f + dP_m + dP_g

        node_dP   = node_P_outlet_Pa - node_P_inlet_Pa
        pump_dP   = max(dP + node_dP, 0.0)
        pump_dP_kPa = pump_dP / 1000.0
        power_kW  = (flow_rate_m3s * pump_dP / (self.efficiency * 1000.0)) if pump_dP > 0 else 0.0
        feasible  = pump_dP_kPa <= 5000.0

        state = PumpState(
            required_delta_p_kPa = pump_dP_kPa,
            required_delta_p_Pa  = pump_dP,
            shaft_power_kW       = power_kW,
            flow_rate_m3s        = flow_rate_m3s,
            efficiency           = self.efficiency,
            feasible             = feasible,
            warning              = "" if feasible else f"Pump ΔP={pump_dP_kPa:.1f} kPa exceeds 5000 kPa",
            regime               = regime,
            reynolds_number      = Re,
            velocity_m_s         = v,
            friction_factor      = f,
            fanning_factor       = f / 4.0,
            head_loss_m          = dP / (rho * 9.80665) if rho > 0 else 0.0,
            pressure_drop_kPa    = dP / 1000.0,
        )
        self._state = state
        return state

    # ── Time series ───────────────────────────────────────────────────────────

    def power_demand_over_time(
        self,
        t_span      : Tuple[float, float],
        n_points    : int   = 100,
        density_fn  = None,  # optional callable: t → density (kg/m³)
        geom        = None,
        base_flow   : float = None,
        node_P_inlet_Pa : float = P_ATM,
        node_P_outlet_Pa: float = P_ATM,
    ) -> PumpTimeSeries:
        """
        Compute pump power demand over a time span.

        If density_fn is supplied (e.g., interpolated from reactor simulation),
        the pump ΔP is recalculated at each time point as fluid density changes.
        Otherwise, constant power from the last solve is used.

        Parameters
        ----------
        t_span      : (t_start, t_end) in seconds
        n_points    : number of time points
        density_fn  : optional callable t → density kg/m³
        geom        : PipelineSpec (required if density_fn supplied)
        base_flow   : flow rate m³/s (defaults to last solved value)
        node_P_inlet_Pa  : source node pressure Pa
        node_P_outlet_Pa : target node pressure Pa

        Returns
        -------
        PumpTimeSeries
        """
        times  = np.linspace(t_span[0], t_span[1], n_points)
        Q      = base_flow or (self._state.flow_rate_m3s if self._state else 1e-4)
        dP_kPa = self._state.required_delta_p_kPa if self._state else 0.0

        if density_fn is not None and geom is not None:
            # Recalculate power at each time step using time-varying density
            powers = np.zeros(n_points)
            for i, t in enumerate(times):
                rho = float(density_fn(t))
                from .pipeline import FluidProperties
                fluid = FluidProperties(density=rho)
                state = self._solve_python(geom, Q, node_P_inlet_Pa, node_P_outlet_Pa, fluid)
                powers[i] = state.shaft_power_kW
                dP_kPa    = state.required_delta_p_kPa
        else:
            # Constant power
            power = self._state.shaft_power_kW if self._state else 0.0
            powers = np.full(n_points, power)

        return PumpTimeSeries(
            pipe_name   = self.pipe_name,
            times       = times,
            power_kW    = powers,
            flow_m3s    = Q,
            delta_p_kPa = dP_kPa,
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> Optional[PumpState]:
        return self._state

    @property
    def shaft_power_kW(self) -> float:
        return self._state.shaft_power_kW if self._state else 0.0

    @property
    def required_delta_p_kPa(self) -> float:
        return self._state.required_delta_p_kPa if self._state else 0.0

    def to_dict(self) -> dict:
        d = {
            "pipe_name" : self.pipe_name,
            "efficiency": self.efficiency,
        }
        if self._state:
            d.update(self._state.to_dict())
        return d

    def __repr__(self) -> str:
        if self._state:
            return (f"PumpModel('{self.pipe_name}', η={self.efficiency:.0%}, "
                    f"ΔP={self._state.required_delta_p_kPa:.1f} kPa, "
                    f"P={self._state.shaft_power_kW:.3f} kW)")
        return f"PumpModel('{self.pipe_name}', η={self.efficiency:.0%}, unsolved)"
