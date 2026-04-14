"""
network/pipeline/connection.py  –  Pipeline connections (v2).

v2 changes:
  - Every Connection owns a PumpModel (per-pipe efficiency).
  - diagnose() returns pump ΔP and power in kPa / kW.
  - set_inlet_pressure() kept for legacy compatibility but pump solve
    now takes node pressures directly via PumpModel.solve().
  - Unset roughness/density flagged in diagnostic.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_proj_dir = "/mnt/project"
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

from transportsim.pipeline import Pipeline, PipelineSpec, FluidProperties, HydraulicState
from transportsim.pump import PumpModel, PumpState

P_ATM = 101_325.0


# ── Diagnostic ────────────────────────────────────────────────────────────────

@dataclass
class ConnectionDiagnostic:
    """Full diagnostic for a pipeline connection after solving."""
    pipeline_name         : str
    source                : str
    target                : str
    flow_rate_Ls          : float
    # Pipe hydraulics
    pipe_pressure_drop_kPa: float
    velocity_m_s          : float
    reynolds_number       : float
    flow_regime           : str
    head_loss_m           : float
    friction_factor_darcy : float
    friction_factor_fanning: float
    # Pump
    pump_delta_p_kPa      : float
    pump_power_kW         : float
    pump_efficiency       : float
    pump_feasible         : bool
    # Flags
    roughness_default     : bool
    density_default       : bool
    issues                : List[str]

    def to_dict(self) -> dict:
        return {
            "pipeline_name"          : self.pipeline_name,
            "source"                 : self.source,
            "target"                 : self.target,
            "flow_rate_Ls"           : round(self.flow_rate_Ls, 4),
            "pipe_pressure_drop_kPa" : round(self.pipe_pressure_drop_kPa, 3),
            "velocity_m_s"           : round(self.velocity_m_s, 3),
            "reynolds_number"        : round(self.reynolds_number, 0),
            "flow_regime"            : self.flow_regime,
            "head_loss_m"            : round(self.head_loss_m, 3),
            "friction_factor_darcy"  : round(self.friction_factor_darcy, 6),
            "friction_factor_fanning": round(self.friction_factor_fanning, 6),
            "pump_delta_p_kPa"       : round(self.pump_delta_p_kPa, 3),
            "pump_power_kW"          : round(self.pump_power_kW, 4),
            "pump_efficiency_pct"    : round(self.pump_efficiency * 100, 1),
            "pump_feasible"          : self.pump_feasible,
            "roughness_default"      : self.roughness_default,
            "density_default"        : self.density_default,
            "issues"                 : self.issues,
            # Legacy compat keys
            "pressure_drop_kPa"      : round(self.pipe_pressure_drop_kPa, 3),
            "needs_compressor"       : not self.pump_feasible,
            "compressor_kW"          : round(self.pump_power_kW, 4),
            "outlet_pressure_bar"    : 0.0,  # node-owned in v2
        }


# ── Connection ────────────────────────────────────────────────────────────────

class Connection:
    """
    A directed flow connection between two reactor/SourceSink nodes.

    Every Connection has a PumpModel at its inlet.

    Parameters
    ----------
    pipeline       : Pipeline
    flow_fraction  : float  fraction of upstream flow through this pipe
    pump_efficiency: float  per-pipe pump efficiency (default 0.75)
    """

    def __init__(
        self,
        pipeline       : Pipeline,
        flow_fraction  : float = 1.0,
        pump_efficiency: float = 0.75,
    ):
        self.pipeline       = pipeline
        self.flow_fraction  = flow_fraction
        self.pump           = PumpModel(pipeline.name, efficiency=pump_efficiency)
        self._hyd_state: Optional[HydraulicState] = None
        self._diag: Optional[ConnectionDiagnostic] = None

    # ── Solve (called by NetworkPressureSolver) ───────────────────────────────

    def solve_pump(
        self,
        node_P_inlet_Pa : float = P_ATM,
        node_P_outlet_Pa: float = P_ATM,
    ) -> PumpState:
        """
        Solve pump ΔP for given node pressures. Returns PumpState.
        Also computes hydraulic state for diagnostics.
        """
        pump_state = self.pump.solve(
            geom             = self.pipeline.spec,
            flow_rate_m3s    = self.pipeline.flow_rate_m3s,
            node_P_inlet_Pa  = node_P_inlet_Pa,
            node_P_outlet_Pa = node_P_outlet_Pa,
            fluid            = self.pipeline.fluid,
        )
        # Also compute hydraulic state for non-pump metrics
        self._hyd_state = self.pipeline.compute(inlet_pressure_Pa=node_P_inlet_Pa)
        return pump_state

    def compute(self) -> HydraulicState:
        """Legacy: run hydraulic calculation without pump solve."""
        self._hyd_state = self.pipeline.compute()
        return self._hyd_state

    def flow_sweep(
        self,
        q_min: float = 1e-6,
        q_max: float = 1e-2,
        n_points: int = 60,
        node_P_inlet_Pa: float = P_ATM,
        node_P_outlet_Pa: float = P_ATM,
    ) -> dict:
        """Pump curve sweep: ΔP + power vs flow rate."""
        return self.pipeline.flow_sweep(
            q_min, q_max, n_points,
            node_P_inlet_Pa=node_P_inlet_Pa,
            node_P_outlet_Pa=node_P_outlet_Pa,
            efficiency=self.pump.efficiency,
        )

    def set_flow(self, flow_rate_m3s: float) -> None:
        self.pipeline.flow_rate_m3s = flow_rate_m3s

    def set_inlet_pressure(self, pressure_Pa: float) -> None:
        """Legacy: store on pipeline object for compatibility."""
        self.pipeline.inlet_pressure_Pa = pressure_Pa

    def set_pump_efficiency(self, efficiency: float) -> None:
        if not 0 < efficiency <= 1.0:
            raise ValueError("Pump efficiency must be in (0, 1]")
        self.pump.efficiency = efficiency

    # ── Diagnose ──────────────────────────────────────────────────────────────

    def diagnose(
        self,
        node_P_inlet_Pa : float = P_ATM,
        node_P_outlet_Pa: float = P_ATM,
    ) -> ConnectionDiagnostic:
        """Solve pump + compute hydraulics, return structured diagnostic."""
        pump_state = self.solve_pump(node_P_inlet_Pa, node_P_outlet_Pa)
        hyd        = self._hyd_state or self.pipeline.compute(node_P_inlet_Pa)

        roughness_default = not self.pipeline.spec.roughness_user_set
        density_default   = not self.pipeline.fluid.density_user_set

        issues = list(hyd.issues)
        if not pump_state.feasible:
            issues.append(f"⚠ Pump ΔP {pump_state.required_delta_p_kPa:.1f} kPa — "
                          "check pipe geometry or reduce flow rate.")

        self._diag = ConnectionDiagnostic(
            pipeline_name          = self.pipeline.name,
            source                 = self.pipeline.source,
            target                 = self.pipeline.target,
            flow_rate_Ls           = self.pipeline.flow_rate_m3s * 1000.0,
            pipe_pressure_drop_kPa = pump_state.pressure_drop_kPa,
            velocity_m_s           = pump_state.velocity_m_s,
            reynolds_number        = pump_state.reynolds_number,
            flow_regime            = pump_state.regime,
            head_loss_m            = pump_state.head_loss_m,
            friction_factor_darcy  = pump_state.friction_factor,
            friction_factor_fanning= pump_state.fanning_factor,
            pump_delta_p_kPa       = pump_state.required_delta_p_kPa,
            pump_power_kW          = pump_state.shaft_power_kW,
            pump_efficiency        = pump_state.efficiency,
            pump_feasible          = pump_state.feasible,
            roughness_default      = roughness_default,
            density_default        = density_default,
            issues                 = issues,
        )
        return self._diag

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self.pipeline.name

    @property
    def source(self) -> str:
        return self.pipeline.source

    @property
    def target(self) -> str:
        return self.pipeline.target

    @property
    def state(self) -> Optional[HydraulicState]:
        return self._hyd_state

    @property
    def pump_power_kW(self) -> float:
        return self.pump.shaft_power_kW

    @property
    def pump_delta_p_kPa(self) -> float:
        return self.pump.required_delta_p_kPa

    def to_dict(self) -> dict:
        d = self.pipeline.to_dict()
        d["flow_fraction"]   = self.flow_fraction
        d["pump_efficiency"] = self.pump.efficiency
        d["pump"]            = self.pump.to_dict()
        if self._diag:
            d["diagnostic"]  = self._diag.to_dict()
        return d

    def __repr__(self) -> str:
        q_Ls = self.pipeline.flow_rate_m3s * 1000.0
        return (f"Connection('{self.name}', {q_Ls:.3f} L/s, "
                f"pump η={self.pump.efficiency:.0%}, "
                f"ΔP={self.pump.required_delta_p_kPa:.1f} kPa)")
