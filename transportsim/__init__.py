"""
transportsim  –  Pipeline hydraulics and pump auto-balance engine (v2).

Public API
----------
Pipeline, PipelineSpec, FluidProperties, HydraulicState
PumpModel, PumpState, PumpTimeSeries
NetworkPressureSolver, PressureBalanceSolution
FlowRegimeSummary helpers
Sweep / plotting functions
"""

from .pipeline import (
    Pipeline,
    PipelineSpec,
    FluidProperties,
    HydraulicState,
    P_ATM,
    DENSITY_WATER,
    VISCOSITY_WATER,
    ROUGHNESS_STEEL,
)

from .pump import (
    PumpModel,
    PumpState,
    PumpTimeSeries,
)

from .pressure_solver import (
    NetworkPressureSolver,
    PressureBalanceSolution,
)

from .flow_regimes import (
    classify_regime,
    regime_summary,
    fanning_to_darcy,
    darcy_to_fanning,
)

from .sweep import (
    plot_pressure_sweep,
    plot_pump_operating_curve,
    plot_flow_regime_map,
    plot_fanning_vs_darcy,
    plot_pressure_breakdown,
    plot_pump_power_over_time,
)

__version__ = "2.0.0"
__all__ = [
    "Pipeline", "PipelineSpec", "FluidProperties", "HydraulicState",
    "PumpModel", "PumpState", "PumpTimeSeries",
    "NetworkPressureSolver", "PressureBalanceSolution",
    "classify_regime", "regime_summary", "fanning_to_darcy", "darcy_to_fanning",
    "plot_pressure_sweep", "plot_pump_operating_curve", "plot_flow_regime_map",
    "plot_fanning_vs_darcy", "plot_pressure_breakdown", "plot_pump_power_over_time",
    "P_ATM", "DENSITY_WATER", "VISCOSITY_WATER", "ROUGHNESS_STEEL",
]
