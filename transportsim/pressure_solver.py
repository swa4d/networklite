"""
transportsim/pressure_solver.py  –  Network-level pressure auto-balance (v2).

NetworkPressureSolver is the auto-balance engine.

Design assumptions:
  - Every node (CSTR, Source, Sink) operates at constant pressure.
  - Every pipe has a virtual pump at its inlet.
  - The pump ΔP = pipe friction losses + minor losses + gravity
                  + (outlet_node_P - inlet_node_P)
  - For cycles, we use a fixed-point iteration (≤ 30 iterations).

Output:
  PressureBalanceSolution  –  per-pipe PumpState + aggregate power summary.
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

_ts_dir = os.path.dirname(os.path.abspath(__file__))
if _ts_dir not in sys.path:
    sys.path.insert(0, _ts_dir)

from .pump import PumpModel, PumpState, PumpTimeSeries

P_ATM = 101_325.0  # Pa


# ── Solution container ────────────────────────────────────────────────────────

@dataclass
class PressureBalanceSolution:
    """
    Result of a NetworkPressureSolver run.

    Attributes
    ----------
    pump_states    : dict  pipe_name → PumpState
    total_power_kW : float  sum of all pump shaft powers
    converged      : bool   True if iterative solve converged
    n_iterations   : int    number of iterations taken
    warnings       : list   per-pipe or global warnings
    infeasible_pipes : list  pipe names with unreasonably high ΔP
    """
    pump_states      : Dict[str, PumpState] = field(default_factory=dict)
    total_power_kW   : float                = 0.0
    converged        : bool                 = True
    n_iterations     : int                  = 1
    warnings         : List[str]            = field(default_factory=list)
    infeasible_pipes : List[str]            = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pump_states"      : {k: v.to_dict() for k, v in self.pump_states.items()},
            "total_power_kW"   : round(self.total_power_kW, 4),
            "converged"        : self.converged,
            "n_iterations"     : self.n_iterations,
            "warnings"         : self.warnings,
            "infeasible_pipes" : self.infeasible_pipes,
        }

    # Per-pipe convenience
    def power_breakdown(self) -> Dict[str, float]:
        """Returns {pipe_name: shaft_power_kW} dict."""
        return {k: round(v.shaft_power_kW, 4) for k, v in self.pump_states.items()}


# ── Solver ────────────────────────────────────────────────────────────────────

class NetworkPressureSolver:
    """
    Auto-balance solver for a PlantNetwork.

    Iterates over all pipe connections, calling PumpModel.solve() for each,
    and handles cycles via fixed-point iteration.

    Parameters
    ----------
    max_iter : int    maximum iterations for cycle resolution (default 30)
    tol_kPa  : float  convergence tolerance in kPa (default 0.1 kPa)
    """

    def __init__(self, max_iter: int = 30, tol_kPa: float = 0.1):
        self.max_iter = max_iter
        self.tol_kPa  = tol_kPa

    def solve(
        self,
        connections : dict,       # {name: Connection}
        node_pressures: Dict[str, float],  # {node_name: Pa}
        pipe_pumps  : Dict[str, PumpModel], # {conn_name: PumpModel}
    ) -> PressureBalanceSolution:
        """
        Solve pressure balance for the network.

        Parameters
        ----------
        connections    : dict of Connection objects keyed by name
        node_pressures : dict mapping each node name → operating pressure (Pa)
        pipe_pumps     : dict mapping each connection name → PumpModel

        Returns
        -------
        PressureBalanceSolution
        """
        solution = PressureBalanceSolution()
        prev_powers: Dict[str, float] = {}

        for iteration in range(self.max_iter):
            max_change_kPa = 0.0

            for conn_name, conn in connections.items():
                pump = pipe_pumps.get(conn_name)
                if pump is None:
                    pump = PumpModel(conn_name)
                    pipe_pumps[conn_name] = pump

                src_P = node_pressures.get(conn.source, P_ATM)
                tgt_P = node_pressures.get(conn.target, P_ATM)

                try:
                    state = pump.solve(
                        geom             = conn.pipeline.spec,
                        flow_rate_m3s    = conn.pipeline.flow_rate_m3s,
                        node_P_inlet_Pa  = src_P,
                        node_P_outlet_Pa = tgt_P,
                        fluid            = conn.pipeline.fluid,
                    )
                except Exception as exc:
                    solution.warnings.append(
                        f"[{conn_name}] Pump solve failed: {exc}"
                    )
                    state = PumpState(warning=str(exc), feasible=False)
                    pump._state = state

                solution.pump_states[conn_name] = state

                if not state.feasible:
                    if conn_name not in solution.infeasible_pipes:
                        solution.infeasible_pipes.append(conn_name)
                    solution.warnings.append(
                        f"[{conn_name}] {state.warning}"
                    )

                # Track convergence for cycle detection
                prev_kW = prev_powers.get(conn_name, None)
                if prev_kW is not None:
                    change = abs(state.shaft_power_kW - prev_kW)
                    # Convert to ΔP change proxy via power (good enough for convergence)
                    max_change_kPa = max(max_change_kPa, change)
                prev_powers[conn_name] = state.shaft_power_kW

            solution.n_iterations = iteration + 1

            # Converged when max power change < tol (proxy for ΔP convergence)
            if iteration > 0 and max_change_kPa < self.tol_kPa:
                break
        else:
            solution.converged = False
            solution.warnings.append(
                f"Pressure solver did not fully converge in {self.max_iter} iterations "
                f"(max residual {max_change_kPa:.3f} kW). "
                "Network may have pressure cycles. Results are approximate."
            )

        solution.total_power_kW = sum(
            s.shaft_power_kW for s in solution.pump_states.values()
        )

        return solution

    def compute_power_time_series(
        self,
        connections     : dict,
        pipe_pumps      : Dict[str, PumpModel],
        node_pressures  : Dict[str, float],
        t_span          : Tuple[float, float],
        n_points        : int = 100,
        density_profiles: Optional[Dict[str, callable]] = None,
    ) -> Dict[str, PumpTimeSeries]:
        """
        Compute pump power over time for each pipe.

        Parameters
        ----------
        connections      : dict of Connection objects
        pipe_pumps       : dict of PumpModel per connection
        node_pressures   : constant node pressures
        t_span           : (t_start, t_end) seconds
        n_points         : time resolution
        density_profiles : optional {conn_name: callable t→density}

        Returns
        -------
        dict of PumpTimeSeries keyed by connection name
        """
        series: Dict[str, PumpTimeSeries] = {}

        for conn_name, conn in connections.items():
            pump = pipe_pumps.get(conn_name)
            if pump is None or pump.state is None:
                continue

            src_P = node_pressures.get(conn.source, P_ATM)
            tgt_P = node_pressures.get(conn.target, P_ATM)

            density_fn = None
            if density_profiles and conn_name in density_profiles:
                density_fn = density_profiles[conn_name]

            ts = pump.power_demand_over_time(
                t_span           = t_span,
                n_points         = n_points,
                density_fn       = density_fn,
                geom             = conn.pipeline.spec,
                base_flow        = conn.pipeline.flow_rate_m3s,
                node_P_inlet_Pa  = src_P,
                node_P_outlet_Pa = tgt_P,
            )
            series[conn_name] = ts

        return series
