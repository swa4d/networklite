"""
network/plant.py  –  PlantNetwork v2.

Key v2 changes:
  - Nodes dict holds both CSTR and SourceSink objects (unified).
  - Pressure solving delegated to NetworkPressureSolver (TransportSim).
  - Mass balance includes source/sink nodes.
  - topology() reports pump data per edge.
  - FigureRegistry prevents stale graph caching.
"""

from __future__ import annotations

import copy
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

_proj_dir = "/mnt/project"
if _proj_dir not in sys.path:
    sys.path.insert(0, _proj_dir)

# ── Internal imports ──────────────────────────────────────────────────────────
try:
    from network.reactor.cstr import CSTR, FeedStream, TemperatureGradient, CSTRResult
    _HAS_CSTR = True
except ImportError:
    _HAS_CSTR = False
    CSTR = None

try:
    from network.reactor.source_sink import SourceSink, SourceSinkResult
    _HAS_SS = True
except ImportError:
    _HAS_SS = False
    SourceSink = None

from network.pipeline.connection import Connection, ConnectionDiagnostic

try:
    from transportsim.pressure_solver import NetworkPressureSolver, PressureBalanceSolution
    from transportsim.pump import PumpModel
    _HAS_TS = True
except ImportError:
    _HAS_TS = False
    NetworkPressureSolver = None

P_ATM = 101_325.0   # Pa

# Type alias for any node
ReactorNode = Union["CSTR", "SourceSink"]


# ── Figure registry (graph cleanup) ──────────────────────────────────────────

class FigureRegistry:
    """
    Tracks matplotlib figures associated with open inspector windows.
    When a window is closed, call close_window() to release the figures.
    """
    def __init__(self):
        self._registry: Dict[str, list] = {}  # window_id → [fig_ids]

    def register(self, window_id: str, fig_b64_key: str) -> None:
        self._registry.setdefault(window_id, []).append(fig_b64_key)

    def close_window(self, window_id: str) -> List[str]:
        """Remove all cached plot keys for this window. Returns removed keys."""
        return self._registry.pop(window_id, [])

    def close_all(self) -> None:
        self._registry.clear()


# ── Network simulation result ─────────────────────────────────────────────────

@dataclass
class NetworkSimulationResult:
    """Aggregated result from a full plant network simulation."""
    reactor_results    : Dict[str, object]          # CSTRResult or SourceSinkResult
    connection_diags   : Dict[str, ConnectionDiagnostic]
    pressure_balance   : Optional["PressureBalanceSolution"]
    global_issues      : List[str]
    material_balance   : Dict[str, dict]
    total_flow_m3s     : float
    network_converged  : bool
    simulation_order   : List[str]
    total_pump_kW      : float = 0.0
    run_dry_reactors   : List[str] = field(default_factory=list)
    overflow_reactors  : List[str] = field(default_factory=list)   # v2.1: inlet > outlet
    stability_errors   : Dict[str, str] = field(default_factory=dict)
    source_pressures   : Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        pb = self.pressure_balance.to_dict() if self.pressure_balance else {}
        return {
            "reactor_results"  : {k: v.to_dict() for k, v in self.reactor_results.items()},
            "connection_diags" : {k: v.to_dict() for k, v in self.connection_diags.items()},
            "pressure_balance" : pb,
            "global_issues"    : self.global_issues,
            "material_balance" : self.material_balance,
            "total_flow_m3s"   : self.total_flow_m3s,
            "network_converged": self.network_converged,
            "simulation_order" : self.simulation_order,
            "total_pump_kW"    : round(self.total_pump_kW, 4),
            "run_dry_reactors" : self.run_dry_reactors,
            "overflow_reactors": self.overflow_reactors,
            "stability_errors" : self.stability_errors,
            "source_pressures" : {k: round(v / 1000, 3) for k, v in self.source_pressures.items()},
            "pump_breakdown"   : pb.get("pump_states", {}),
        }


# ── PlantNetwork ──────────────────────────────────────────────────────────────

class PlantNetwork:
    """
    Top-level plant: directed graph of nodes (CSTR + SourceSink) connected
    by pipelines, each with a virtual pump.
    """

    def __init__(self, name: str = "Plant"):
        self.name             : str                        = name
        self._nodes           : Dict[str, ReactorNode]    = {}   # unified node dict
        self._connections     : Dict[str, Connection]     = {}
        self._last_result     : Optional[NetworkSimulationResult] = None
        self.species_properties: Dict[str, dict]          = {}
        self._pump_registry   : Dict[str, PumpModel]      = {}   # conn_name → PumpModel
        self.figure_registry  : FigureRegistry            = FigureRegistry()

        # Pressure solver
        if _HAS_TS:
            self._pressure_solver = NetworkPressureSolver()
        else:
            self._pressure_solver = None

    # ── Node management (unified: CSTR + SourceSink) ──────────────────────────

    def add_node(self, node: ReactorNode) -> None:
        self._nodes[node.name] = node

    def add_reactor(self, reactor) -> None:
        """Alias for backward compatibility."""
        self._nodes[reactor.name] = reactor

    def add_source_sink(self, node: "SourceSink") -> None:
        self._nodes[node.name] = node

    def remove_node(self, name: str) -> None:
        self._nodes.pop(name, None)
        to_remove = [k for k, c in self._connections.items()
                     if c.source == name or c.target == name]
        for k in to_remove:
            del self._connections[k]
        self._pump_registry.pop(name, None)

    def remove_reactor(self, name: str) -> None:
        self.remove_node(name)

    def get_node(self, name: str) -> ReactorNode:
        if name not in self._nodes:
            raise KeyError(f"Node '{name}' not found.")
        return self._nodes[name]

    def get_reactor(self, name: str) -> ReactorNode:
        return self.get_node(name)

    @property
    def reactors(self) -> Dict[str, ReactorNode]:
        """All nodes (CSTR + SourceSink) — backward-compatible name."""
        return dict(self._nodes)

    @property
    def cstr_nodes(self) -> Dict[str, "CSTR"]:
        if _HAS_CSTR:
            return {k: v for k, v in self._nodes.items() if isinstance(v, CSTR)}
        return {}

    @property
    def source_sink_nodes(self) -> Dict[str, "SourceSink"]:
        if _HAS_SS:
            return {k: v for k, v in self._nodes.items() if isinstance(v, SourceSink)}
        return {}

    def _node_pressure_Pa(self, name: str) -> float:
        node = self._nodes.get(name)
        if node is None:
            return P_ATM
        return getattr(node, "pressure_Pa", P_ATM)

    # ── Connection management ─────────────────────────────────────────────────

    def add_connection(self, conn: Connection) -> None:
        self._connections[conn.name] = conn
        # Sync pump registry
        self._pump_registry[conn.name] = conn.pump

    def remove_connection(self, name: str) -> None:
        self._connections.pop(name, None)
        self._pump_registry.pop(name, None)

    def get_connection(self, name: str) -> Connection:
        if name not in self._connections:
            raise KeyError(f"Connection '{name}' not found.")
        return self._connections[name]

    def set_flow(self, name: str, flow_rate_m3s: float) -> None:
        self.get_connection(name).set_flow(flow_rate_m3s)

    @property
    def connections(self) -> Dict[str, Connection]:
        return dict(self._connections)

    # ── Topology ──────────────────────────────────────────────────────────────

    def _topological_order(self) -> List[str]:
        """Kahn's algorithm — returns all node names in simulation order."""
        in_degree: Dict[str, int] = {n: 0 for n in self._nodes}
        adj: Dict[str, List[str]] = {n: [] for n in self._nodes}

        for conn in self._connections.values():
            if conn.source in adj and conn.target in in_degree:
                adj[conn.source].append(conn.target)
                in_degree[conn.target] += 1

        queue = [n for n, d in in_degree.items() if d == 0]
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nb in adj[node]:
                in_degree[nb] -= 1
                if in_degree[nb] == 0:
                    queue.append(nb)

        # Handle cycles
        for n in self._nodes:
            if n not in order:
                order.append(n)
        return order

    # ── Pressure solving (v2 auto-balance) ───────────────────────────────────

    def _solve_pressures(self) -> Tuple[Dict[str, ConnectionDiagnostic], Optional["PressureBalanceSolution"]]:
        """
        Run NetworkPressureSolver for all connections.
        Every pipe's pump ΔP is solved using node operating pressures.
        """
        # Collect node pressures
        node_pressures = {name: self._node_pressure_Pa(name) for name in self._nodes}

        if self._pressure_solver and self._connections:
            pressure_balance = self._pressure_solver.solve(
                connections    = self._connections,
                node_pressures = node_pressures,
                pipe_pumps     = self._pump_registry,
            )
        else:
            pressure_balance = None

        # Build connection diagnostics from pump solutions
        diags: Dict[str, ConnectionDiagnostic] = {}
        for conn_name, conn in self._connections.items():
            src_P = node_pressures.get(conn.source, P_ATM)
            tgt_P = node_pressures.get(conn.target, P_ATM)
            diag  = conn.diagnose(
                node_P_inlet_Pa  = src_P,
                node_P_outlet_Pa = tgt_P,
            )
            diags[conn_name] = diag

        return diags, pressure_balance

    # ── Outlet routing ────────────────────────────────────────────────────────

    def _route_outlet_to_feeds(self, upstream_names: List[str]) -> None:
        """Inject upstream reactor outlet into downstream reactor feeds."""
        last = upstream_names[-1]
        node = self._nodes[last]

        is_source_sink = _HAS_SS and isinstance(node, SourceSink)

        # SourceSinks route their composition forward directly
        if is_source_sink:
            outlet  = node.outlet_composition
            Q_out   = node.flow_rate_m3s
            sp_comp = dict(outlet)
        elif hasattr(node, "result") and node.result is not None:
            outlet  = node.result.outlet_composition
            Q_out   = getattr(node, "total_flow_m3s", 0.0)
            sp_comp = dict(outlet)
        else:
            return

        for conn in self._connections.values():
            if conn.source != last:
                continue
            target_name = conn.target
            target_node = self._nodes.get(target_name)
            if target_node is None:
                continue
            # Only inject into CSTR nodes (SourceSinks don't need feeds)
            if _HAS_CSTR and isinstance(target_node, CSTR):
                if is_source_sink:
                    # The SourceSink IS the physical feed for this CSTR.
                    # Remove ALL existing explicit (non-auto) feeds from this
                    # source direction to prevent double-counting: the pipe from
                    # the Source node replaces any manually-added direct feeds.
                    target_node._feeds = [
                        f for f in target_node._feeds
                        if f.name.startswith("_from_")
                    ]
                else:
                    # For CSTR→CSTR routing, only remove the previous auto-feed
                    target_node._feeds = [
                        f for f in target_node._feeds
                        if not f.name.startswith(f"_from_{last}")
                    ]

                Q_pipe = conn.pipeline.flow_rate_m3s
                if sp_comp and Q_pipe > 0:
                    target_node.add_feed(FeedStream(
                        name=f"_from_{last}",
                        compositions=dict(sp_comp),
                        flow_rate_m3s=Q_pipe,
                    ))

    # ── Material balance ──────────────────────────────────────────────────────

    def _compute_material_balance(self, order: List[str]) -> Dict[str, dict]:
        """
        Compute molar flow balance for all species across all nodes.

        Sources contribute to inflow; sinks and CSTR outlets contribute to outflow.
        Internal flows (node→node connections) cancel out.
        """
        balance: Dict[str, dict] = {}

        internal_flows: Set[Tuple[str, str]] = set()
        for conn in self._connections.values():
            if conn.source in self._nodes and conn.target in self._nodes:
                internal_flows.add((conn.source, conn.target))

        for rname in order:
            node = self._nodes[rname]

            # Inlet contributions
            if _HAS_CSTR and isinstance(node, CSTR):
                for feed in node.feeds:
                    if feed.name.startswith("_from_"):
                        continue  # internal — skip to avoid double-counting
                    for sp, c in feed.compositions.items():
                        mf = c * feed.flow_rate_Ls
                        balance.setdefault(sp, {"in_mol_s": 0.0, "out_mol_s": 0.0})
                        balance[sp]["in_mol_s"] += mf

            elif _HAS_SS and isinstance(node, SourceSink) and node.node_type == "source":
                # Source: inject its composition at its flow rate
                for sp, c in node.species.items():
                    mf = c * node.total_flow_Ls
                    balance.setdefault(sp, {"in_mol_s": 0.0, "out_mol_s": 0.0})
                    balance[sp]["in_mol_s"] += mf

            # Outlet contributions: only for terminal nodes (not passed to another internal node)
            outlet_conns = [c for c in self._connections.values() if c.source == rname]
            terminal     = not any(c.target in self._nodes for c in outlet_conns
                                   if (rname, c.target) in internal_flows)

            if terminal or not outlet_conns:
                outlet_comp = {}
                Q_out_Ls    = 0.0

                if _HAS_CSTR and isinstance(node, CSTR) and node.result:
                    outlet_comp = node.result.outlet_composition
                    Q_out_Ls    = node.total_flow_Ls
                elif _HAS_SS and isinstance(node, SourceSink) and node.node_type == "sink":
                    outlet_comp = node.species
                    Q_out_Ls    = node.total_flow_Ls

                for sp, c in outlet_comp.items():
                    mf = c * Q_out_Ls
                    balance.setdefault(sp, {"in_mol_s": 0.0, "out_mol_s": 0.0})
                    balance[sp]["out_mol_s"] += mf

        for sp in balance:
            bal = balance[sp]
            bal["generated_mol_s"] = bal["out_mol_s"] - bal["in_mol_s"]
            bal["closure_pct"] = (
                100.0 * bal["out_mol_s"] / bal["in_mol_s"]
                if bal["in_mol_s"] > 1e-12 else 0.0
            )
        return balance

    # ── Stability / run-dry checks ────────────────────────────────────────────

    def _check_stability(
        self,
        order           : List[str],
        reactor_results : Dict[str, object],
        material_balance: Dict[str, dict],
    ) -> Tuple[List[str], Dict[str, str], List[str]]:
        run_dry:    List[str]       = []
        errors:     Dict[str, str] = {}
        new_issues: List[str]      = []

        for rname in order:
            node = self._nodes[rname]

            # ── Sink demand validation ─────────────────────────────────────
            if _HAS_SS and isinstance(node, SourceSink) and node.node_type == "sink":
                rr = reactor_results.get(rname)
                # Check that each upstream connection is actually delivering flow
                inlet_conns = [c for c in self._connections.values() if c.target == rname]
                total_inlet = sum(c.pipeline.flow_rate_m3s for c in inlet_conns)
                if total_inlet < 1e-9 and inlet_conns:
                    msg = f"Sink '{rname}' has no flow — upstream reactors may be empty."
                    new_issues.append(f"[{rname}] {msg}")
                # Check min-concentration specs on sink
                specs = getattr(node, "product_specs", {})
                if specs and rr is not None:
                    for sp, min_c in specs.items():
                        actual = rr.outlet_composition.get(sp, 0.0)
                        if actual < min_c:
                            msg = (f"Sink '{rname}': {sp} = {actual:.4f} M < "
                                   f"required {min_c:.4f} M (deficit {(min_c-actual):.4f} M).")
                            new_issues.append(f"[{rname}] {msg}")
                continue

            if _HAS_SS and isinstance(node, SourceSink):
                continue  # source — never runs dry

            rr = reactor_results.get(rname)
            if rr is None:
                continue

            downstream_Q = sum(
                c.pipeline.flow_rate_m3s
                for c in self._connections.values()
                if c.source == rname
            )

            if _HAS_CSTR and isinstance(node, CSTR):
                inlet_Q = node.total_flow_m3s   # now correct — no double-counting
                vol_L   = node.volume_L
            else:
                inlet_Q = getattr(node, "total_flow_m3s", 0.0)
                vol_L   = getattr(node, "volume_L", 0.0)

            # ── Run-dry: demand exceeds supply ─────────────────────────────
            if downstream_Q > inlet_Q * 1.15 and inlet_Q > 1e-9:
                msg = (
                    f"Run-dry: downstream demand {downstream_Q*1000:.3f} L/s "
                    f"exceeds inlet {inlet_Q*1000:.3f} L/s "
                    f"({downstream_Q/inlet_Q:.2f}×)."
                )
                run_dry.append(rname)
                errors[rname] = msg
                new_issues.append(f"[{rname}] {msg}")

            # ── Overflow: more fed in than piped out ───────────────────────
            elif inlet_Q > downstream_Q * 1.10 and downstream_Q > 1e-9 and vol_L < float("inf"):
                excess_Ls = (inlet_Q - downstream_Q) * 1000.0
                # Time until volume overflows at this accumulation rate
                # (vol_L already full; marginal overflow rate = excess_Ls L/s)
                fill_time_s = vol_L / max(excess_Ls, 1e-9)
                msg = (
                    f"Overflow: inlet {inlet_Q*1000:.3f} L/s > outlet {downstream_Q*1000:.3f} L/s. "
                    f"Excess {excess_Ls:.3f} L/s accumulating — overflow in ~{fill_time_s:.0f} s."
                )
                errors[rname] = msg
                new_issues.append(f"[{rname}] OVERFLOW: {msg}")
                # Overflow is a warning, not run-dry — don't add to run_dry list
                # but flag it so the UI can show the overflow marker

            if not getattr(rr, "converged", True):
                if rname not in errors:
                    run_dry.append(rname)
                    errors[rname] = "Reactor did not converge."
                    new_issues.append(f"[{rname}] Reactor did not converge.")

        return run_dry, errors, new_issues


    # ── Main simulation ───────────────────────────────────────────────────────

    def simulate(
        self,
        t_end            : float = 500.0,
        n_segments       : int   = 20,
        solver_params    : Optional[dict] = None,
        mode             : str   = "deterministic",
        stochastic_params: Optional[dict] = None,
    ) -> NetworkSimulationResult:
        global_issues: List[str] = []
        order = self._topological_order()

        # 1. Auto-balance pressure solve (v2)
        diags, pressure_balance = self._solve_pressures()
        for dname, diag in diags.items():
            for iss in diag.issues:
                global_issues.append(f"[{dname}] {iss}")

        # 2. Sequential node simulation
        reactor_results: Dict[str, object] = {}
        for i, rname in enumerate(order):
            node = self._nodes[rname]
            if i > 0:
                self._route_outlet_to_feeds(order[:i])
            try:
                sim_kwargs = {}
                if _HAS_CSTR and isinstance(node, CSTR):
                    sim_kwargs["solver_params"]     = solver_params
                    sim_kwargs["mode"]              = mode
                    sim_kwargs["stochastic_params"] = stochastic_params
                result = node.simulate(
                    t_end=t_end, n_segments=n_segments,
                    **sim_kwargs,
                )
                reactor_results[rname] = result
                for iss in getattr(result, "issues", []):
                    global_issues.append(f"[{rname}] {iss}")
            except Exception as e:
                global_issues.append(f"[{rname}] Simulation failed: {e}")

        # 3. Material balance (all nodes including sources/sinks)
        material_balance = self._compute_material_balance(order)

        # 4. Mass balance closure warnings
        for sp, bal in material_balance.items():
            if abs(bal["closure_pct"] - 100.0) > 10.0 and bal["in_mol_s"] > 1e-12:
                global_issues.append(
                    f"Mass balance closure for {sp}: {bal['closure_pct']:.1f}% "
                    "(>10% deviation)."
                )

        # 5. Stability / run-dry / overflow
        run_dry, stability_errors, stab_issues = self._check_stability(
            order, reactor_results, material_balance
        )
        global_issues.extend(stab_issues)

        # Separate overflow reactors from run-dry (overflow not in run_dry list)
        overflow_reactors = [
            rname for rname, msg in stability_errors.items()
            if "Overflow" in msg and rname not in run_dry
        ]

        # 6. Pump energy from pressure balance
        total_pump_kW = 0.0
        if pressure_balance:
            total_pump_kW = pressure_balance.total_power_kW
        else:
            total_pump_kW = sum(d.pump_power_kW for d in diags.values())

        # 7. Source pressures (for display)
        source_pressures = {
            name: node.pressure_Pa
            for name, node in self._nodes.items()
            if _HAS_SS and isinstance(node, SourceSink)
        }

        total_flow = sum(
            getattr(n, "total_flow_m3s", 0.0) for n in self._nodes.values()
        )

        self._last_result = NetworkSimulationResult(
            reactor_results    = reactor_results,
            connection_diags   = diags,
            pressure_balance   = pressure_balance,
            global_issues      = global_issues,
            material_balance   = material_balance,
            total_flow_m3s     = total_flow,
            network_converged  = (
                len(reactor_results) == len(self._nodes) and not run_dry
            ),
            simulation_order   = order,
            total_pump_kW      = round(total_pump_kW, 4),
            run_dry_reactors   = run_dry,
            overflow_reactors  = overflow_reactors,
            stability_errors   = stability_errors,
            source_pressures   = source_pressures,
        )
        return self._last_result

    # ── Inspection API ────────────────────────────────────────────────────────

    def inspect_reactor(self, name: str) -> dict:
        node = self.get_node(name)
        d    = node.to_dict() if hasattr(node, "to_dict") else {}

        d["inlet_connections"] = [
            {"name": cn, "source": c.source,
             "flow_Ls": round(c.pipeline.flow_rate_m3s * 1000, 4)}
            for cn, c in self._connections.items() if c.target == name
        ]
        d["outlet_connections"] = [
            {"name": cn, "target": c.target,
             "flow_Ls": round(c.pipeline.flow_rate_m3s * 1000, 4)}
            for cn, c in self._connections.items() if c.source == name
        ]

        if _HAS_CSTR and isinstance(node, CSTR) and node.result:
            d["result_summary"] = {
                "outlet_composition": node.result.outlet_composition,
                "conversion"        : node.result.conversion,
                "residence_time_s"  : node.result.residence_time_s,
                "converged"         : node.result.converged,
                "issues"            : node.result.issues,
            }
        if self._last_result and name in self._last_result.run_dry_reactors:
            d["error_state"]   = True
            d["error_message"] = self._last_result.stability_errors.get(name, "")
        else:
            d["error_state"]   = False
            d["error_message"] = ""
        return d

    def inspect_connection(self, name: str) -> dict:
        conn = self.get_connection(name)
        src_P = self._node_pressure_Pa(conn.source)
        tgt_P = self._node_pressure_Pa(conn.target)
        diag  = conn.diagnose(src_P, tgt_P)
        d     = conn.to_dict()
        d["diagnostic"] = diag.to_dict()
        return d

    def topology(self) -> dict:
        nodes = []
        for nname, node in self._nodes.items():
            rd    = self._last_result
            error = bool(rd and nname in rd.run_dry_reactors) if rd else False
            n_sp  = len(getattr(getattr(node, "reaction_network", None),
                                "species_names", [])) if hasattr(node, "reaction_network") else \
                    len(getattr(node, "species", {}))
            nodes.append({
                "id"           : nname,
                "label"        : nname,
                "node_type"    : getattr(node, "node_type", "cstr"),
                "volume_L"     : None if getattr(node, "volume_L", None) == float("inf")
                                 else getattr(node, "volume_L", None),
                "pressure_kPa" : round(self._node_pressure_Pa(nname) / 1000, 3),
                "pressure_mode": getattr(node, "pressure_mode", "atm"),
                "n_feeds"      : len(getattr(node, "feeds", [])),
                "n_reactions"  : len(getattr(getattr(node, "reaction_network", None),
                                             "reactions", [])),
                "n_species"    : n_sp,
                "tau_s"        : getattr(node, "residence_time_s", None),
                "converged"    : getattr(getattr(node, "result", None), "converged", None),
                "error_state"  : error,
            })

        edges = []
        for cname, conn in self._connections.items():
            diag = self._last_result.connection_diags.get(cname) if self._last_result else None
            edges.append({
                "id"                   : cname,
                "source"               : conn.source,
                "target"               : conn.target,
                "flow_Ls"              : round(conn.pipeline.flow_rate_m3s * 1000, 4),
                "length_m"             : conn.pipeline.spec.length,
                "diameter_m"           : conn.pipeline.spec.diameter,
                "pump_delta_p_kPa"     : round(diag.pump_delta_p_kPa, 3) if diag else 0.0,
                "pump_power_kW"        : round(diag.pump_power_kW, 4)     if diag else 0.0,
                "pump_efficiency_pct"  : round(conn.pump.efficiency * 100, 1),
                "flow_regime"          : diag.flow_regime  if diag else "unknown",
                "pump_feasible"        : diag.pump_feasible if diag else True,
                "roughness_default"    : not conn.pipeline.spec.roughness_user_set,
                "density_default"      : not conn.pipeline.fluid.density_user_set,
                # Legacy keys
                "needs_compressor"     : not diag.pump_feasible if diag else False,
                "pressure_drop_kPa"    : round(diag.pipe_pressure_drop_kPa, 3) if diag else 0.0,
                "pump_kW"              : round(diag.pump_power_kW, 4) if diag else 0.0,
            })

        pb = self._last_result.pressure_balance if self._last_result else None
        return {
            "nodes"         : nodes,
            "edges"         : edges,
            "total_pump_kW" : round(self._last_result.total_pump_kW, 4) if self._last_result else 0.0,
            "pressure_balance": pb.to_dict() if pb else {},
        }

    def to_dict(self) -> dict:
        return {
            "name"       : self.name,
            "reactors"   : {n: node.to_dict() for n, node in self._nodes.items()},
            "connections": {n: c.to_dict() for n, c in self._connections.items()},
        }