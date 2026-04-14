"""
dashboard/app.py  –  Flask REST API for NetworkLite v2.

v2 changes:
  - /api/nodes  replaces /api/reactors for unified node access
  - /api/reactors  kept as alias (backward-compat)
  - /api/nodes/<n>/pressure  – set node operating pressure
  - /api/connections/<n>/pump  – update pump efficiency
  - /api/connections/<n>/sweep  – pump curve sweep (ΔP + power)
  - /api/pipe_plot/<n>  – pressure sweep + operating curve + regime map + Fanning
  - /api/reactor_plot/<n>  – includes full time array for slider + numeric concs
  - /api/plots/close/<window_id>  – release figure cache on window close
  - /api/balance  – fixed: includes source/sink nodes
  - /api/sync_plant  – handles source/sink nodes as SourceSink objects
  - /api/validate  – validates pump feasibility + unset defaults
"""

from __future__ import annotations

import copy
import json
import sys
import os
import traceback
from typing import Optional

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ── Path setup ────────────────────────────────────────────────────────────────
_dash_dir = os.path.dirname(os.path.abspath(__file__))
_proj_dir = "/mnt/project"

for p in [_dash_dir, os.path.dirname(_dash_dir), _proj_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Core imports ──────────────────────────────────────────────────────────────
from network.plant import PlantNetwork, NetworkSimulationResult
from network.reactor.source_sink import SourceSink

try:
    from network.reactor.cstr import CSTR, FeedStream, TemperatureGradient
    _HAS_CSTR = True
except ImportError:
    _HAS_CSTR = False

from network.pipeline.connection import Connection
from network.optimizer.multi_objective import (
    NetworkOptimizer, DecisionVariable, Objective,
    yield_objective, conversion_objective,
    residence_time_objective, total_compressor_power_objective,
)
from network.analysis.diagnostics import generate_all_plots

try:
    from transportsim import (
        Pipeline, PipelineSpec, FluidProperties,
        plot_pressure_sweep, plot_pump_operating_curve,
        plot_flow_regime_map, plot_fanning_vs_darcy,
        plot_pressure_breakdown, plot_pump_power_over_time,
    )
    _HAS_TS = True
except ImportError:
    _HAS_TS = False

try:
    from chemsim.network import ReactionNetwork
    _HAS_CHEMSIM = True
except ImportError:
    _HAS_CHEMSIM = False

# ── App ───────────────────────────────────────────────────────────────────────
app  = Flask(__name__, static_folder=_dash_dir, static_url_path="")
CORS(app)

_plant      : Optional[PlantNetwork] = None
_last_result                         = None
_last_opt                            = None
# Plot cache: {window_id → {plot_key: b64_string}}
# Cleared when the frontend closes a window (DELETE /api/plots/window/<id>)
_plot_cache : dict = {}


def _get_plant() -> PlantNetwork:
    global _plant
    if _plant is None:
        _plant = PlantNetwork("Default Plant")
    return _plant


def _ok(data=None, **kwargs):
    payload = {"status": "ok"}
    if data is not None:
        payload["data"] = data
    payload.update(kwargs)
    return jsonify(payload)


def _err(msg: str, code: int = 400):
    return jsonify({"status": "error", "message": msg}), code


# ── Static ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # index.html lives alongside app.py in the dashboard/ directory
    return send_from_directory(_dash_dir, "index.html")


@app.route("/api/status")
def status():
    plant = _get_plant()
    return _ok({
        "plant_name"     : plant.name,
        "n_nodes"        : len(plant.reactors),
        "n_reactors"     : len(plant.cstr_nodes),
        "n_source_sinks" : len(plant.source_sink_nodes),
        "n_connections"  : len(plant.connections),
        "chemsim"        : _HAS_CHEMSIM,
        "transportsim"   : _HAS_TS,
        "simulated"      : _last_result is not None,
        "version"        : "2.0.0",
    })


@app.route("/api/topology")
def topology():
    return _ok(_get_plant().topology())


# ── Nodes (unified: CSTR + SourceSink) ───────────────────────────────────────

@app.route("/api/nodes")
@app.route("/api/reactors")
def list_nodes():
    plant = _get_plant()
    return _ok([n.to_dict() for n in plant.reactors.values()])


@app.route("/api/nodes/<n>")
@app.route("/api/reactors/<n>")
def get_node(n):
    try:
        return _ok(_get_plant().inspect_reactor(n))
    except KeyError as e:
        return _err(str(e))


@app.route("/api/nodes/<n>/pressure", methods=["POST"])
def set_node_pressure(n):
    """
    Set operating pressure for any node (CSTR or SourceSink).
    Body: { "pressure_kPa": float }
    """
    data = request.get_json() or {}
    try:
        plant  = _get_plant()
        node   = plant.get_node(n)
        kPa    = float(data.get("pressure_kPa", 101.325))
        node.pressure_Pa   = kPa * 1000.0
        node.pressure_mode = "fixed"
        return _ok({"node": n, "pressure_kPa": round(kPa, 3), "pressure_Pa": kPa * 1000})
    except Exception as e:
        return _err(str(e))


@app.route("/api/nodes", methods=["POST"])
@app.route("/api/reactors", methods=["POST"])
def add_reactor():
    """
    Add a new CSTR node.
    Body: { "name": str, "volume_L": float }
    """
    global _plant
    data = request.get_json() or {}
    name = data.get("name", "")
    vol  = float(data.get("volume_L", 100.0))

    if not name:
        return _err("Node name required.")
    if not _HAS_CHEMSIM:
        return _err("ChemSim not installed — cannot create reactors.")

    plant = _get_plant()
    if name in plant.reactors:
        return _err(f"Node '{name}' already exists.")

    rn = ReactionNetwork()
    r  = CSTR(name=name, volume_L=vol, reaction_network=rn)
    plant.add_reactor(r)
    return _ok(r.to_dict())


@app.route("/api/nodes/<n>", methods=["DELETE"])
@app.route("/api/reactors/<n>", methods=["DELETE"])
def delete_node(n):
    _get_plant().remove_node(n)
    # Clear any cached plots for this node
    _plot_cache.pop(n, None)
    return _ok({"removed": n})


# ── CSTR-specific endpoints ───────────────────────────────────────────────────

@app.route("/api/reactors/<n>/feed", methods=["POST"])
def add_feed(n):
    try:
        plant   = _get_plant()
        reactor = plant.get_reactor(n)
        data    = request.get_json() or {}
        feed = FeedStream(
            name          = data.get("feed_name", "feed"),
            compositions  = data.get("compositions", {}),
            flow_rate_m3s = float(data.get("flow_rate_m3s", 1e-4)),
            temperature_K = float(data.get("temperature_K", 298.15)),
        )
        reactor.remove_feed(feed.name)
        reactor.add_feed(feed)
        return _ok(reactor.to_dict())
    except Exception as e:
        return _err(str(e))


@app.route("/api/reactors/<n>/species", methods=["POST"])
def add_species(n):
    try:
        plant   = _get_plant()
        reactor = plant.get_reactor(n)
        data    = request.get_json() or {}
        sp      = data.get("species", "")
        init    = float(data.get("initial", 0.0))
        if not sp:
            return _err("Species name required.")
        reactor.add_species(sp, initial=init)
        return _ok(reactor.to_dict())
    except Exception as e:
        return _err(str(e))


@app.route("/api/reactors/<n>/reaction", methods=["POST"])
def add_reaction(n):
    try:
        plant   = _get_plant()
        reactor = plant.get_reactor(n)
        data    = request.get_json() or {}
        reactor.add_reaction(
            reactants         = data.get("reactants", []),
            products          = data.get("products",  []),
            rate              = float(data.get("rate", 0.1)),
            reactant_stoich   = data.get("reactant_stoich"),
            product_stoich    = data.get("product_stoich"),
            activation_energy = float(data.get("activation_energy", 0.0)),
            pre_exponential   = float(data.get("pre_exponential",   0.0)),
            label             = data.get("label", ""),
        )
        return _ok(reactor.to_dict())
    except Exception as e:
        return _err(str(e))


@app.route("/api/reactors/<n>/temperature", methods=["POST"])
def set_temperature(n):
    try:
        plant   = _get_plant()
        reactor = plant.get_reactor(n)
        data    = request.get_json() or {}
        reactor.temperature_gradient = TemperatureGradient(
            mode         = data.get("mode", "isothermal"),
            T_initial_K  = float(data.get("T_initial_K", 298.15)),
            T_final_K    = float(data.get("T_final_K",   298.15)),
            t_ramp       = float(data.get("t_ramp",      100.0)),
            t_step       = float(data.get("t_step",       50.0)),
            custom_times = data.get("custom_times", []),
            custom_temps = data.get("custom_temps", []),
        )
        return _ok(reactor.to_dict())
    except Exception as e:
        return _err(str(e))


@app.route("/api/reactors/<n>/volume", methods=["POST"])
def set_volume(n):
    data    = request.get_json() or {}
    vol     = float(data.get("volume_L", 100.0))
    plant   = _get_plant()
    reactor = plant.get_reactor(n)
    if isinstance(reactor, SourceSink):
        return _err("SourceSink nodes have infinite volume.")
    reactor.volume_L = vol
    return _ok(reactor.to_dict())


# ── Connections ───────────────────────────────────────────────────────────────

@app.route("/api/connections")
def list_connections():
    plant = _get_plant()
    return _ok([c.to_dict() for c in plant.connections.values()])


@app.route("/api/connections/<n>")
def get_connection(n):
    try:
        return _ok(_get_plant().inspect_connection(n))
    except Exception as e:
        return _err(str(e))


@app.route("/api/connections", methods=["POST"])
def add_connection():
    """
    Add a pipeline connection.
    Body: { name, source, target, flow_rate_m3s,
            length_m, diameter_m, roughness_m, elevation_m, k_fittings,
            density, viscosity, pump_efficiency }
    """
    if not _HAS_TS:
        return _err("TransportSim not available.")
    try:
        plant = _get_plant()
        data  = request.get_json() or {}

        pipe_name = data.get("name", "")
        source    = data.get("source", "")
        target    = data.get("target", "")
        if not all([pipe_name, source, target]):
            return _err("name, source, target required.")

        roughness_raw  = data.get("roughness_m")
        roughness      = float(roughness_raw) if roughness_raw is not None else 4.6e-5
        roughness_set  = roughness_raw is not None

        density_raw    = data.get("density")
        density        = float(density_raw) if density_raw is not None else 1000.0
        density_set    = density_raw is not None

        spec = PipelineSpec(
            length             = float(data.get("length_m",    10.0)),
            diameter           = float(data.get("diameter_m",  0.05)),
            roughness          = roughness,
            elevation_change   = float(data.get("elevation_m", 0.0)),
            n_fittings_K       = int(data.get("k_fittings",   2)),
            roughness_user_set = roughness_set,
        )
        fluid = FluidProperties(
            density            = density,
            viscosity          = float(data.get("viscosity", 1.002e-3)),
            density_user_set   = density_set,
        )
        pipe = Pipeline(
            name          = pipe_name,
            source        = source,
            target        = target,
            spec          = spec,
            fluid         = fluid,
            flow_rate_m3s = float(data.get("flow_rate_m3s", 1e-4)),
        )
        efficiency = float(data.get("pump_efficiency", 0.75))
        conn = Connection(pipe, pump_efficiency=efficiency)
        plant.add_connection(conn)
        return _ok(conn.to_dict())
    except Exception as e:
        return _err(str(e))


@app.route("/api/connections/<n>", methods=["DELETE"])
def delete_connection(n):
    _get_plant().remove_connection(n)
    _plot_cache.pop(f"pipe_{n}", None)
    return _ok({"removed": n})


@app.route("/api/connections/<n>/flow", methods=["POST"])
def update_flow(n):
    data = request.get_json() or {}
    q    = float(data.get("flow_rate_m3s", 1e-4))
    try:
        _get_plant().set_flow(n, q)
        return _ok({"flow_rate_m3s": q, "flow_rate_Ls": round(q * 1000, 4)})
    except Exception as e:
        return _err(str(e))


@app.route("/api/connections/<n>/pump", methods=["POST"])
def update_pump(n):
    """
    Update pump efficiency (per-pipe).
    Body: { "efficiency": float  (0-1) }
    """
    data = request.get_json() or {}
    try:
        plant = _get_plant()
        conn  = plant.get_connection(n)
        eta   = float(data.get("efficiency", 0.75))
        conn.set_pump_efficiency(eta)
        # Re-solve pump with current node pressures
        src_P = plant._node_pressure_Pa(conn.source)
        tgt_P = plant._node_pressure_Pa(conn.target)
        state = conn.pump.solve(
            conn.pipeline.spec, conn.pipeline.flow_rate_m3s,
            src_P, tgt_P, conn.pipeline.fluid
        )
        return _ok({
            "pipe"      : n,
            "efficiency": round(eta, 3),
            "pump"      : state.to_dict(),
        })
    except Exception as e:
        return _err(str(e))


@app.route("/api/connections/<n>/sweep")
def pipeline_sweep(n):
    """Pump curve sweep: ΔP + power vs flow rate."""
    try:
        plant  = _get_plant()
        conn   = plant.get_connection(n)
        q_min  = float(request.args.get("q_min", 1e-6))
        q_max  = float(request.args.get("q_max", max(conn.pipeline.flow_rate_m3s * 5, 1e-3)))
        src_P  = plant._node_pressure_Pa(conn.source)
        tgt_P  = plant._node_pressure_Pa(conn.target)
        sweep  = conn.flow_sweep(q_min=q_min, q_max=q_max, n_points=60,
                                 node_P_inlet_Pa=src_P, node_P_outlet_Pa=tgt_P)
        return _ok(sweep)
    except Exception as e:
        return _err(str(e))


# ── Simulation ────────────────────────────────────────────────────────────────

@app.route("/api/simulate", methods=["POST"])
def simulate():
    global _last_result
    data       = request.get_json() or {}
    t_end      = float(data.get("t_end",       300.0))
    n_segments = int(data.get("n_segments",    15))

    plant = _get_plant()
    if not plant.reactors:
        return _err("Plant has no nodes. Add reactors or sources first.")
    try:
        result       = plant.simulate(t_end=t_end, n_segments=n_segments)
        _last_result = result
        return _ok(result.to_dict())
    except Exception as e:
        return _err(f"Simulation failed: {e}\n{traceback.format_exc()}")


# ── Plots ─────────────────────────────────────────────────────────────────────

@app.route("/api/plots")
def plots():
    global _last_result, _last_opt
    plant = _get_plant()
    if _last_result is None:
        return _err("No simulation result. Run /api/simulate first.")
    try:
        plot_dict = generate_all_plots(plant, _last_result, _last_opt)
        return _ok(plot_dict)
    except Exception as e:
        return _err(f"Plot generation failed: {e}")


@app.route("/api/plots/window/<window_id>", methods=["DELETE"])
def close_plot_window(window_id):
    """
    Called by the frontend when an inspector window is closed.
    Clears cached plot data so the next open generates a fresh plot.
    """
    removed = _plot_cache.pop(window_id, {})
    _get_plant().figure_registry.close_window(window_id)
    return _ok({"window_id": window_id, "cleared_keys": list(removed.keys())})


@app.route("/api/reactor_plot/<n>")
def reactor_plot(n):
    """
    Full reactor time-series data for inspector panel.

    Returns:
      - trajectory: base64 PNG
      - time: full time array (for slider)
      - conc: full concentrations array [time × species]
      - species: species name list
      - outlet: final outlet composition
      - conc_at_t: {species: value} at every time index (for numeric display)
      - temp: temperature profile
    """
    global _last_result
    if not _last_result:
        return _err("No simulation result. Run simulation first.")

    rr = _last_result.reactor_results.get(n)
    if not rr:
        return _err(f"Node '{n}' not found in results.")

    try:
        from network.analysis.diagnostics import plot_reactor_trajectories
        plots = plot_reactor_trajectories(_last_result, [n])
        trajectory_b64 = plots.get(f"trajectory_{n}", "")

        # Full time + concentration arrays for slider
        import numpy as np
        times = rr.time.tolist() if hasattr(rr.time, "tolist") else list(rr.time)
        concs = rr.concentrations.tolist() if hasattr(rr.concentrations, "tolist") else []
        names = list(rr.species_names) if hasattr(rr, "species_names") else []
        temp  = (rr.temperature_profile.tolist()
                 if hasattr(rr, "temperature_profile") else [])

        # Concentration table at each time step: [{species: value}, ...]
        conc_table = []
        for row in concs:
            conc_table.append({sp: round(row[i], 6) for i, sp in enumerate(names)})

        return _ok({
            "trajectory" : trajectory_b64,
            "outlet"     : rr.outlet_composition,
            "conversion" : rr.conversion if hasattr(rr, "conversion") else {},
            "time"       : times,
            "conc"       : concs,
            "conc_table" : conc_table,   # per-timestep dict for numeric display
            "species"    : names,
            "temp"       : temp,
        })
    except Exception as e:
        return _err(str(e))


@app.route("/api/pipe_plot/<n>")
def pipe_plot(n):
    """
    All four pressure-related plots for a pipeline inspector.

    Returns:
      sweep_plot    : system resistance curve (coloured by regime)
      pump_curve    : pump operating curve (ΔP + power panels)
      regime_map    : Re vs flow rate
      fanning_plot  : Darcy vs Fanning friction comparison
      breakdown     : pressure/power breakdown bar chart (at operating point)
      sweep_data    : raw sweep numbers for frontend charting
    """
    try:
        plant  = _get_plant()
        conn   = plant.get_connection(n)
        pipe   = conn.pipeline
        geom   = pipe.spec
        fluid  = pipe.fluid
        Q      = pipe.flow_rate_m3s
        eta    = conn.pump.efficiency
        src_P  = plant._node_pressure_Pa(conn.source)
        tgt_P  = plant._node_pressure_Pa(conn.target)

        sweep_data = conn.flow_sweep(
            q_min=max(Q * 0.05, 1e-6),
            q_max=max(Q * 5,    1e-3),
            n_points=60,
            node_P_inlet_Pa=src_P,
            node_P_outlet_Pa=tgt_P,
        )

        # Re-solve pump so state is fresh
        conn.diagnose(src_P, tgt_P)
        pump_state = conn.pump.state

        sweep_plot    = plot_pressure_sweep(
            n, geom, fluid,
            q_min_Ls=max(Q * 0.05 * 1000, 0.05),
            q_max_Ls=max(Q * 5 * 1000, 1.0),
            current_flow_m3s=Q,
            node_P_inlet_Pa=src_P, node_P_outlet_Pa=tgt_P,
        )
        pump_curve    = plot_pump_operating_curve(
            n, geom, fluid, efficiency=eta,
            q_min_Ls=max(Q * 0.05 * 1000, 0.05),
            q_max_Ls=max(Q * 5 * 1000, 1.0),
            current_flow_m3s=Q,
            node_P_inlet_Pa=src_P, node_P_outlet_Pa=tgt_P,
        )
        regime_map    = plot_flow_regime_map(
            n, geom, fluid,
            q_min_Ls=max(Q * 0.05 * 1000, 0.01),
            q_max_Ls=max(Q * 5 * 1000, 1.0),
        )
        fanning_plot  = plot_fanning_vs_darcy(n, geom, fluid)
        breakdown     = plot_pressure_breakdown(n, pump_state)

        return _ok({
            "sweep_plot"   : sweep_plot,
            "pump_curve"   : pump_curve,
            "regime_map"   : regime_map,
            "fanning_plot" : fanning_plot,
            "breakdown"    : breakdown,
            "sweep_data"   : sweep_data,
            "current_flow_m3s": Q,
            "pump"         : pump_state.to_dict() if pump_state else {},
            "spec"         : {
                "length"        : geom.length,
                "diameter"      : geom.diameter,
                "roughness"     : geom.roughness,
                "roughness_default": not geom.roughness_user_set,
                "elevation"     : geom.elevation_change,
                "k_fittings"    : geom.n_fittings_K,
            },
            "fluid"         : {
                "density"          : fluid.density,
                "density_default"  : not fluid.density_user_set,
                "viscosity"        : fluid.viscosity,
            },
            "node_pressures": {
                conn.source: round(src_P / 1000, 3),
                conn.target: round(tgt_P / 1000, 3),
            },
        })
    except Exception as e:
        return _err(str(e))


@app.route("/api/pump_power_plot")
def pump_power_plot():
    """Time-series pump power chart across all pipes."""
    if not _last_result:
        return _err("No simulation result.")
    try:
        plant  = _get_plant()
        t_end  = float(request.args.get("t_end", 300.0))
        series = {}
        from transportsim.pump import PumpTimeSeries
        import numpy as np
        for cname, conn in plant.connections.items():
            if conn.pump.state:
                ts = conn.pump.power_demand_over_time(
                    t_span=(0.0, t_end), n_points=80,
                    node_P_inlet_Pa=plant._node_pressure_Pa(conn.source),
                    node_P_outlet_Pa=plant._node_pressure_Pa(conn.target),
                )
                series[cname] = ts
        if not series:
            return _err("No pump data available. Run simulation first.")
        plot_b64 = plot_pump_power_over_time(series)
        summary  = {k: v.to_dict() for k, v in series.items()}
        return _ok({"plot": plot_b64, "series": summary})
    except Exception as e:
        return _err(str(e))


# ── Optimization ──────────────────────────────────────────────────────────────

@app.route("/api/optimize", methods=["POST"])
def optimize():
    global _last_opt, _last_result
    data   = request.get_json() or {}
    method = data.get("method", "slsqp")
    mode   = data.get("mode",   "pareto")
    pop    = int(data.get("pop_size",    20))
    n_gen  = int(data.get("n_gen",       10))
    t_end  = float(data.get("sim_t_end", data.get("t_end", 200.0)))
    n_seg  = int(data.get("sim_n_segments", data.get("n_segments", 6)))

    plant = _get_plant()
    if not plant.cstr_nodes:
        return _err("No CSTR reactors defined.")

    if _last_result is None:
        try:
            _last_result = plant.simulate(t_end=t_end, n_segments=max(n_seg, 4))
        except Exception as e:
            return _err(f"Baseline simulation failed: {e}")

    first_cstr = list(plant.cstr_nodes.keys())[0]
    raw_objs   = data.get("objectives", [])
    objectives = []
    for o in raw_objs:
        otype = o.get("type", "yield").replace("_objective", "")
        rname = o.get("reactor", first_cstr)
        sp    = o.get("species", "")
        dirn  = o.get("direction", "maximize")
        if rname not in plant.cstr_nodes:
            rname = first_cstr
        if otype == "yield" and sp:
            objectives.append(yield_objective(rname, sp, dirn))
        elif otype == "conversion" and sp:
            objectives.append(conversion_objective(rname, sp, dirn))
        elif otype == "residence_time":
            objectives.append(residence_time_objective(rname, dirn))
        elif otype == "compressor_power":
            objectives.append(total_compressor_power_objective(dirn))

    if not objectives:
        first_reactor = list(plant.cstr_nodes.values())[0]
        sp_list = list(first_reactor.reaction_network.species_names)
        if sp_list:
            objectives = [
                yield_objective(first_cstr, sp_list[-1], "maximize"),
                residence_time_objective(first_cstr, "minimize"),
            ]
        else:
            return _err("No species defined.")

    raw_dvars = data.get("decision_variables", [])
    dvars = []
    for dv in raw_dvars:
        vtype = dv.get("target_type") or dv.get("variable_type", "reactor_volume")
        vname = dv.get("target_name") or dv.get("target", first_cstr)
        if vname not in plant.cstr_nodes and vname not in plant.connections:
            vname = first_cstr
        dvars.append(DecisionVariable(
            name=dv.get("name", "dv"), target_type=vtype, target_name=vname,
            sub_target=dv.get("sub_target", ""),
            lower=float(dv.get("lower", 50.0)), upper=float(dv.get("upper", 1000.0)),
        ))

    if not dvars:
        for rname in plant.cstr_nodes:
            dvars.append(DecisionVariable(
                name="V_" + rname, target_type="reactor_volume",
                target_name=rname, lower=10.0, upper=2000.0
            ))

    try:
        optimizer = NetworkOptimizer(
            plant=plant, decision_variables=dvars,
            objectives=objectives, sim_t_end=t_end, sim_n_segments=n_seg,
        )
        if mode == "pareto" or method in ("nsga2", "weighted"):
            result = optimizer.optimize_pareto(pop_size=pop, n_gen=n_gen)
        else:
            result = optimizer.optimize_single(method=method)
        _last_opt    = result
        _last_result = plant.simulate(t_end=t_end, n_segments=n_seg)
        return _ok(result.to_dict())
    except Exception as e:
        return _err(f"Optimization failed: {e}\n{traceback.format_exc()}")


# ── Example plant builder ─────────────────────────────────────────────────────

@app.route("/api/build_example", methods=["POST"])
def build_example():
    global _plant, _last_result, _last_opt
    if not _HAS_CHEMSIM:
        return _err("ChemSim not installed.")
    if not _HAS_TS:
        return _err("TransportSim not installed.")

    data    = request.get_json() or {}
    example = data.get("example", "cascade")

    if example == "cascade":
        plant = _build_cascade_example()
    elif example == "parallel":
        plant = _build_parallel_example()
    else:
        plant = _build_cascade_example()

    _plant = plant
    _last_result = _last_opt = None
    return _ok(plant.to_dict())


def _build_cascade_example() -> PlantNetwork:
    plant = PlantNetwork("3-CSTR Cascade Example")

    # Source node
    src = SourceSink("Feed", node_type="source",
                     species={"A": 2.0}, flow_rate_m3s=5e-4,
                     pressure_Pa=3.5e5, pressure_mode="fixed")
    plant.add_source_sink(src)

    rn1 = ReactionNetwork()
    rn1.add_species("A", initial=2.0); rn1.add_species("B", initial=0.0)
    rn1.add_reaction(["A"], ["B"], rate=0.05, activation_energy=30000.0, pre_exponential=1e6)
    r1 = CSTR("CSTR-1", volume_L=200.0, reaction_network=rn1,
               temperature_gradient=TemperatureGradient(
                   mode="ramp", T_initial_K=303.15, T_final_K=343.15, t_ramp=200.0))
    r1.add_feed(FeedStream("primary", {"A": 2.0}, flow_rate_m3s=5e-4))
    r1.pressure_Pa   = 101325.0
    r1.pressure_mode = "atm"

    rn2 = ReactionNetwork()
    rn2.add_species("B", initial=0.0); rn2.add_species("C", initial=0.0)
    rn2.add_species("D", initial=0.0)
    rn2.add_reaction(["B"], ["C"], rate=0.08)
    rn2.add_reaction(["B"], ["D"], rate=0.02)
    r2 = CSTR("CSTR-2", volume_L=150.0, reaction_network=rn2,
               temperature_gradient=TemperatureGradient(mode="isothermal", T_initial_K=333.15))
    r2.pressure_Pa = 101325.0; r2.pressure_mode = "atm"

    rn3 = ReactionNetwork()
    rn3.add_species("C", initial=0.0); rn3.add_species("E", initial=0.0)
    rn3.add_species("F", initial=0.0)
    rn3.add_reaction(["C"], ["E"], rate=0.12, activation_energy=25000.0, pre_exponential=5e5)
    rn3.add_reaction(["C", "E"], ["F"], rate=0.005)
    r3 = CSTR("CSTR-3", volume_L=300.0, reaction_network=rn3,
               temperature_gradient=TemperatureGradient(
                   mode="step", T_initial_K=313.15, T_final_K=353.15, t_step=150.0))
    r3.pressure_Pa = 101325.0; r3.pressure_mode = "atm"

    sink = SourceSink("Product", node_type="sink", pressure_Pa=101325.0, pressure_mode="atm")
    plant.add_source_sink(sink)

    for r in [r1, r2, r3]:
        plant.add_reactor(r)

    for src_n, tgt_n, L, D, elev, K, Q in [
        ("Feed",   "CSTR-1", 15.0, 0.05, 0.5, 2, 5e-4),
        ("CSTR-1", "CSTR-2", 30.0, 0.06, 1.5, 3, 5e-4),
        ("CSTR-2", "CSTR-3", 20.0, 0.05,-0.5, 2, 5e-4),
        ("CSTR-3", "Product",10.0, 0.05, 0.0, 1, 5e-4),
    ]:
        spec = PipelineSpec(length=L, diameter=D, elevation_change=elev,
                            n_fittings_K=K, roughness_user_set=False)
        pipe = Pipeline(f"{src_n}→{tgt_n}", src_n, tgt_n,
                        spec=spec, fluid=FluidProperties(), flow_rate_m3s=Q)
        plant.add_connection(Connection(pipe, pump_efficiency=0.75))

    return plant


def _build_parallel_example() -> PlantNetwork:
    plant = PlantNetwork("Parallel CSTR Example")

    for tag, sp_in, sp_out, vol, rate, P_Pa in [
        ("CSTR-A", "A", "B", 120.0, 0.07, 101325.0),
        ("CSTR-B", "X", "Y", 180.0, 0.05, 101325.0),
    ]:
        src = SourceSink(f"Src-{tag}", node_type="source",
                         species={sp_in: 1.5}, flow_rate_m3s=3e-4,
                         pressure_Pa=2e5, pressure_mode="fixed")
        plant.add_source_sink(src)

        rn = ReactionNetwork()
        rn.add_species(sp_in, initial=1.5); rn.add_species(sp_out, initial=0.0)
        rn.add_reaction([sp_in], [sp_out], rate=rate,
                        activation_energy=20000.0, pre_exponential=5e5)
        r = CSTR(tag, volume_L=vol, reaction_network=rn,
                 temperature_gradient=TemperatureGradient(
                     mode="isothermal", T_initial_K=318.15))
        r.add_feed(FeedStream("feed", {sp_in: 1.5}, flow_rate_m3s=3e-4))
        r.pressure_Pa = P_Pa; r.pressure_mode = "atm"
        plant.add_reactor(r)

        spec = PipelineSpec(length=10.0, diameter=0.05, roughness_user_set=False)
        pipe = Pipeline(f"Src-{tag}→{tag}", f"Src-{tag}", tag,
                        spec=spec, fluid=FluidProperties(), flow_rate_m3s=3e-4)
        plant.add_connection(Connection(pipe, pump_efficiency=0.75))

    rn_m = ReactionNetwork()
    rn_m.add_species("B", initial=0.0); rn_m.add_species("Y", initial=0.0)
    rn_m.add_species("P", initial=0.0)
    rn_m.add_reaction(["B", "Y"], ["P"], rate=0.04)
    r_m = CSTR("CSTR-Mix", volume_L=250.0, reaction_network=rn_m,
                temperature_gradient=TemperatureGradient(mode="isothermal", T_initial_K=323.15))
    r_m.pressure_Pa = 101325.0; r_m.pressure_mode = "atm"
    plant.add_reactor(r_m)

    sink = SourceSink("Sink-Out", node_type="sink", pressure_Pa=101325.0)
    plant.add_source_sink(sink)

    for src_n, tgt_n in [("CSTR-A", "CSTR-Mix"), ("CSTR-B", "CSTR-Mix")]:
        spec = PipelineSpec(length=15.0, diameter=0.05, n_fittings_K=2, roughness_user_set=False)
        pipe = Pipeline(f"{src_n}→{tgt_n}", src_n, tgt_n,
                        spec=spec, fluid=FluidProperties(), flow_rate_m3s=3e-4)
        plant.add_connection(Connection(pipe, pump_efficiency=0.75))

    spec = PipelineSpec(length=8.0, diameter=0.05, roughness_user_set=False)
    pipe = Pipeline("CSTR-Mix→Sink-Out", "CSTR-Mix", "Sink-Out",
                    spec=spec, fluid=FluidProperties(), flow_rate_m3s=6e-4)
    plant.add_connection(Connection(pipe, pump_efficiency=0.75))

    return plant


# ── Sync plant from UI state ──────────────────────────────────────────────────

@app.route("/api/sync_plant", methods=["POST"])
def sync_plant():
    """
    Rebuild backend plant from UI node/edge state.
    v2: source and sink nodes become SourceSink objects.
    """
    global _plant, _last_result, _last_opt
    if not _HAS_CHEMSIM:
        return _err("ChemSim not installed.")
    if not _HAS_TS:
        return _err("TransportSim not installed.")

    data  = request.get_json() or {}
    name  = data.get("name", "Custom Plant")
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    nodes_by_label = {n["label"]: n for n in nodes}
    plant = PlantNetwork(name)

    # ── 1. Source/Sink nodes ──────────────────────────────────────────────────
    for node in nodes:
        ntype = node.get("type", "")
        if ntype not in ("source", "sink"):
            continue
        label  = node["label"]
        cfg    = node.get("config", {})
        P_kPa  = float(cfg.get("pressure_kPa", 101.325))
        P_mode = cfg.get("pressure_mode", "atm")
        species_raw = cfg.get("species", {})
        # Coerce list → dict if needed
        if isinstance(species_raw, list):
            species_raw = {s.get("name", "A"): float(s.get("conc", 1.0))
                           for s in species_raw if s.get("name")}
        Q      = float(cfg.get("flow_rate_m3s", 5e-4))

        ss = SourceSink(
            label, node_type=ntype,
            pressure_Pa=P_kPa * 1000.0,
            pressure_mode=P_mode,
            species=dict(species_raw),
            flow_rate_m3s=Q,
        )
        plant.add_source_sink(ss)

    # ── 2. CSTR nodes ─────────────────────────────────────────────────────────
    cstr_nodes = [n for n in nodes if n.get("type") == "cstr"]
    for node in cstr_nodes:
        cfg   = node.get("config", {})
        label = node["label"]

        rn       = ReactionNetwork()
        sp_added = set()

        for sp in cfg.get("species", []):
            sp_name = sp.get("name", "")
            if sp_name and sp_name not in sp_added:
                rn.add_species(sp_name, initial=float(sp.get("initial", 0.0)))
                sp_added.add(sp_name)

        for feed in cfg.get("feeds", []):
            for sp_name in feed.get("species", {}).keys():
                if sp_name not in sp_added:
                    rn.add_species(sp_name, initial=0.0)
                    sp_added.add(sp_name)

        # Pull species from connected source nodes
        for edge in edges:
            if edge.get("tgt_label", "") != label:
                continue
            src_node = nodes_by_label.get(edge.get("src_label", ""))
            if src_node and src_node.get("type") == "source":
                for sp_name in src_node.get("config", {}).get("species", {}).keys():
                    if sp_name not in sp_added:
                        rn.add_species(sp_name, initial=0.0)
                        sp_added.add(sp_name)

        if not sp_added:
            rn.add_species("A", initial=2.0)
            rn.add_species("B", initial=0.0)
            sp_added = {"A", "B"}

        # Reactions
        for rxn in cfg.get("reactions", []):
            reactants_raw = rxn.get("reactants", "")
            products_raw  = rxn.get("products",  "")
            reactants = ([s.strip() for s in reactants_raw.split() if s.strip()]
                         if isinstance(reactants_raw, str) else list(reactants_raw))
            products  = ([s.strip() for s in products_raw.split() if s.strip()]
                         if isinstance(products_raw, str) else list(products_raw))
            all_sp    = set(rn.species_names)
            reactants = [r for r in reactants if r in all_sp]
            products  = [p for p in products  if p in all_sp]
            if not reactants and not products:
                continue
            try:
                rn.add_reaction(
                    reactants or [rn.species_names[0]],
                    products  or [rn.species_names[-1]],
                    rate=float(rxn.get("rate", 0.05)),
                    activation_energy=float(rxn.get("activation_energy", 0.0)),
                    pre_exponential=float(rxn.get("pre_exponential", 0.0)),
                )
            except Exception:
                pass

        if not rn.reactions:
            sp_list = rn.species_names
            if len(sp_list) >= 2:
                try:
                    rn.add_reaction([sp_list[0]], [sp_list[1]], rate=0.05)
                except Exception:
                    pass

        # Temperature
        t_cfg = cfg.get("temperature", {})
        if isinstance(t_cfg, dict):
            mode = t_cfg.get("mode", "isothermal")
            T0   = float(t_cfg.get("T_kelvin", 298.15))
            Tf   = float(t_cfg.get("T_final",  T0))
            ts   = float(t_cfg.get("t_step",   100.0))
            tr   = float(t_cfg.get("t_ramp",   200.0))
            tg_map = {
                "ramp"      : TemperatureGradient(mode="ramp", T_initial_K=T0, T_final_K=Tf, t_ramp=tr),
                "step"      : TemperatureGradient(mode="step", T_initial_K=T0, T_final_K=Tf, t_step=ts),
            }
            tg = tg_map.get(mode, TemperatureGradient(mode="isothermal", T_initial_K=T0))
        else:
            tg = TemperatureGradient(mode="isothermal", T_initial_K=298.15)

        P_kPa  = float(cfg.get("pressure_kPa", 101.325))
        P_mode = cfg.get("pressure_mode", "atm")
        cstr   = CSTR(label, volume_L=float(cfg.get("volume_L", 200.0)),
                      reaction_network=rn, temperature_gradient=tg)
        cstr.pressure_Pa   = P_kPa * 1000.0
        cstr.pressure_mode = P_mode

        # Explicit feeds
        for feed in cfg.get("feeds", []):
            try:
                cstr.add_feed(FeedStream(
                    name=feed.get("name", "feed"),
                    compositions=feed.get("species", {}),
                    flow_rate_m3s=float(feed.get("flow_rate_m3s", 5e-4)),
                ))
            except Exception:
                pass

        # Feeds from source nodes
        for edge in edges:
            if edge.get("tgt_label", "") != label:
                continue
            src_node = nodes_by_label.get(edge.get("src_label", ""))
            if not src_node or src_node.get("type") != "source":
                continue
            src_cfg  = src_node.get("config", {})
            feed_sp  = src_cfg.get("species", {})
            if isinstance(feed_sp, list):
                feed_sp = {s.get("name", "A"): float(s.get("conc", 1.0)) for s in feed_sp}
            if feed_sp:
                try:
                    cstr.add_feed(FeedStream(
                        name=src_node["label"],
                        compositions=feed_sp,
                        flow_rate_m3s=float(src_cfg.get("flow_rate_m3s", 5e-4)),
                    ))
                except Exception:
                    pass

        if not cstr.feeds:
            sp_list = rn.species_names
            if sp_list:
                try:
                    cstr.add_feed(FeedStream("default", {sp_list[0]: 2.0}, flow_rate_m3s=5e-4))
                except Exception:
                    pass

        plant.add_reactor(cstr)

    if not plant.cstr_nodes and not plant.source_sink_nodes:
        return _err("No nodes found in plant definition.")

    # ── 3. Pipeline connections ───────────────────────────────────────────────
    for edge in edges:
        src_label = edge.get("src_label", "")
        tgt_label = edge.get("tgt_label", "")
        if src_label not in plant.reactors or tgt_label not in plant.reactors:
            continue
        ecfg      = edge.get("config", {})
        pipe_name = f"{src_label}\u2192{tgt_label}"

        roughness_raw = ecfg.get("roughness")
        roughness     = float(roughness_raw) if roughness_raw is not None else 4.6e-5
        density_raw   = ecfg.get("density")
        density       = float(density_raw) if density_raw is not None else 1000.0

        try:
            spec = PipelineSpec(
                length             = float(ecfg.get("length",            10.0)),
                diameter           = float(ecfg.get("diameter",          0.05)),
                roughness          = roughness,
                elevation_change   = float(ecfg.get("elevation_change",  0.0)),
                n_fittings_K       = int(round(float(ecfg.get("n_fittings_K", 2)))),
                roughness_user_set = roughness_raw is not None,
            )
            fluid = FluidProperties(
                density          = density,
                viscosity        = float(ecfg.get("viscosity", 1.002e-3)),
                density_user_set = density_raw is not None,
            )
            pipe = Pipeline(
                pipe_name, src_label, tgt_label,
                spec=spec, fluid=fluid,
                flow_rate_m3s=float(ecfg.get("flow_rate_m3s", 5e-4)),
            )
            eta  = float(ecfg.get("pump_efficiency", 0.75))
            conn = Connection(pipe, pump_efficiency=eta)
            plant.add_connection(conn)
        except Exception:
            pass

    _plant = plant
    _last_result = _last_opt = None
    return _ok({
        "name"            : plant.name,
        "n_nodes"         : len(plant.reactors),
        "n_reactors"      : len(plant.cstr_nodes),
        "n_source_sinks"  : len(plant.source_sink_nodes),
        "n_connections"   : len(plant.connections),
        "node_names"      : list(plant.reactors.keys()),
    })


# ── Misc endpoints ────────────────────────────────────────────────────────────

@app.route("/api/validate")
def validate():
    plant    = _get_plant()
    issues   = []
    warnings = []

    for name, node in plant.reactors.items():
        if isinstance(node, SourceSink):
            if node.pressure_mode == "atm":
                warnings.append(f"{name}: pressure not set — using atmospheric default.")
            if not node.species and node.node_type == "source":
                warnings.append(f"{name}: source has no species defined.")
            continue

        real_feeds = [f for f in node.feeds if not f.name.startswith("_from_")]
        if not real_feeds:
            upstream = [c for c in plant.connections.values() if c.target == name]
            if not upstream:
                warnings.append(f"{name}: No feed streams and no upstream connections.")

        if not node.reaction_network.species_names:
            issues.append(f"{name}: No species defined.")
        if not node.reaction_network.reactions:
            warnings.append(f"{name}: No reactions defined.")

    for cname, conn in plant.connections.items():
        if conn.source not in plant.reactors:
            issues.append(f"Pipeline '{cname}': source '{conn.source}' not found.")
        if conn.target not in plant.reactors:
            issues.append(f"Pipeline '{cname}': target '{conn.target}' not found.")
        if not conn.pipeline.spec.roughness_user_set:
            warnings.append(f"Pipeline '{cname}': roughness not set — using steel default.")
        if not conn.pipeline.fluid.density_user_set:
            warnings.append(f"Pipeline '{cname}': fluid density not set — using water default.")
        if conn.pump.state and not conn.pump.state.feasible:
            issues.append(
                f"Pipeline '{cname}': pump ΔP {conn.pump.required_delta_p_kPa:.1f} kPa "
                "is unreasonably high — check geometry."
            )

    return _ok({
        "valid"        : len(issues) == 0,
        "issues"       : issues,
        "warnings"     : warnings,
        "n_nodes"      : len(plant.reactors),
        "n_connections": len(plant.connections),
    })


@app.route("/api/balance", methods=["POST"])
def compute_balance():
    """
    Mass balance for a subset of node names.
    v2: includes source/sink nodes — they are no longer excluded.
    """
    global _last_result
    data       = request.get_json() or {}
    node_names = data.get("nodes", [])

    if not _last_result:
        return _err("Run simulation first.")

    plant   = _get_plant()
    result  = _last_result
    balance : dict = {}

    for nname in node_names:
        node = plant.reactors.get(nname)
        rr   = result.reactor_results.get(nname)
        if not node:
            continue

        # Source: contributes inflow
        if isinstance(node, SourceSink) and node.node_type == "source":
            for sp, c in node.species.items():
                mf = c * node.total_flow_Ls
                balance.setdefault(sp, {"in": 0.0, "out": 0.0, "generated": 0.0})
                balance[sp]["in"] += mf

        # Sink: contributes outflow
        elif isinstance(node, SourceSink) and node.node_type == "sink":
            if rr:
                for sp, c in rr.outlet_composition.items():
                    mf = c * node.total_flow_Ls
                    balance.setdefault(sp, {"in": 0.0, "out": 0.0, "generated": 0.0})
                    balance[sp]["out"] += mf

        # CSTR: standard inlet/outlet accounting
        elif _HAS_CSTR and isinstance(node, CSTR):
            if rr is None:
                continue
            for feed in node.feeds:
                if feed.name.startswith("_from_"):
                    continue
                for sp, c in feed.compositions.items():
                    mf = c * feed.flow_rate_Ls
                    balance.setdefault(sp, {"in": 0.0, "out": 0.0, "generated": 0.0})
                    balance[sp]["in"] += mf

            is_internal = any(
                conn.source == nname and conn.target in node_names
                for conn in plant.connections.values()
            )
            if not is_internal:
                Q_out = node.total_flow_Ls
                for sp, c in rr.outlet_composition.items():
                    mf = c * Q_out
                    balance.setdefault(sp, {"in": 0.0, "out": 0.0, "generated": 0.0})
                    balance[sp]["out"] += mf

    for sp in balance:
        b = balance[sp]
        b["generated"]   = b["out"] - b["in"]
        b["closure_pct"] = 100.0 * b["out"] / b["in"] if b["in"] > 0 else 0.0

    return _ok(balance)


@app.route("/api/balance_detail", methods=["GET"])
def balance_detail():
    plant  = _get_plant()
    result = _last_result
    if result is None:
        return _err("No simulation result.")
    detail = {}
    for nname, rr in result.reactor_results.items():
        node = plant.reactors.get(nname)
        if node is None:
            continue
        is_ss = isinstance(node, SourceSink)
        detail[nname] = {
            "node_type"         : node.node_type if is_ss else "cstr",
            "inlet_flows"       : [] if is_ss else [
                {
                    "feed_name"   : f.name,
                    "flow_Ls"     : round(f.flow_rate_Ls, 4),
                    "compositions": {sp: round(c, 4) for sp, c in f.compositions.items()},
                }
                for f in node.feeds
            ],
            "outlet_composition": {k: round(v, 4) for k, v in rr.outlet_composition.items()},
            "outlet_flow_Ls"    : round(getattr(node, "total_flow_Ls", 0.0), 4),
            "pressure_kPa"      : round(getattr(node, "pressure_kPa", 101.325), 3),
            "conversion"        : {k: round(v, 4) for k, v in
                                   (rr.conversion.items() if hasattr(rr, "conversion") else {}.items())},
            "error_state"       : nname in result.run_dry_reactors,
            "error_msg"         : result.stability_errors.get(nname, ""),
            "inlet_connections" : [
                {"conn": cn, "src": c.source, "flow_Ls": round(c.pipeline.flow_rate_m3s*1000, 4)}
                for cn, c in plant.connections.items() if c.target == nname
            ],
            "outlet_connections": [
                {"conn": cn, "tgt": c.target, "flow_Ls": round(c.pipeline.flow_rate_m3s*1000, 4)}
                for cn, c in plant.connections.items() if c.source == nname
            ],
        }
    return _ok({
        "nodes": detail,
        "total_pump_kW": result.total_pump_kW,
        "pump_breakdown": result.to_dict().get("pump_breakdown", {}),
    })


@app.route("/api/demand_series", methods=["POST"])
def demand_series():
    plant  = _get_plant()
    result = _last_result
    if result is None:
        return _err("No simulation result.")
    data       = request.get_json() or {}
    node_label = data.get("node_label", "")
    series     = {}
    for rname, rr in result.reactor_results.items():
        if not hasattr(rr, "time") or not rr.time.size:
            continue
        series[rname] = {
            "time"          : rr.time.tolist(),
            "species_names" : list(rr.species_names) if hasattr(rr, "species_names") else [],
            "concentrations": (rr.concentrations.tolist()
                               if hasattr(rr, "concentrations") else []),
            "outlet_flow_Ls": getattr(plant.reactors.get(rname), "total_flow_Ls", 0.0),
        }
    return _ok({
        "node_label"     : node_label,
        "reactor_series" : series,
        "source_pressures": {k: round(v/1000, 3) for k, v in result.source_pressures.items()},
        "total_pump_kW"  : result.total_pump_kW,
    })


@app.route("/api/propose_fix", methods=["POST"])
def propose_fix():
    plant  = _get_plant()
    result = _last_result
    if result is None:
        return _err("No simulation result.")
    if not result.run_dry_reactors and not result.stability_errors:
        return _ok({"proposals": [], "message": "No instability detected."})

    data        = request.get_json() or {}
    target_name = data.get("reactor_name", "")
    targets     = (
        [target_name] if target_name in result.run_dry_reactors
        else list(result.run_dry_reactors)
    )

    proposals = []
    for rname in targets:
        node = plant.reactors.get(rname)
        if not node or isinstance(node, SourceSink):
            continue
        rr        = result.reactor_results.get(rname)
        sp_names  = list(node.reaction_network.species_names)
        outlet_oc = rr.outlet_composition if rr else {}
        Q_in      = node.total_flow_m3s      # m³/s — actual inlet flow to this reactor
        mb        = result.material_balance

        # --- Downstream demand (what pipes are pulling from this reactor) ---
        downstream_Q_m3s = sum(
            c.pipeline.flow_rate_m3s
            for c in plant.connections.values()
            if c.source == rname
        )

        # --- Max sustainable outlet: inlet_Q * 0.95 (leave 5% buffer) ---
        # This is what downstream pipes should be throttled to
        sustainable_Q_Ls = round(Q_in * 0.95 * 1000, 4)

        # --- Alternatively: increase the feed to satisfy demand ---
        required_feed_Ls = round(downstream_Q_m3s * 1.05 * 1000, 4)

        # --- Volume increase: τ increases, more product at same flow ---
        proposed_vol = round(node.volume_L * 1.5, 1)

        proposals.extend([
            {
                "priority": 1,
                "type": "reduce_outlet_flow",
                "reactor": rname,
                "description": (
                    f"Throttle downstream pipe flow to match reactor capacity. "
                    f"Inlet: {Q_in*1000:.3f} L/s → set outlets to {sustainable_Q_Ls} L/s."
                ),
                "proposed_value": sustainable_Q_Ls,
                "field": "flow_rate_Ls",
                "feasible": Q_in > 1e-9,
            },
            {
                "priority": 2,
                "type": "increase_feed",
                "reactor": rname,
                "description": (
                    f"Increase inlet feed to meet downstream demand of {downstream_Q_m3s*1000:.3f} L/s. "
                    f"Required feed: {required_feed_Ls} L/s."
                ),
                "proposed_value": required_feed_Ls,
                "field": "feed_flow_Ls",
                "feasible": True,
            },
            {
                "priority": 3,
                "type": "increase_volume",
                "reactor": rname,
                "description": (
                    f"Increase reactor volume {node.volume_L:.0f} L → {proposed_vol:.0f} L "
                    f"(longer residence time increases conversion and production)."
                ),
                "proposed_value": proposed_vol,
                "field": "volume_L",
                "feasible": True,
            },
        ])

        if any(r.activation_energy > 0 for r in node.reaction_network.reactions):
            T_c = node.temperature_gradient.T_initial_K
            T_p = min(T_c + 20, 500)
            proposals.append({
                "priority": 4,
                "type": "increase_temperature",
                "reactor": rname,
                "description": (
                    f"Raise temperature {T_c:.1f} K → {T_p:.1f} K. "
                    "Arrhenius kinetics detected — higher T increases reaction rate and output."
                ),
                "proposed_value": round(T_p, 1),
                "field": "T_kelvin",
                "feasible": True,
            })

    proposals.sort(key=lambda p: p["priority"])
    return _ok({"reactor_errors": result.stability_errors, "proposals": proposals})


@app.route("/api/auto_balance", methods=["POST"])
def auto_balance():
    """
    Auto-balance: solve flow rates so no reactor runs dry.

    Strategy (iterative, up to 10 passes):
      1. Simulate to detect run-dry reactors.
      2. For each run-dry reactor:
         - Compute its actual sustainable outlet = inlet_Q * 0.90
         - Set all downstream pipe flow rates to that value (split equally if
           multiple outlets)
         - If reactor has no inlet feed, increase feed flow to match demand.
      3. Re-simulate. Repeat until converged or max iterations reached.

    Returns the final simulation result with a balance_log describing changes.
    """
    global _last_result
    data       = request.get_json() or {}
    t_end      = float(data.get("t_end",       300.0))
    n_segments = int(data.get("n_segments",    12))
    max_iter   = int(data.get("max_iter",      8))

    plant = _get_plant()
    if not plant.reactors:
        return _err("No nodes in plant. Add reactors first.")

    balance_log = []

    for iteration in range(max_iter):
        # Run simulation
        try:
            result = plant.simulate(t_end=t_end, n_segments=n_segments)
        except Exception as e:
            return _err(f"Simulation failed: {e}")

        _last_result = result

        if not result.run_dry_reactors:
            balance_log.append(f"Pass {iteration+1}: converged — no run-dry reactors.")
            break

        changed = False
        for rname in result.run_dry_reactors:
            node = plant.reactors.get(rname)
            if node is None or isinstance(node, SourceSink):
                continue

            inlet_Q  = node.total_flow_m3s   # m³/s — actual current inlet
            outlets  = [c for c in plant.connections.values() if c.source == rname]
            n_out    = len(outlets)

            if inlet_Q > 1e-9 and n_out > 0:
                # Throttle each outlet pipe to share inlet_Q equally with a safety margin
                safe_Q_each = (inlet_Q * 0.90) / n_out
                for conn in outlets:
                    old_q = conn.pipeline.flow_rate_m3s
                    conn.pipeline.flow_rate_m3s = safe_Q_each
                    balance_log.append(
                        f"Pass {iteration+1}: [{rname}] outlet '{conn.name}' "
                        f"{old_q*1000:.3f} → {safe_Q_each*1000:.3f} L/s"
                    )
                changed = True
            elif inlet_Q <= 1e-9:
                # No feed at all — boost inlet pipes or feeds
                inlets = [c for c in plant.connections.values() if c.target == rname]
                if inlets:
                    # Boost upstream pipe flows to satisfy downstream demand
                    downstream_Q = sum(c.pipeline.flow_rate_m3s for c in outlets)
                    boost_each   = (downstream_Q * 1.1) / len(inlets) if inlets else 0
                    for conn in inlets:
                        old_q = conn.pipeline.flow_rate_m3s
                        conn.pipeline.flow_rate_m3s = boost_each
                        balance_log.append(
                            f"Pass {iteration+1}: [{rname}] inlet '{conn.name}' "
                            f"{old_q*1000:.3f} → {boost_each*1000:.3f} L/s"
                        )
                    changed = True
                else:
                    # No upstream pipe either — boost reactor's direct feeds
                    if _HAS_CSTR and isinstance(node, CSTR):
                        downstream_Q = sum(c.pipeline.flow_rate_m3s for c in outlets)
                        for feed in node._feeds:
                            if not feed.name.startswith("_from_"):
                                old_q = feed.flow_rate_m3s
                                feed.flow_rate_m3s = downstream_Q * 1.1
                                balance_log.append(
                                    f"Pass {iteration+1}: [{rname}] feed '{feed.name}' "
                                    f"{old_q*1000:.3f} → {feed.flow_rate_m3s*1000:.3f} L/s"
                                )
                                changed = True
                                break

        if not changed:
            balance_log.append(f"Pass {iteration+1}: no adjustable connections found.")
            break
    else:
        balance_log.append(
            f"Reached max iterations ({max_iter}). Network may need manual adjustment."
        )

    # Final simulation with balanced flows
    try:
        result = plant.simulate(t_end=t_end, n_segments=n_segments)
        _last_result = result
    except Exception as e:
        return _err(f"Final simulation failed: {e}")

    resp = result.to_dict()
    resp["balance_log"]   = balance_log
    resp["iterations"]    = iteration + 1
    resp["still_dry"]     = result.run_dry_reactors
    resp["converged"]     = len(result.run_dry_reactors) == 0
    return _ok(resp)


    return _ok(_get_plant().species_properties)


@app.route("/api/species_properties", methods=["POST"])
def set_species_properties():
    plant = _get_plant()
    data  = request.get_json() or {}
    for sp, props in data.items():
        plant.species_properties.setdefault(sp, {}).update(props)
    densities = [v["density_kg_m3"] for v in plant.species_properties.values()
                 if "density_kg_m3" in v]
    avg_density = sum(densities) / len(densities) if densities else None
    if avg_density:
        for conn in plant.connections.values():
            conn.pipeline.fluid.density = avg_density
            conn.pipeline.fluid.density_user_set = True
    return _ok(plant.species_properties)


@app.route("/api/project")
def get_project():
    plant = _get_plant()
    return _ok({
        "version"    : "2.0.0",
        "name"       : plant.name,
        "nodes"      : {n: node.to_dict() for n, node in plant.reactors.items()},
        "connections": {n: c.to_dict() for n, c in plant.connections.items()},
        "topology"   : plant.topology(),
        "result"     : _last_result.to_dict() if _last_result else None,
    })


@app.route("/api/project/name", methods=["POST"])
def set_project_name():
    data  = request.get_json() or {}
    plant = _get_plant()
    plant.name = data.get("name", plant.name)
    return _ok({"name": plant.name})


@app.route("/api/readme")
def get_readme():
    for p in ["/home/claude/README.md", "/mnt/user-data/outputs/README.md"]:
        if os.path.exists(p):
            with open(p) as f:
                return _ok({"content": f.read(), "path": p})
    return _ok({"content": "README not found.", "path": ""})


# ── Entry point ───────────────────────────────────────────────────────────────

def run_server(host="0.0.0.0", port=5050, debug=False):
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
