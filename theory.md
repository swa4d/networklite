# NetworkLite v3.0 — Developer System Prompt

This application combines **ChemSim** (reaction kinetics, ODE and SSA solvers) and **TransportSim** (pipeline hydraulics, C++ + Python) with a Flask REST API and a single-page HTML/JS dashboard.

---

## 1. Architecture Overview

### Entry Point
```
python run.py [--host 0.0.0.0] [--port 5050] [--debug]
```
Flask serves `dashboard/index.html` at `/` and `dashboard/optimizer.html` at `/optimizer` (loaded in an iframe). All `/api/*` routes are served by the same process.

### Backend: Flask (`dashboard/app.py`, ~1 880 lines)
- Single-process, in-memory plant state (`_plant`, `_last_result`, `_last_opt`, `_plot_cache` globals)
- Structured logging to stdout (INFO) and `dashboard/networklite.log` (DEBUG) via Python `logging`
- CORS enabled via `flask_cors`
- All plant state is rebuilt from the frontend's canonical `PLANT` object on every `/api/sync_plant` call before simulation
- v2 unified node API: `/api/nodes` and `/api/reactors` are aliases; source/sink nodes are first-class

### Frontend: Single-page app (`dashboard/index.html`, ~3 600 lines)
- Vanilla JS — no framework, no build step
- **Plotly.js 2.32** loaded from CDN (deferred) — used for the 3-D stochastic PDF surface
- All canvas rendering via the HTML5 2-D Canvas API (concentration timelines, temperature profiles, pipe flow sweep, Pareto front, sink flow chart, dashboard slots)
- Three-tab layout: **Control Center** / **Dashboard** / **Optimizer** (iframe)
- Google Fonts: Orbitron (display), Barlow (body), Fira Code (monospace)

### Package Structure
```
networklite/
├── run.py                                 ← entry point (Flask dev server)
├── chemsim/                               ← chemical simulation package
│   ├── __init__.py
│   ├── network.py         (444 lines)     ← ReactionNetwork, Species, Reaction dataclasses
│   ├── simulator.py       (445 lines)     ← Simulator class + _run_scipy() BDF fallback
│   ├── analysis.py        (307 lines)     ← steady-state detection, peak-finding
│   ├── plotting.py        (368 lines)     ← matplotlib trajectory and phase-portrait helpers
│   ├── writer.py          (277 lines)     ← HDF5 / CSV export
│   ├── loader.py          (317 lines)     ← HDF5 / CSV import
│   ├── renderer.py        (383 lines)     ← particle visualisation (pygame / OpenGL)
│   └── conf.py             (49 lines)     ← solver defaults
├── network/
│   ├── plant.py           (685 lines)     ← PlantNetwork, NetworkSimulationResult, FigureRegistry
│   ├── reactor/
│   │   ├── cstr.py        (680 lines)     ← CSTR, FeedStream, TemperatureGradient, CSTRResult
│   │   └── source_sink.py (230 lines)     ← SourceSink, SourceSinkResult
│   ├── pipeline/
│   │   └── connection.py  (150 lines)     ← Connection, ConnectionDiagnostic
│   ├── optimizer/
│   │   └── multi_objective.py (516 lines) ← NetworkOptimizer, DecisionVariable, Objective, NSGA-II
│   └── analysis/
│       └── diagnostics.py (535 lines)     ← matplotlib plot generators for dashboard
├── transportsim/
│   ├── core/
│   │   ├── transportsim.hpp               ← C++ hydraulics (Darcy-Weisbach, Colebrook-White)
│   │   ├── bindings.cpp                   ← pybind11 → _transportsim_core
│   │   └── setup.py                       ← build script
│   ├── pipeline.py        (350 lines)     ← Pipeline, PipelineSpec, FluidProperties
│   ├── pump.py                            ← PumpModel, PumpState, PumpTimeSeries
│   ├── pressure_solver.py                 ← NetworkPressureSolver (iterative auto-balance)
│   ├── sweep.py                           ← all pressure/pump matplotlib plot functions
│   ├── flow_regimes.py                    ← Re → regime classification
│   └── __init__.py                        ← public re-exports
├── gillespie.py           (270 lines)     ← v3.0 Gillespie SSA + τ-leaping engine (NEW)
└── dashboard/
    ├── app.py             (~1 880 lines)  ← Flask application
    ├── index.html         (~3 600 lines)  ← single-page frontend
    └── optimizer.html                     ← standalone optimizer page (served in iframe)
```

### Critical Import Rule
Every file using `from __future__ import annotations` **must** have it as the **absolute first line** — before path-injection blocks, docstrings, and everything else. Placing anything before it causes a `SyntaxError` at import time.

---

## 2. CSTR Equations

### 2.1 Residence Time
```
τ = V / Q_total    [seconds]
V        = reactor volume (L)
Q_total  = Σ feed flow rates (L/s)  — active feeds only, no double-counting
```
`Q_total` is computed from `self._feeds`. When a `SourceSink` source routes into a CSTR, `_route_outlet_to_feeds` clears any existing explicit feeds from that source direction first, then injects `_from_<SourceName>` at the pipe's flow rate.

### 2.2 Mixed Inlet Composition
```
C_in(species) = Σ_feeds [ C_feed_i(species) × (Q_feed_i / Q_total) ]
```
Flow-weighted average across all active feed streams.

### 2.3 Reaction Rate Law (power-law)
```
r_j = k_j(T) · ∏_i [C_i]^ν_ij
```
- `k_j` = rate constant (units depend on order; 1/s for first-order)
- `ν_ij` = stoichiometric coefficient of species i in reaction j (reactant side)
- Concentrations clipped to ≥ 0 before exponentiation

### 2.4 Arrhenius Temperature Dependence
Two forms, applied in both the ChemSim C++ core and the `_run_scipy` fallback:
```
# Form 1: explicit pre-exponential (A > 0, Ea > 0)
k(T) = A · exp(-Ea / (R · T))

# Form 2: correction from reference rate at 298.15 K (A = 0, Ea > 0)
k(T) = k_ref · exp(-Ea/R · (1/T - 1/298.15))
```
`R = 8.314 J/(mol·K)`. Applied at the mid-point of each time segment (`t_mid = t_offset + Δt/2`).

### 2.5 CSTR Dynamic ODE — Deterministic (species balance)
The ODE solved by ChemSim (C++ BDF or scipy BDF fallback):
```
dC_i/dt = Σ_j [ ν_ij · r_j(C, T) ]
```
This is the **batch reactor ODE** (no dilution term in the integrator). Dilution is applied as a post-processing correction after each segment.

### 2.6 CSTR Dilution Correction (per segment)
After solving the batch ODE over one time segment `Δt = t_end / n_segments`:
```
C_out(t+Δt) = C_batch(Δt) · exp(-Δt/τ)  +  C_in · (1 - exp(-Δt/τ))
```
- `α = exp(-Δt/τ)` — mixing fraction
- `τ → ∞` (no flow): `α → 1`, `C_out = C_batch` (pure batch)
- `τ → 0` (very fast flow): `α → 0`, `C_out = C_in` (pass-through)

### 2.7 Temperature Profiles (piecewise, applied per segment)
```
isothermal:  T(t) = T₀
step:        T(t) = T₀  for t < t_step;  T_final  for t ≥ t_step
ramp:        T(t) = T₀ + (T_final − T₀) · (t / t_ramp)
custom:      T(t) = piecewise linear from (times[], temps[]) arrays
```
Each of the `n_segments` uses `T` at `t_mid = t_offset + Δt/2`. Segment count capped at 30 to prevent ODE step exhaustion.

### 2.8 Conversion
```
X_i = max(0, (C_in_i − C_out_i) / C_in_i)   for each species present in inlet
```

### 2.9 Steady-State Convergence Criterion
Used by `/api/seek_steady_state`:
```
CoV_i = std(C_i[last 20%]) / mean(C_i[last 20%])
Converged when CoV_i < tol for ALL species in ALL CSTRs
```
Default `tol = 0.01` (1%). Iterates by doubling `t_end` up to `t_max`.

---

## 3. Gillespie SSA Equations (v3.0 — Stochastic Mode)

### 3.1 System-Size Parameter Ω
```
N_i = round(C_i × Ω)       [molecules]
C_i = N_i / Ω              [mol/L]
```
`Ω` is the noise-to-speed trade-off: higher Ω = less stochastic noise, slower simulation. Default Ω = 300.

### 3.2 Stochastic Rate Constant
```
c_j = k_j / ( Ω^(ord_j − 1) × ∏_i ν_ij! )
ord_j = Σ_i ν_ij   (reaction order)
```

### 3.3 Propensity Function (falling factorial)
```
a_j = c_j × ∏_i  [ N_i! / (N_i − ν_ij)! ]   (falling factorial)
a_j = 0 if any N_i < ν_ij
```

### 3.4 CSTR Open-Boundary Pseudo-Reactions
```
Inflow  ∅ → X_i :  a_in_i  = D × C_in_i × Ω     D = Q/V [s⁻¹]
Outflow X_i → ∅ :  a_out_i = D × N_i
```

### 3.5 τ-Leaping Step Size (Cao et al. 2006)
```
μ_i   = Σ_j ν_ij · a_j        (expected drift per species)
σ²_i  = Σ_j ν_ij² · a_j       (variance per species)

τ = min_i min( ε·max(N_i,1)/|μ_i|,  ε²·max(N_i,1)²/σ²_i )
```
`ε = 0.03` default. If `τ · a₀ < 10`: fall back to exact Direct Method for that step (one event, exponential waiting time).

### 3.6 Ensemble Output
N_traj independent trajectories produce: ensemble mean, std, p5, p95, and per-species probability density surfaces `P(C, t)` for the Plotly `surface3d` visualisation.

---

## 4. TransportSim Hydraulics Equations

### 4.1 Darcy-Weisbach (friction pressure drop)
```
ΔP_friction = f_D · (L/D) · (ρv²/2)    [Pa]
v = Q / A,   A = π(D/2)²,   Re = ρvD/μ
```

### 4.2 Colebrook-White (Darcy friction factor, iterative Newton)
```
1/√f = −2 log₁₀( ε/(3.7D) + 2.51/(Re·√f) )
```
Initial guess: Swamee-Jain explicit approximation. Laminar fallback: `f = 64/Re` for `Re < 2300`.

### 4.3 Fanning Friction Factor
```
f_Fanning = f_Darcy / 4
```

### 4.4 Total Pressure Drop
```
ΔP_total = ΔP_friction + ΔP_minor + ΔP_gravity
ΔP_minor   = K_total · (ρv²/2)        [K = sum of fitting K-values]
ΔP_gravity = ρ · g · Δz
```

### 4.5 Auto-Balance Pump Solver (per pipe)
```
ΔP_pump = ΔP_total + (P_outlet_node − P_inlet_node)
W_shaft = Q · ΔP_pump / η              [W]
```
- All nodes (CSTR + SourceSink) operate at **constant pressure** (default P_atm = 101 325 Pa)
- Every pipe has a virtual pump at its inlet providing exactly `ΔP_pump`
- `η` = per-pipe pump efficiency (default 0.75, user-editable)
- Flagged infeasible if `ΔP_pump > 5 000 kPa`

### 4.6 Flow Regime Classification
| Re | Regime |
|---|---|
| < 2 300 | Laminar (Hagen-Poiseuille, `f = 64/Re`) |
| 2 300–4 000 | Transitional (friction factor ±40% uncertainty) |
| ≥ 4 000 | Turbulent (Colebrook-White reliable) |

---

## 5. Node Types

### CSTR (`network/reactor/cstr.py`)
| Property | Notes |
|---|---|
| `volume_L` | Reactor volume |
| `reaction_network` | ChemSim `ReactionNetwork` |
| `temperature_gradient` | `TemperatureGradient` (isothermal / ramp / step / custom) |
| `_feeds` | List of `FeedStream`; auto-feeds prefixed `_from_<name>` |
| `outlet_composition` | Dict updated after each `.simulate()` call |
| `pressure_Pa` / `pressure_mode` | Node operating pressure for pump solver |
| `simulate(mode=, stochastic_params=)` | v3: dispatches to ODE or Gillespie path |

`CSTRResult` fields: `time`, `concentrations`, `species_names`, `outlet_composition`, `residence_time_s`, `conversion`, `temperature_profile`, `converged`, `issues`, `simulation_mode` (v3), `stochastic_data` (v3).

### SourceSink (`network/reactor/source_sink.py`)
Infinite-volume source or sink. No reactions. Composition fixed by `species` dict. Implements the same `.simulate()` interface as CSTR (returns `SourceSinkResult` with constant trajectory). `node_type = "source" | "sink"` is a UI rendering hint only — both are treated identically by the pressure solver and mass-balance tool.

### Connection (`network/pipeline/connection.py`)
Wraps a `Pipeline` (TransportSim) and a `PumpModel`. `.diagnose(src_P, tgt_P)` returns a `ConnectionDiagnostic` with fields: `pressure_drop_kPa`, `velocity_m_s`, `reynolds_number`, `flow_regime`, `outlet_pressure_bar`, `needs_compressor`, `compressor_kW`.

---

## 6. Flask API Reference

### Static
| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Serve `index.html` |
| `GET` | `/optimizer` | Serve `optimizer.html` |

### Status & Topology
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/status` | Health check: plant name, node counts, ChemSim/TS availability |
| `GET` | `/api/topology` | Reactor graph: nodes + edges + pump data per edge |

### Node Management (v2 unified — `/api/nodes` and `/api/reactors` are aliases)
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/nodes` | List all nodes (CSTR + SourceSink) |
| `GET` | `/api/nodes/<n>` | Inspect one node |
| `POST` | `/api/nodes/<n>/pressure` | Set node operating pressure |
| `POST` | `/api/nodes` | Add a CSTR |
| `DELETE` | `/api/nodes/<n>` | Remove a node |
| `POST` | `/api/reactors/<n>/feed` | Add/update a feed stream |
| `POST` | `/api/reactors/<n>/species` | Add species to reaction network |
| `POST` | `/api/reactors/<n>/reaction` | Add a reaction |
| `POST` | `/api/reactors/<n>/temperature` | Set temperature gradient |
| `POST` | `/api/reactors/<n>/volume` | Update reactor volume |

### Connections
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/connections` | List all pipelines |
| `GET` | `/api/connections/<n>` | Inspect one pipeline |
| `POST` | `/api/connections` | Add a pipeline connection |
| `DELETE` | `/api/connections/<n>` | Remove a pipeline |
| `POST` | `/api/connections/<n>/flow` | Update flow rate |
| `POST` | `/api/connections/<n>/pump` | Update pump efficiency |
| `GET` | `/api/connections/<n>/sweep` | Pump curve sweep (ΔP + power vs Q) |

### Simulation
| Method | Route | Description |
|---|---|---|
| `POST` | `/api/sync_plant` | Rebuild backend plant from UI node/edge state |
| `POST` | `/api/simulate` | Run full plant simulation. Body: `{t_end, n_segments, mode, n_trajectories, omega, tau_eps}` |
| `GET` | `/api/stochastic_status` | SSA availability + per-reactor simulation_mode from last result |
| `GET` | `/api/reactor_plot/<n>` | Full time-series + stochastic_data for reactor inspector |
| `GET` | `/api/pipe_plot/<n>` | All four pressure plots (sweep, pump curve, regime map, Fanning) |
| `GET` | `/api/pump_power_plot` | Time-series pump power chart across all pipes |
| `POST` | `/api/seek_steady_state` | Iteratively extend t_end until CoV < tol |
| `POST` | `/api/auto_balance` | Iterative flow-balance (up to 8 passes) |
| `GET` | `/api/sink_flow` | Molar flow vs time at each sink node |

### Analysis & Optimization
| Method | Route | Description |
|---|---|---|
| `GET` | `/api/plots` | All analysis plots (base64 PNG dict) |
| `DELETE` | `/api/plots/window/<id>` | Release figure cache when inspector closes |
| `POST` | `/api/optimize` | Run NSGA-II or SLSQP optimization |
| `GET` | `/api/validate` | Real-time plant validation (pump feasibility, unset defaults) |
| `POST` | `/api/balance` | Mass balance for a subset of node names |
| `GET` | `/api/balance_detail` | Per-node inlet/outlet flows and pressure |
| `POST` | `/api/demand_series` | Full time-series for all nodes (dashboard use) |
| `POST` | `/api/propose_fix` | Generate fix proposals for run-dry reactors |
| `GET/POST` | `/api/species_properties` | Per-species physical properties (ρ, MW, μ) |

### Project I/O
| Method | Route | Description |
|---|---|---|
| `POST` | `/api/build_example` | Build a pre-wired example plant (`cascade` or `parallel`) |
| `GET` | `/api/project` | Serialize full plant state as JSON |
| `POST` | `/api/project/name` | Update plant name |
| `GET` | `/api/readme` | Return README.md content |

### Response Convention
All routes return `_ok(data)` or `_err(message, code)`:
```python
{"status": "ok",    "data": {...}}
{"status": "error", "message": "..."}
```

---

## 7. Frontend Architecture

### Global State
```javascript
PLANT = {
  nodes: [],      // {id, type:'source'|'cstr'|'sink', label, x, y, config:{}}
  edges: [],      // {id, src, tgt, config:{length, diameter, elevation_change,
                  //   n_fittings_K, flow_rate_m3s, pump_efficiency}}
  result: null,   // full /api/simulate response
  simMode: 'deterministic'  // v3: 'deterministic' | 'stochastic'
}

UI = {
  selected, dragging, drawingEdge, marquee,
  inspectorNodeId, inspectorEdgeId,
  tlFrame, tlPlaying, tlAnim, tlData,  // timeline state
  rTMode,                              // 'constant'|'ramp'|'step'
  optDVars, optObjectives,             // optimizer UI state
  stochReactorLabel                    // v3: reactor open in PDF tab
}

DASH_SLOTS = [{type, target}, ...]    // 6 dashboard slot configs
```

### Key JS Functions
| Function | Purpose |
|---|---|
| `switchMainTab(name)` | Switch between `'cc'`, `'dash'`, `'opt'` tabs |
| `runSimulation()` | Sync plant → simulate (deterministic or SSA) → update UI |
| `setSimMode(mode)` | Toggle `'deterministic'` / `'stochastic'`; show/hide param panels |
| `autoBalance()` | `POST /api/auto_balance` → sync flows back to UI |
| `seekSteadyState()` | `POST /api/seek_steady_state` → update `t_end` field on convergence |
| `loadExample(ex)` | Build example → auto-simulate → render |
| `renderWorkspace()` | Redraw SVG canvas (nodes, edges, badges) |
| `openInspector(id)` | Open CSTR reactor modal or source/sink config modal |
| `openPipeInspector(eid)` | Open pipe modal, load 4 plots via `/api/pipe_plot/<n>` |
| `runPipePlots()` | Fetch and display all 4 pipe plots as PNG images |
| `refreshDashboard()` | Refresh all 6 dashboard slots |
| `refreshDashSlot(i)` | Refresh one dashboard slot based on its type |
| `drawSinkFlowCanvas(...)` | Draw molar flow vs time for a sink (canvas) |
| `drawRConcCanvas()` | Render deterministic concentration timeline on canvas |
| `drawRConcCanvasStoch(data)` | v3: render mean ± std + spaghetti bands on canvas |
| `renderStochasticTab()` | v3: render Plotly 3-D PDF surface for selected species |
| `drawConcTable(frameIdx)` | Show numeric concentration badges at timeline position |
| `addReactionToReactor()` | Add reaction with stoichiometry parsing (`2A + B → C`) |
| `collectReactionsFromUI(cfg)` | Read inline-editable reaction fields back to config |
| `applyFix(proposal)` | Apply a fix proposal and re-simulate |
| `initDashboard()` | Create dashboard slot HTML on load |
| `saveProject()` / `uploadProject(e)` | JSON project file I/O |

### Frontend Data Flow
```
User action
  → PLANT.nodes / PLANT.edges mutated locally
  → runSimulation()
      → POST /api/sync_plant   (rebuild backend from UI state)
      → POST /api/simulate     (mode + stochastic_params forwarded)
      → PLANT.result = r.data
      → renderWorkspace() + renderContextPanel() + refreshDashboard()
      → SSA badge visible iff PLANT.simMode === 'stochastic'
```

### Node Rendering
| Type | Shape | Colour |
|---|---|---|
| Source | Teal triangle | `#18d8b0` |
| CSTR | Amber circle | `#f0a030` |
| Sink | Green diamond | `#40c878` |
| Run-dry | Red border + `DRY` badge | `#e84040` |
| Overflow | Blue border + `OVERFLOW` badge | `#4a90f5` |

### Reactor Inspector Modal Tabs
`Info` · `Feeds` · `Species` · `Reactions` · `Outlet` · `Connections` · **`PDF Density`** (v3 — hidden unless `PLANT.simMode === 'stochastic'` or stochastic data present)

### Dashboard Slot Types
`concentration` · `material_balance` · `sink_flow` · `conversion_heatmap` · `pressure_sweep` · `pump_power`

---

## 8. Full Data Path

### 8.1 Deterministic Simulation
```
Frontend (PLANT state)
  │
  ├─ POST /api/sync_plant
  │     PlantNetwork rebuilt from node/edge JSON:
  │       • Source/Sink nodes → SourceSink objects
  │       • CSTR nodes → CSTR(ReactionNetwork, TemperatureGradient, FeedStreams)
  │       • Edges → Connection(Pipeline, PumpModel)
  │
  ├─ POST /api/simulate {mode:"deterministic", t_end, n_segments}
  │     PlantNetwork.simulate()
  │       1. _topological_order()  (Kahn's algorithm)
  │       2. _solve_pressures()
  │            → NetworkPressureSolver (TransportSim)
  │            → Per-pipe: Darcy-Weisbach + Colebrook-White + pump ΔP
  │            → Returns Dict[name, ConnectionDiagnostic]
  │       3. For each node in topological order:
  │            _route_outlet_to_feeds()  ← inject upstream outlet into feeds
  │            node.simulate(t_end, n_segments, mode="deterministic")
  │              CSTR._simulate_deterministic():
  │                For each segment:
  │                  T_mid = temperature_gradient.temperature_at(t_mid)
  │                  Arrhenius correction on k for all reactions
  │                  Simulator(ReactionNetwork).run()
  │                    → Try: _chemsim_core.run_simulation() [C++ BDF]
  │                    → Fallback: scipy.integrate.solve_ivp BDF
  │                  Dilution correction: C_out = C_batch·α + C_in·(1−α)
  │              SourceSink.simulate():
  │                Returns constant composition at all time points
  │       4. _compute_material_balance()
  │       5. _check_stability()  (run-dry, overflow detection)
  │       6. Aggregate pump energy from pressure_balance.total_power_kW
  │     → NetworkSimulationResult
  │
  └─ Response: result.to_dict()
       reactor_results[name].to_dict():
         time[], concentrations[][], species_names[],
         outlet_composition{}, residence_time_s, conversion{},
         temperature_profile[], converged, issues[],
         simulation_mode, stochastic_data (None in deterministic mode)
       connection_diags[name].to_dict():
         pressure_drop_kPa, velocity_m_s, reynolds_number,
         flow_regime, outlet_pressure_bar, needs_compressor, compressor_kW
       global_issues[], material_balance{},
       total_pump_kW, run_dry_reactors[], overflow_reactors[],
       source_pressures{}
```

### 8.2 Stochastic Simulation (v3 — Gillespie SSA)
```
POST /api/simulate {mode:"stochastic", n_trajectories, omega, tau_eps, n_samples}
  │
  PlantNetwork.simulate(mode="stochastic", stochastic_params={...})
    │
    CSTR.simulate(mode="stochastic", stochastic_params)
      → CSTR.simulate_stochastic()
          1. inlet = _mixed_inlet()                ← same feed-averaging as deterministic
          2. species_names = reaction_network species ∪ inlet keys  ← B1 fix
          3. reactions_raw = [{reactants, products, stoichs, k, Ea, A}, ...]
          4. D = Q_Ls / V_L                        ← dilution rate [s⁻¹]
          5. GillespieSimulator(species, reactions, C_initial=inlet,
                               C_inlet=inlet, D, T_fn, Ω)
          6. GillespieSimulator.run(t_end, n_trajectories, n_samples, tau_eps)
               For each trajectory:
                 _run_one(sample_times, rng, n_segments)
                   For each temperature segment:
                     _build_rxns(T_K)     ← Arrhenius + stochastic c_j
                     _stoich_matrix()     ← chemical + inflow + outflow channels
                     Direct/τ-leap loop:
                       Compute propensities a[]
                       τ-leap if τ·a₀ ≥ 10: fire Poisson(a_j·τ) events
                       Exact SSA otherwise:  one event, exponential dt
               → ensemble statistics: mean, std, p5, p95 per (time, species)
               → pdf_surfaces[species]: density[t_idx][bin_idx]
               → sample_trajectories[:5]
          7. CSTRResult(
               concentrations = ensemble mean,  ← backward-compatible 2D canvas
               simulation_mode = "stochastic",
               stochastic_data = GillespieResult.to_dict()
             )
  │
  Frontend receives stochastic_data:
    • drawRConcCanvasStoch()  ← 2D canvas: mean + std band + spaghetti
    • renderStochasticTab()   ← Plotly surface3d: P(C,t) per species
```

### 8.3 Plot Generation
```
GET /api/reactor_plot/<n>
  → plot_reactor_trajectories(result, [n])
      Deterministic: matplotlib line plot of concentrations[][]
      Stochastic:    fill_between for p5/p95 + std, mean line, spaghetti
  → Returns: trajectory (base64 PNG), time[], conc[][], conc_table[],
             species[], temp[], simulation_mode, stochastic_data

GET /api/pipe_plot/<n>
  → conn.flow_sweep(q_min, q_max, n_points=60)
  → conn.diagnose(src_P, tgt_P)
  → plot_pressure_sweep()     ← ΔP vs Q coloured by regime
  → plot_pump_operating_curve() ← ΔP + power with operating point
  → plot_flow_regime_map()    ← Re vs Q
  → plot_fanning_vs_darcy()   ← friction factor comparison
  → plot_pressure_breakdown() ← bar chart at operating point
```

### 8.4 Optimization
```
POST /api/optimize {method, mode, pop_size, n_gen, sim_t_end, decision_variables, objectives}
  │
  NetworkOptimizer(plant, dvars, objectives, sim_t_end, sim_n_segments)
  │
  ├─ NSGA-II (if pymoo installed, n_obj ≥ 2):
  │    pymoo NetworkProblem._evaluate(X):
  │      For each x in population:
  │        plant_copy = deepcopy(plant)
  │        _apply_x(plant_copy, x)    ← mutate volumes/flows/temps/diameters
  │        plant_copy.simulate()      ← always deterministic (no SSA in optimizer)
  │        obj_vals = [fn(result) for fn in objectives]
  │        sign-flip for maximize objectives
  │    Pareto front → pareto_X, pareto_F
  │    Best single solution by weighted-sum on Pareto front
  │
  ├─ Weighted-sum sweep (fallback, no pymoo):
  │    20 random weight vectors on simplex
  │    Each: L-BFGS-B minimize of weighted scalar objective
  │
  └─ SLSQP single-objective:
       3 random restarts, scipy.optimize.minimize with bounds
  │
  Objective functions available:
    yield_objective(reactor, species)   → outlet molar fraction C_sp / Σ C_i
    conversion_objective(reactor, species) → X_i = (C_in−C_out)/C_in
    residence_time_objective(reactor)   → τ = V/Q [s]
    total_compressor_power_objective()  → Σ compressor_kW (getattr-guarded)
    selectivity_objective(reactor, product, byproduct)
```

### 8.5 ChemSim Package — Internal Data Flow
```
ReactionNetwork
  ├─ _species: OrderedDict[name → Species(name, initial, unit)]
  ├─ _species_order: List[str]
  ├─ reactions: List[Reaction]
  │     Reaction: reactants[], products[], reactant_stoich[], product_stoich[],
  │               rate, activation_energy, pre_exponential, label
  ├─ species_names (property) → List[str]
  ├─ initial_conditions_array() → np.ndarray shape (M,)
  ├─ reactions_as_core_dicts() → List[dict] for C++ core
  └─ temperature_spec: dict

Simulator(network).run(**params) → SimulationResult
  Try:
    _chemsim_core.run_simulation(species_names, reactions, initial_conditions,
                                 temperature_spec, solver_params, progress_callback)
    → raw dict: {time, concentrations, species_names, n_steps,
                 n_rhs_evals, n_jac_evals, converged, message, conservation_laws}
  Fallback (_run_scipy):
    scipy.integrate.solve_ivp(method='BDF', ...)
    → same dict structure

SimulationResult
  ├─ time: np.ndarray (N,)
  ├─ concentrations: np.ndarray (N, M)
  ├─ species(name) → np.ndarray
  ├─ final_state() → Dict[str, float]
  ├─ at_time(t) → Dict[str, float]  (linear interpolation)
  ├─ steady_state(window, tol) → Optional[Dict]
  ├─ save(path, fmt='hdf5'|'csv')
  └─ plot() / phase_portrait()
```

### 8.6 TransportSim Package — Internal Data Flow
```
PipelineSpec(length, diameter, roughness, elevation_change, n_fittings_K)
FluidProperties(density, viscosity, [density_user_set])
Pipeline(name, source, target, spec, fluid, flow_rate_m3s)

Pipeline._compute_state() → PipelineState
  Try:
    _transportsim_core.compute_pipeline_state()  [C++ Darcy-Weisbach]
  Fallback:
    Python: Darcy-Weisbach + Colebrook-White (Newton iterations)
  → PipelineState:
      pressure_drop_Pa, velocity_m_s, reynolds_number,
      friction_factor_darcy, friction_factor_fanning,
      flow_regime, outlet_pressure_Pa,
      needs_compressor, compressor_power_kW

PumpModel(efficiency)
  .solve(spec, Q, src_P, tgt_P, fluid) → PumpState
      ΔP_pump = ΔP_friction + (P_tgt − P_src)
      W_shaft = Q · ΔP_pump / η
  → PumpState: required_delta_p_kPa, shaft_power_kW, feasible

NetworkPressureSolver(nodes, connections)
  .solve() → PressureBalanceSolution
      Iterative: for each pipe compute ΔP_pump at current node pressures
      → total_power_kW, pump_states{}, node_pressures{}

Connection(pipeline, pump_efficiency)
  .diagnose(src_P, tgt_P) → ConnectionDiagnostic
  .flow_sweep(q_min, q_max, n_points) → List[dict]
```

---

## 9. Working Features in v3.0

### Core Simulation — Deterministic
- [x] CSTR dynamic simulation via ChemSim (C++ BDF or scipy BDF fallback)
- [x] Multi-species, multi-reaction networks with arbitrary stoichiometry coefficients
- [x] Arrhenius kinetics (pre-exponential + activation energy, or rate-only correction)
- [x] Temperature gradients: isothermal, step, ramp, custom piecewise linear
- [x] Sequential cascade simulation in topological order (Kahn's algorithm)
- [x] Piecewise segment simulation with dilution correction per segment
- [x] Segment count capped at 30 to prevent ODE step exhaustion

### Core Simulation — Stochastic (v3.0)
- [x] Gillespie SSA (Gillespie 1977 Direct Method) with Cao et al. 2006 τ-leaping
- [x] CSTR open boundaries as inflow/outflow pseudo-reactions at dilution rate D = Q/V
- [x] Correct stochastic rate constants: `c_j = k_j / (Ω^(ord−1) × ∏νᵢ!)`
- [x] Falling-factorial propensities for arbitrary integer stoichiometry
- [x] Arrhenius kinetics applied piecewise per temperature segment
- [x] Feed-only inlet species correctly included (B1 fix)
- [x] Ensemble statistics: mean, std, p5/p95 per (time, species)
- [x] Per-species probability density surface `P(C, t)` for Plotly 3-D visualisation
- [x] Robust Gillespie import: tries `__file__` directory, parent, then `/mnt/project` (B2 fix)

### Network Topology
- [x] Source, CSTR, Sink nodes — all first-class (same solver treatment)
- [x] SourceSink: infinite volume, constant composition, fixed or atmospheric pressure
- [x] Arbitrary connection topology (feed routing through pipes)
- [x] Feed de-duplication: Source→CSTR pipe replaces explicit feeds; no double-counting
- [x] Topological sort for correct simulation order (Kahn's algorithm)

### Pressure & Hydraulics (TransportSim)
- [x] Per-pipe virtual pump, auto-balance on every simulation
- [x] `NetworkPressureSolver` iterative pump ΔP solver
- [x] Darcy-Weisbach with Colebrook-White friction (C++ core or Python fallback)
- [x] Fanning friction factor exposed alongside Darcy
- [x] Flow regime classification (laminar/transitional/turbulent) with diagnostics
- [x] Per-pipe efficiency (default 0.75, user-editable)
- [x] Node operating pressure (kPa, user-settable per node)
- [x] 4-panel pipe inspector: system curve, pump curve, regime map, Fanning vs Darcy
- [x] All pressures in kPa in user-facing output

### Stability Detection
- [x] Run-dry detection (downstream demand > 1.15× inlet)
- [x] Overflow detection (inlet > 1.10× outlet) — warning only, no halt
- [x] Overflow time estimate (`vol_L / excess_flow_Ls`)
- [x] Sink demand validation (zero-flow, min concentration specs)
- [x] Visual badges: red `DRY`, blue `OVERFLOW` on canvas nodes

### Steady-State Seeking
- [x] `/api/seek_steady_state` — iterative t_end doubling until CoV < tol
- [x] "Seek Steady State" button with configurable tolerance (%)
- [x] Reports convergence time, updates t_end in UI on success

### Fix Proposals & Auto-Balance
- [x] `/api/propose_fix` — throttle outlets, boost feeds, increase volume, raise temperature
- [x] Fix application re-simulates immediately
- [x] `/api/auto_balance` — up to 8 iterative passes, adjusts outlet pipe flows to 90% of inlet

### Dashboard
- [x] Three-tab layout: Control Center / Dashboard / Optimizer
- [x] 6 configurable dashboard slots with type dropdown + target selector
- [x] Auto-refresh after every simulation
- [x] Slot types: concentration trajectory, material balance, sink flow, conversion heatmap, pressure sweep, pump power

### Concentration Display
- [x] Timeline slider in reactor inspector
- [x] Numeric concentration table under graph (updates live on slider drag)
- [x] Stochastic mode: mean ± std bands + p5/p95 envelope on 2-D canvas (v3)
- [x] Stochastic mode: `PDF Density` tab with Plotly `surface3d` per species (v3)
- [x] Species selector dropdown in PDF tab; spaghetti trace toggle (v3)

### Sink Material Flow
- [x] `/api/sink_flow` — computes `Q_pipe × C_out(t)` for each species at each sink
- [x] Canvas chart in Dashboard slot (species coloured by index)
- [x] Units: mol/s

### Species Physical Properties
- [x] Per-species density (kg/m³), molar mass (g/mol), viscosity (mPa·s)
- [x] Mixture-weighted density propagated to pipeline fluid after update
- [x] Unset density/roughness flagged with ⚠ in pipe inspector and validation

### Optimization (Optimizer Tab)
- [x] Embedded as iframe at `/optimizer`
- [x] NSGA-II Pareto front (pymoo if available, weighted-sum sweep fallback)
- [x] SLSQP single-objective with 3 random restarts
- [x] Decision variables: reactor volume, feed flow, temperature, pipe flow, pipe diameter
- [x] Objectives: molar fraction, conversion, residence time, pump power
- [x] `total_compressor_power_objective` guarded with `getattr` (B5 fix)
- [x] `yield_objective` correctly documented as outlet molar fraction (B6 fix)
- [x] Pareto front canvas chart, convergence history, best solution table
- [x] Plant state passed via sessionStorage on tab switch

### Mass Balance
- [x] Source/sink nodes included in balance
- [x] Shift+drag marquee selection of nodes
- [x] Molar flow closure percentage per species

### Logging
- [x] Python `logging` with stdout (INFO) + file (DEBUG) handlers
- [x] Log file: `dashboard/networklite.log`
- [x] Key events logged: simulate, sync_plant, auto_balance, stochastic params, errors

### Graph/Plot Cleanup
- [x] `DELETE /api/plots/window/<id>` clears plot cache when inspector closes
- [x] Fresh plots on every window open (no stale cached PNGs)
- [x] `plot_reactor_trajectories` emits `trajectory_{rname}` keys; no double-prefix in `generate_all_plots` (B4 fix)
- [x] Stochastic trajectories in dashboard plots: mean ± std bands + p5/p95 + spaghetti (B4 fix)

### Project I/O
- [x] Save/load JSON project files (full PLANT state)
- [x] Cascade and Parallel example plants (with Source + Sink nodes)

---

## 10. Known Limitations / Open Work

- The ChemSim C++ extension (`_chemsim_core`) is not compiled in the deployment environment; the scipy BDF fallback is used. This is ~5–10× slower for large networks.
- The `_run_scipy` fallback does not use an analytical Jacobian — may be slower on very stiff systems.
- Gillespie SSA is always run in deterministic mode inside the optimizer (no SSA during NSGA-II/SLSQP — cost is prohibitive). The optimizer always uses the ODE path.
- Auto-balance does not yet check sink minimum-concentration constraints while adjusting flows.
- Optimizer is not yet validated against live plant constraints (run-dry, overflow) — it can propose infeasible solutions.
- Reaction stoichiometry with non-integer exponents (e.g. 1.5-order kinetics) is supported in the data model but not exposed via the add-reaction form parser.
- Species physical properties (density, viscosity) do not yet feed back into the CSTR volume/concentration calculation — only pipeline fluid density is updated.
- Stochastic τ-leaping uses integer stoichiometry (fractional stoichs rounded). Non-integer order kinetics in SSA mode fall back to nearest integer stoich.

---

## 11. Development Notes

### Adding a New API Route
1. Add `@app.route(...)` function to `dashboard/app.py`
2. Use `logger.info(...)` / `logger.error(...)` for observability
3. Return `_ok(data)` or `_err(message, code)` consistently
4. If the route needs plant state, call `_get_plant()` and check `_last_result`

### Adding a New Node Type
1. Create a class in `network/reactor/` with `.simulate()`, `.to_dict()`, `.pressure_Pa`, `.pressure_mode`, `.total_flow_m3s`
2. Register in `PlantNetwork._nodes` dict
3. Add `isinstance` guards in `plant.py` stability checks and mass balance
4. Add node type rendering in `nodeShape()` in `index.html`

### Adding a New Objective Function
1. Define `def fn(result: NetworkSimulationResult) -> float` accessing `result.reactor_results`
2. Wrap in `Objective(name, direction, fn)` and return from a factory function
3. Guard any `ConnectionDiagnostic` attribute access with `getattr(d, "attr", default)`
4. Register in `app.py` optimize route's objective-parsing block

### File Header Rule (Critical)
```python
from __future__ import annotations   # MUST be line 1 — before everything

import sys as _sys
...
```

### Frontend Data Flow
```
User action → PLANT.nodes/edges mutated locally
  → runSimulation() → sync_plant (POST) rebuilds backend
  → simulate (POST) → returns result
  → PLANT.result = r.data
  → renderWorkspace() + renderContextPanel() + refreshDashboard()
  → if stochastic: SSA badge shown, drawRConcCanvasStoch(), PDF tab revealed
```