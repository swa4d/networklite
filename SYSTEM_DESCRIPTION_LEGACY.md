# NetworkLite v2.2 — Developer System Prompt

You are a senior software engineer working on **NetworkLite**, a browser-based chemical plant network simulator. The application combines **ChemSim** (reaction kinetics ODE solver) and **TransportSim** (pipeline hydraulics, C++ + Python) with a Flask REST API and a single-page HTML/JS dashboard.

---

## 1. Architecture Overview

### Entry Point
```
python run.py [--host 0.0.0.0] [--port 5050] [--debug]
```
Flask serves `dashboard/index.html` at `/` and all `/api/*` routes.

### Backend: Flask (`dashboard/app.py`)
- Single-process, in-memory plant state (`_plant`, `_last_result`, `_last_opt` globals)
- Structured logging to stdout (INFO) and `dashboard/networklite.log` (DEBUG) via Python `logging`
- CORS enabled via `flask_cors`
- All plant state is rebuilt from the frontend's canonical `PLANT` object on every `/api/sync_plant` call before simulation

### Package structure (all imports relative to `networklite/`):
```
networklite/
├── run.py                            ← entry point
├── chemsim/                          ← chemical simulation
│   ├── network.py                    ← ReactionNetwork, Species, Reaction dataclasses
│   ├── simulator.py                  ← Simulator class + _run_scipy() fallback
│   ├── analysis.py, plotting.py, writer.py, loader.py, renderer.py
│   └── __init__.py
├── network/
│   ├── plant.py                      ← PlantNetwork (orchestration, topological sort, auto-balance)
│   ├── reactor/
│   │   ├── cstr.py                   ← CSTR, FeedStream, TemperatureGradient, CSTRResult
│   │   └── source_sink.py            ← SourceSink (infinite-volume source/sink node)
│   ├── pipeline/
│   │   └── connection.py             ← Connection (wraps Pipeline + PumpModel)
│   ├── optimizer/
│   │   └── multi_objective.py        ← NetworkOptimizer, NSGA-II / SLSQP
│   └── analysis/
│       └── diagnostics.py            ← matplotlib plot generators
├── transportsim/
│   ├── core/
│   │   ├── transportsim.hpp          ← C++ hydraulics (Darcy-Weisbach, pump solver)
│   │   ├── bindings.cpp              ← pybind11 bindings → _transportsim_core
│   │   └── setup.py                  ← build script
│   ├── pipeline.py                   ← Pipeline, PipelineSpec, FluidProperties
│   ├── pump.py                       ← PumpModel, PumpState, PumpTimeSeries
│   ├── pressure_solver.py            ← NetworkPressureSolver (auto-balance)
│   ├── sweep.py                      ← all pressure/pump matplotlib plot functions
│   ├── flow_regimes.py               ← regime classification helpers
│   └── __init__.py                   ← public re-exports
└── dashboard/
    ├── app.py                        ← Flask application (~1860 lines)
    ├── index.html                    ← single-page frontend (~3260 lines)
    └── optimizer.html                ← standalone optimizer page (served in iframe)
```

### Critical import rules
Every file that uses `from __future__ import annotations` **must** have it as the **absolute first line** of the file (before path-injection blocks, before docstrings, before everything). This has caused SyntaxErrors in prior development — always check this when editing file headers.

---

## 2. CSTR Equations

### 2.1 Residence Time
```
τ = V / Q_total    [seconds]
V = reactor volume (L),  Q_total = sum of all feed flow rates (L/s)
```
`Q_total` is computed from **only** the currently-active feed streams in `self._feeds`. When a `SourceSink` source routes into a CSTR, `_route_outlet_to_feeds` clears any existing explicit feeds from that source direction first (to prevent double-counting) and injects `_from_<SourceName>` at the pipe's flow rate.

### 2.2 Mixed Inlet Composition
```
C_in(species) = Σ_feeds [ C_feed_i(species) × (Q_feed_i / Q_total) ]
```
Flow-weighted average across all active feed streams.

### 2.3 Reaction Rate Law (power-law)
```
r_j = k_j(T) · ∏_i [C_i]^ν_ij
```
- `k_j` = rate constant (1/s for first-order)
- `ν_ij` = stoichiometric coefficient of species i in reaction j
- Concentrations clipped to zero before exponentiation

### 2.4 Arrhenius Temperature Dependence
Two forms, applied in both the ChemSim C++ core and `_run_scipy` fallback:
```
# Form 1: explicit pre-exponential (A > 0, Ea > 0)
k(T) = A · exp(-Ea / (R · T))

# Form 2: correction from reference rate at 298.15 K (A = 0, Ea > 0)
k(T) = k_ref · exp(-Ea/R · (1/T - 1/298.15))
```

### 2.5 CSTR Dynamic ODE (species balance)
The ODE solved by ChemSim (C++ BDF or scipy BDF fallback):
```
dC_i/dt = Σ_j [ ν_ij · r_j(C, T) ]
```
This is the **batch reactor ODE** (no dilution term). The CSTR dilution is applied as a post-processing correction after each time segment:

### 2.6 CSTR Dilution Correction (per segment)
After solving the batch ODE over one time segment `Δt = t_end / n_segments`:
```
C_out(t+Δt) = C_batch(Δt) · exp(-Δt/τ)  +  C_in · (1 - exp(-Δt/τ))
```
- `α = exp(-Δt/τ)` is the mixing fraction
- As `τ → ∞` (no flow), `α → 1` and `C_out = C_batch` (pure batch)
- As `τ → 0` (very fast flow), `α → 0` and `C_out = C_in` (pass-through)
- Typical CSTR behaviour is intermediate

### 2.7 Temperature Profiles (piecewise, applied per segment)
```
isothermal:  T(t) = T₀
step:        T(t) = T₀  for t < t_step,  T_final  for t ≥ t_step
ramp:        T(t) = T₀ + (T_final - T₀) · (t / t_ramp)
custom:      T(t) = piecewise linear from (times[], temps[]) arrays
```
Each of the `n_segments` uses the temperature at `t_mid = t_offset + Δt/2`.

### 2.8 Conversion
```
X_i = max(0, (C_in_i - C_out_i) / C_in_i)   for each species in inlet
```

### 2.9 Steady-State Convergence Criterion
Used by `/api/seek_steady_state`:
```
CoV_i = std(C_i[last 20%]) / mean(C_i[last 20%])
Converged when CoV_i < tol for ALL species in ALL CSTRs
```
Default `tol = 0.01` (1%). Iterates by doubling `t_end` up to `t_max = 32 × t_start`.

---

## 3. TransportSim Hydraulics Equations

### 3.1 Darcy-Weisbach (friction pressure drop)
```
ΔP_friction = f_D · (L/D) · (ρv²/2)    [Pa]
v = Q / A,   A = π(D/2)²,   Re = ρvD/μ
```

### 3.2 Colebrook-White (Darcy friction factor, iterative Newton)
```
1/√f = -2 log₁₀( ε/(3.7D) + 2.51/(Re·√f) )
```
Initial guess: Swamee-Jain explicit approximation. Laminar fallback: `f = 64/Re` for `Re < 2300`.

### 3.3 Fanning Friction Factor
```
f_Fanning = f_Darcy / 4
```

### 3.4 Total Pressure Drop
```
ΔP_total = ΔP_friction + ΔP_minor + ΔP_gravity
ΔP_minor   = K_total · (ρv²/2)        [K = sum of fitting K-values]
ΔP_gravity = ρ · g · Δz
```

### 3.5 Auto-Balance Pump Solver (per pipe)
```
ΔP_pump = ΔP_total + (P_outlet_node - P_inlet_node)
W_shaft = Q · ΔP_pump / η              [W]
```
- All nodes (CSTR + SourceSink) operate at **constant pressure** (default P_atm = 101325 Pa)
- Every pipe has a virtual pump at its inlet that provides exactly `ΔP_pump`
- `η` = per-pipe pump efficiency (default 0.75, user-editable)
- Flagged infeasible if `ΔP_pump > 5000 kPa`

### 3.6 Flow Regime Classification
| Re | Regime |
|---|---|
| < 2300 | Laminar (Hagen-Poiseuille) |
| 2300–4000 | Transitional (friction factor ±40%) |
| ≥ 4000 | Turbulent (Colebrook-White reliable) |

---

## 4. Node Types

### CSTR (`network/reactor/cstr.py`)
- Volume, temperature gradient, feeds, reaction network
- `pressure_Pa` attribute (default 101325 Pa), `pressure_mode` = `"atm"` | `"fixed"`
- Simulates via ChemSim (C++ core) with scipy BDF fallback
- `to_dict()` returns full serialisation including `node_type: "cstr"`

### SourceSink (`network/reactor/source_sink.py`)
- `node_type` = `"source"` or `"sink"`
- `volume_L = float("inf")` — never runs dry
- Fixed `species` dict (for sources: inlet compositions; for sinks: product spec minimums)
- `simulate()` returns constant composition at all time points (no ODE)
- Has `pressure_Pa`, `pressure_mode`, `flow_rate_m3s`
- Sources and sinks are fully included in mass balance, pressure sweep, and topology

---

## 5. PlantNetwork (`network/plant.py`)

Key behaviours:
- **Topological sort** (Kahn's algorithm) determines simulation order
- **Feed routing**: `_route_outlet_to_feeds` injects `_from_X` feeds. When source is a SourceSink, all existing explicit feeds at the target CSTR are cleared first to prevent double-counting.
- **Pressure solving**: `NetworkPressureSolver` runs on every simulation, solving pump ΔP for each pipe
- **Stability checks**: run-dry (downstream demand > 1.15× inlet), overflow (inlet > 1.10× downstream), sink demand validation
- **Overflow field**: `overflow_reactors` list in `NetworkSimulationResult` — separate from `run_dry_reactors`; shown as blue OVERFLOW badge in UI, does NOT halt simulation

### NetworkSimulationResult fields
```python
reactor_results:    Dict[str, CSTRResult | SourceSinkResult]
connection_diags:   Dict[str, ConnectionDiagnostic]
pressure_balance:   PressureBalanceSolution
global_issues:      List[str]
material_balance:   Dict[str, dict]   # species → {in_mol_s, out_mol_s, closure_pct}
run_dry_reactors:   List[str]         # inlet < demand × 0.87
overflow_reactors:  List[str]         # inlet > outlet × 1.10 (warning only)
stability_errors:   Dict[str, str]
total_pump_kW:      float
```

---

## 6. ChemSim Integration

ChemSim is an external package providing:
- `ReactionNetwork` — species, reactions, temperature spec
- `Simulator` — runs ODE solver

### C++ core fallback
When `chemsim._chemsim_core` is not installed (compiled C++ extension absent), `simulator.py` automatically uses `_run_scipy()` — a pure Python scipy BDF solver producing identical output format. **No user action required** — fallback is transparent.

### Stoichiometry
Reactions store `reactant_stoich` and `product_stoich` as float lists. The frontend `addReactionToReactor()` parses `"2A + B"` prefix notation and stores these arrays. `sync_plant` passes them through to `rn.add_reaction(stoich=...)`.

---

## 7. Flask API Reference

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/` | Serve `index.html` |
| GET | `/optimizer` | Serve `optimizer.html` (embedded as iframe in Optimizer tab) |
| GET | `/api/status` | Health check, version, module availability |
| GET | `/api/topology` | Node/edge graph for frontend rendering |
| GET/POST | `/api/nodes` `/api/reactors` | List all nodes / add CSTR |
| GET/DELETE | `/api/nodes/<n>` | Inspect or delete a node |
| POST | `/api/nodes/<n>/pressure` | Set node operating pressure (kPa) |
| POST | `/api/reactors/<n>/feed` | Add/update feed stream |
| POST | `/api/reactors/<n>/species` | Add species |
| POST | `/api/reactors/<n>/reaction` | Add reaction |
| POST | `/api/reactors/<n>/temperature` | Set temperature gradient |
| POST | `/api/reactors/<n>/volume` | Set reactor volume |
| GET/POST | `/api/connections` | List / add pipeline connection |
| DELETE | `/api/connections/<n>` | Delete connection |
| POST | `/api/connections/<n>/flow` | Update flow rate |
| POST | `/api/connections/<n>/pump` | Update pump efficiency |
| GET | `/api/connections/<n>/sweep` | Pump curve sweep data |
| POST | `/api/simulate` | Run simulation (n_segments capped at 30) |
| POST | `/api/auto_balance` | Iterative flow-rate balancing (up to 8 passes) |
| POST | `/api/seek_steady_state` | Extend simulation until CoV converges |
| GET | `/api/sink_flow` | Molar flow (mol/s per species) arriving at each sink over time |
| GET | `/api/plots` | All matplotlib plots (base64 PNG) |
| DELETE | `/api/plots/window/<id>` | Release plot cache for closed window |
| GET | `/api/reactor_plot/<n>` | Trajectory plot + full time/conc arrays + `conc_table` |
| GET | `/api/pipe_plot/<n>` | System curve, pump curve, regime map, Fanning/Darcy plots |
| GET | `/api/pump_power_plot` | Pump power vs time chart |
| POST | `/api/optimize` | Run NSGA-II or SLSQP optimization |
| POST | `/api/build_example` | Build cascade or parallel example plant |
| POST | `/api/sync_plant` | Rebuild backend from frontend PLANT state |
| GET | `/api/validate` | Pump feasibility + unset defaults check |
| POST | `/api/balance` | Mass balance for selected nodes |
| GET | `/api/balance_detail` | Per-connection inlet/outlet molar flows |
| POST | `/api/propose_fix` | Generate fix proposals for run-dry/instability |
| GET/POST | `/api/species_properties` | Per-species density/MW/viscosity |
| GET | `/api/project` | Full plant serialisation |
| POST | `/api/project/name` | Update plant name |

---

## 8. Frontend Layout (`dashboard/index.html`)

### Three-tab layout (single window, no navigation away)
```
HEADER  [logo | status | energy draw | run-dry badge | Cascade/Parallel | Save/Upload | Help | Run Simulation | Optimizer]

TAB BAR  [Control Center | Dashboard | Optimizer]

SCREEN-CC (Control Center):
  LEFT PANEL (240px):
    - Component palette (drag or click + to add nodes: Source / CSTR / Sink)
    - Plant tree (hierarchical list of nodes + connections)
    - Simulation params (t_end, segments, plant name, Auto-Balance, Seek Steady State + tol%)
    - Species Properties panel (density, MW, viscosity per species)
    - Mass Balance panel (appears after shift-drag selection)
  CENTER WORKSPACE:
    - SVG canvas (nodes + edges, drag/drop, port-click to connect)
    - Validation bar
  RIGHT INSPECTOR PANEL (256px):
    - Context-sensitive: shows selected node or edge properties + results

SCREEN-DASH (Dashboard):
  2×3 grid of 6 configurable slots
  Each slot: type selector dropdown + optional target (reactor/pipe name) + refresh button
  Slot types: Concentration trajectory | Material balance | Sink material flow |
              Conversion heatmap | Pressure sweep | Pump power
  Auto-refreshes after every simulation

SCREEN-OPT (Optimizer):
  <iframe src="/optimizer"> embedding optimizer.html
  State passed via sessionStorage on tab switch
```

### Node rendering
- Source: teal triangle, label + "SOURCE"
- CSTR: amber circle, label + "CSTR"
- Sink: green diamond, label + "SINK"
- Run-dry: red "DRY" badge above node, red border
- Overflow: blue "OVERFLOW" badge above node, blue border

### Key JS globals
```javascript
PLANT = { nodes: [], edges: [], result: null }
// nodes: {id, type:'source'|'cstr'|'sink', label, x, y, config:{}}
// edges: {id, src, tgt, config:{length, diameter, elevation_change,
//          n_fittings_K, flow_rate_m3s, pump_efficiency}}
// result: full /api/simulate response

UI = { selected, dragging, drawingEdge, inspectorNodeId, inspectorEdgeId,
       tlFrame, tlPlaying, tlData, rTMode, optDVars, optObjectives }

DASH_SLOTS = [{type, target}, ...]  // 6 dashboard slot configs
```

### Key JS functions
| Function | Purpose |
|----------|---------|
| `switchMainTab(name)` | Switch between 'cc', 'dash', 'opt' tabs |
| `runSimulation()` | Sync plant → simulate → update UI |
| `autoBalance()` | Sync → `/api/auto_balance` → sync flows back |
| `seekSteadyState()` | Sync → `/api/seek_steady_state` → update t_end |
| `loadExample(ex)` | Build example → auto-simulate → render |
| `renderWorkspace()` | Redraw SVG canvas |
| `openInspector(id)` | Open CSTR reactor modal or node config modal |
| `openPipeInspector(eid)` | Open pipe modal, load 4 plots via `/api/pipe_plot/<n>` |
| `runPipePlots()` | Fetch and display all 4 pipe plots as PNG images |
| `refreshDashboard()` | Refresh all 6 dashboard slots |
| `refreshDashSlot(i)` | Refresh one dashboard slot based on its type |
| `drawSinkFlowCanvas(...)` | Draw molar flow vs time for a sink (canvas) |
| `drawConcTable(frameIdx)` | Show numeric concentration badges at timeline position |
| `addReactionToReactor()` | Add reaction with stoichiometry parsing (`2A + B`) |
| `collectReactionsFromUI(cfg)` | Read inline-editable reaction fields back to config |
| `applyFix(proposal)` | Apply a fix proposal and re-simulate |
| `initDashboard()` | Create dashboard slot HTML |

---

## 9. Working Features in v2.2

### Core simulation
- [x] CSTR dynamic simulation via ChemSim (C++ core or scipy BDF fallback)
- [x] Multi-species, multi-reaction networks with arbitrary stoichiometry coefficients
- [x] Arrhenius kinetics (pre-exponential + activation energy or rate-only correction)
- [x] Temperature gradients: isothermal, step, ramp, custom piecewise linear
- [x] Sequential cascade simulation (topological order)
- [x] Piecewise segment simulation with dilution correction per segment
- [x] Segment count capped at 30 to prevent ODE step exhaustion

### Network topology
- [x] Source, CSTR, Sink nodes — all first-class (same solver treatment)
- [x] SourceSink: infinite volume, constant composition, fixed or atmospheric pressure
- [x] Arbitrary connection topology (feed routing through pipes)
- [x] Proper feed de-duplication: source→CSTR pipe replaces explicit feeds, no double-counting
- [x] Topological sort for correct simulation order (Kahn's algorithm)

### Pressure & hydraulics (TransportSim)
- [x] Per-pipe virtual pump, auto-balance on every simulation
- [x] `NetworkPressureSolver` iterative pump ΔP solver
- [x] Darcy-Weisbach with Colebrook-White friction (C++ core or Python fallback)
- [x] Fanning friction factor exposed alongside Darcy
- [x] Flow regime classification (laminar/transitional/turbulent) with diagnostics
- [x] Per-pipe efficiency (default 0.75, user-editable)
- [x] Node operating pressure (kPa, user-settable per node)
- [x] 4-panel pipe inspector: system curve, pump curve, regime map, Fanning vs Darcy
- [x] All pressures in kPa in user-facing output

### Stability detection
- [x] Run-dry detection (downstream demand > 1.15× inlet)
- [x] Overflow detection (inlet > 1.10× outlet) — warning only, no halt
- [x] Overflow time estimate (`vol_L / excess_flow_Ls`)
- [x] Sink demand validation (zero-flow, min concentration specs)
- [x] Visual badges: red "DRY", blue "OVERFLOW" on canvas nodes

### Steady-state seeking
- [x] `/api/seek_steady_state` — iterative t_end doubling until CoV < tol
- [x] "Seek Steady State" button with configurable tolerance (%)
- [x] Reports convergence time, updates t_end in UI on success

### Fix proposals & auto-balance
- [x] `/api/propose_fix` — correct proposals: throttle outlets, boost feeds, increase volume, raise temperature
- [x] Fix application re-simulates immediately
- [x] `/api/auto_balance` — up to 8 iterative passes, adjusts outlet pipe flows to 90% of inlet

### Dashboard
- [x] Three-tab layout: Control Center / Dashboard / Optimizer
- [x] 6 configurable dashboard slots with type dropdown + target selector
- [x] Auto-refresh after every simulation
- [x] Slot types: concentration trajectory, material balance, sink flow, conversion heatmap, pressure sweep, pump power

### Concentration display
- [x] Timeline slider in reactor inspector
- [x] Numeric concentration table under graph (updates live on slider drag)
- [x] Full time/concentration arrays returned by `/api/reactor_plot/<n>`
- [x] `conc_table` array: list of `{species: value}` dicts at each time step

### Sink material flow
- [x] `/api/sink_flow` — computes `Q_pipe × C_out(t)` for each species at each sink
- [x] Canvas chart rendered in Dashboard slot (species coloured by index)
- [x] Units: mol/s

### Species physical properties
- [x] Per-species density (kg/m³), molar mass (g/mol), viscosity (mPa·s)
- [x] Mixture-weighted density propagated to pipeline fluid after update
- [x] Unset density/roughness flagged with ⚠ in pipe inspector and validation

### Optimization (Optimizer tab)
- [x] Embedded as iframe at `/optimizer` (no new window)
- [x] NSGA-II Pareto front (pymoo if available, weighted-sum fallback)
- [x] SLSQP single-objective
- [x] Decision variables: reactor volume, feed flow, temperature, pipe flow, pipe diameter
- [x] Objectives: yield, conversion, residence time, pump power
- [x] Pareto front canvas chart, convergence history, best solution table
- [x] Plant state passed via sessionStorage on tab switch

### Mass balance
- [x] Source/sink nodes included in balance (previously excluded)
- [x] Shift+drag marquee selection of nodes
- [x] Molar flow closure percentage per species

### Logging
- [x] Python `logging` with stdout (INFO) + file (DEBUG) handlers
- [x] Log file: `dashboard/networklite.log`
- [x] Key events logged: simulate, sync_plant, auto_balance, errors

### Graph/plot cleanup
- [x] `DELETE /api/plots/window/<id>` clears plot cache when inspector closes
- [x] Fresh plots on every window open (no stale cached PNGs)

### Project I/O
- [x] Save/load JSON project files (full PLANT state)
- [x] Cascade and Parallel example plants (with Source + Sink nodes)

---

## 10. Known Limitations / Open Work

- The ChemSim C++ extension (`_chemsim_core`) is not compiled in the deployment environment; the scipy BDF fallback is used. This is ~5–10× slower for large networks.
- The `_run_scipy` fallback does not use an analytical Jacobian — may be slower on very stiff systems.
- Auto-balance does not yet check sink minimum-concentration constraints while adjusting flows.
- Optimizer is not yet validated against live plant constraints (run-dry, overflow) — it can propose infeasible solutions.
- Reaction stoichiometry with non-integer exponents (e.g. 1.5-order kinetics) is supported in the data model but not exposed via the add-reaction form parser.
- Species physical properties (density, viscosity) do not yet feed back into the CSTR volume/concentration calculation — only pipeline fluid density is updated.

---

## 11. Development Notes

### Adding a new API route
1. Add `@app.route(...)` function to `dashboard/app.py`
2. Use `logger.info(...)` / `logger.error(...)` for observability
3. Return `_ok(data)` or `_err(message, code)` consistently
4. If the route needs plant state, call `_get_plant()` and check `_last_result`

### Adding a new node type
1. Create a class in `network/reactor/` with `.simulate()`, `.to_dict()`, `.pressure_Pa`, `.pressure_mode`, `.total_flow_m3s`
2. Register in `PlantNetwork._nodes` dict (same as CSTR/SourceSink)
3. Add `isinstance` guards in `plant.py` stability checks and mass balance
4. Add node type rendering in `nodeShape()` in `index.html`

### Frontend data flow
```
User action → PLANT.nodes/edges mutated locally
  → runSimulation() → sync_plant (POST) rebuilds backend
  → simulate (POST) → returns result
  → PLANT.result = r.data
  → renderWorkspace() + renderContextPanel() + refreshDashboard()
```

### File header rule (critical)
Every Python file using `from __future__ import annotations` must have it as line 1:
```python
from __future__ import annotations

import sys as _sys  # path injection blocks go here
...
```
Placing anything before the `__future__` import causes a `SyntaxError` at import time.
