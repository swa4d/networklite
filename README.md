# NetworkLite v2

A chemical plant network simulator combining **ChemSim** (reaction kinetics) and **TransportSim** (pipeline hydraulics) with a browser-based dashboard.

---

## What's New in v2

### Architecture
- **Package reorganisation** — flat root files promoted into proper packages: `transportsim/`, `chemsim/`, `network/`, `dashboard/`
- **SourceSink nodes** — sources and sinks are now first-class `SourceSink` objects identical to CSTRs from the solver's perspective (infinite volume, fixed or solver-determined pressure). They participate fully in mass balance, pressure tools, and the network inspector.

### TransportSim Engine
- **Auto-balance pump solver** — every pipe has a virtual pump at its inlet. `NetworkPressureSolver` automatically solves for the pump ΔP required to achieve the user-specified flow rate between two constant-pressure nodes using `solve_pump_delta_p()` in C++.
- **Per-pipe pump efficiency** — pump efficiency is configured per connection (default 0.75), not globally.
- **Fanning friction factor** — `fanning_friction_factor()` exposed alongside Darcy-Weisbach.
- **Flow regime classifier** — `classify_regime()` returns engineer-facing diagnostics for laminar / transitional / turbulent flow.
- **Pump curve plots** — system resistance curve, pump operating curve (ΔP + power), regime map, and Fanning vs. Darcy comparison are all served as high-quality matplotlib figures.
- **Power over time** — `PumpTimeSeries` integrates pump power demand over the simulation span.
- **kPa throughout** — all user-facing pressure values are in kPa.

### Bug Fixes
- **Concentration time-stepping** — the timeline slider in the reactor inspector now shows numeric concentration values for every species at the selected time step.
- **Mass balance for sources/sinks** — the mass balance tool now correctly includes source and sink nodes.
- **Graph cleanup** — closing an inspector window sends `DELETE /api/plots/window/<id>` to the server, clearing cached plot data so reopening generates a fresh plot rather than serving a stale one.
- **Pressure sweep / tools for sources/sinks** — pressure tools now work on all node types.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install chemsim
pip install git+https://github.com/swa4d/chemsim

# Build C++ extension (optional but recommended — Python fallback available)
cd transportsim/core
python setup.py build_ext --inplace
cp _transportsim_core*.so ../../  # or .pyd on Windows
cd ../..

# Start the server
python run.py

# Open in browser
open http://localhost:5050
```

---

## Package Structure

```
networklite/
├── run.py                      ← entry point
├── transportsim/               ← hydraulics engine
│   ├── core/
│   │   ├── transportsim.hpp    ← C++ core (Darcy-Weisbach, pump solver)
│   │   ├── bindings.cpp        ← pybind11 bindings
│   │   └── setup.py            ← C++ build script
│   ├── pipeline.py             ← Pipeline, PipelineSpec, FluidProperties
│   ├── pump.py                 ← PumpModel, PumpState, PumpTimeSeries
│   ├── pressure_solver.py      ← NetworkPressureSolver (auto-balance)
│   ├── sweep.py                ← all pressure/pump plot functions
│   ├── flow_regimes.py         ← regime classification helpers
│   └── __init__.py
├── chemsim/                    ← chemical simulation engine
│   ├── network.py              ← ReactionNetwork
│   ├── simulator.py            ← Simulator + SimulationResult
│   └── ...
├── network/                    ← plant-level orchestration
│   ├── plant.py                ← PlantNetwork (unified nodes + pressure solving)
│   ├── reactor/
│   │   ├── cstr.py             ← CSTR with pressure_Pa attribute
│   │   └── source_sink.py      ← SourceSink (infinite volume, pressure-aware)
│   ├── pipeline/
│   │   └── connection.py       ← Connection with PumpModel
│   ├── optimizer/
│   │   └── multi_objective.py
│   └── analysis/
│       └── diagnostics.py
└── dashboard/
    ├── app.py                  ← Flask API (v2 routes)
    └── index.html              ← Browser UI
```

---

## Key API Endpoints (v2)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/simulate` | Run simulation — includes auto-balance pressure solve |
| POST | `/api/sync_plant` | Rebuild backend plant from UI state |
| POST | `/api/nodes/<n>/pressure` | Set node operating pressure (kPa) |
| POST | `/api/connections/<n>/pump` | Update per-pipe pump efficiency |
| GET  | `/api/connections/<n>/sweep` | Pump curve sweep data |
| GET  | `/api/pipe_plot/<n>` | All 4 pressure plots (PNG) |
| GET  | `/api/reactor_plot/<n>` | Reactor trajectory + numeric concentration table |
| GET  | `/api/pump_power_plot` | Power vs time chart |
| DELETE | `/api/plots/window/<id>` | Clear cached plots (called on window close) |
| POST | `/api/balance` | Mass balance (now includes sources/sinks) |
| GET  | `/api/validate` | Validation including pump feasibility + unset defaults |

---

## Pressure Model

```
Every node (CSTR, Source, Sink) has a constant operating pressure P_node (default P_atm).
Every pipe has a virtual pump at its inlet.

Pump ΔP = ΔP_friction + ΔP_minor + ΔP_gravity + (P_outlet_node - P_inlet_node)
Pump power = Q · ΔP / η    (η = per-pipe efficiency, default 0.75)

The solver runs automatically on every /api/simulate call.
```

---

## Assumptions & Defaults

| Parameter | Default | Note |
|-----------|---------|------|
| Fluid density | 1000 kg/m³ (water) | Yellow ⚠ badge in UI when unset |
| Pipe roughness | 4.6×10⁻⁵ m (commercial steel) | Yellow ⚠ badge in UI when unset |
| Pump efficiency | 0.75 | Per-pipe, editable |
| Node pressure | 101.325 kPa (1 atm) | Yellow ⚠ badge; set explicitly for accurate pump sizing |
| Friction factor | Colebrook-White (iterative) | Darcy-Weisbach formulation |
