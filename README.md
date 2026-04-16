# NetworkLite

A browser-based chemical plant simulator. Build reactor networks, run kinetic simulations, and inspect hydraulics — all from a single-page dashboard.

---

## What it does

- **Draw** source → CSTR → sink networks on a drag-and-drop canvas
- **Simulate** concentration trajectories using ODE kinetics (deterministic) or the Gillespie SSA (stochastic)
- **Inspect** pressure drop, pump power, and flow regime for every pipeline
- **Optimize** reactor volumes, feed flows, and temperatures with NSGA-II or SLSQP
- **Diagnose** run-dry / overflow conditions and apply one-click fix proposals

---

## Quick start

```bash
pip install flask flask-cors numpy scipy
python run.py
```

Open `http://localhost:5050` in your browser.

To build the optional C++ cores (ChemSim BDF solver and TransportSim hydraulics):

```bash
# ChemSim
cd chemsim && pip install -e .

# TransportSim
cd transportsim/core && pip install -e .
```

Without the C++ extensions the app falls back to pure-Python solvers automatically.

---

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.10+, Flask, NumPy, SciPy |
| ODE solver | ChemSim C++ BDF (pybind11) or scipy `solve_ivp` fallback |
| SSA solver | Gillespie Direct Method + Cao τ-leaping (`gillespie.py`) |
| Hydraulics | TransportSim C++ Darcy-Weisbach (pybind11) or Python fallback |
| Optimizer | pymoo NSGA-II or SciPy SLSQP |
| Frontend | Vanilla JS, HTML5 Canvas, Plotly.js 2.32 (CDN) |

---

## Project layout

```
run.py                  entry point
gillespie.py            Gillespie SSA engine (v3)
chemsim/                reaction kinetics package
network/
  plant.py              plant orchestration
  reactor/
    cstr.py             CSTR reactor
    source_sink.py      source / sink nodes
  pipeline/
    connection.py       pipeline + pump wrapper
  optimizer/
    multi_objective.py  NSGA-II / SLSQP
  analysis/
    diagnostics.py      matplotlib plot generators
transportsim/           hydraulics package
dashboard/
  app.py                Flask API (~50 routes)
  index.html            single-page frontend
  optimizer.html        optimizer iframe
```

---

## Simulation modes

**Ideal Model** (default) — solves the CSTR species-balance ODE with a piecewise Arrhenius correction and an exponential dilution correction per segment.

**Stochastic Reality** — runs N independent Gillespie SSA trajectories. The reactor inspector shows a mean ± std band on the 2D canvas and a full probability density surface `P(concentration, time)` in the **PDF Density** tab (powered by Plotly).

Switch between modes with the toggle in the Simulation panel. Stochastic parameters (number of trajectories, system-size Ω, τ-leaping ε) are exposed when stochastic mode is active.

---

## Examples

Click **Cascade Example** or **Parallel Example** in the header to load a pre-wired plant, then hit **Run Simulation**.

---

## Developer notes

- All plant state lives in `PLANT` (JS) and is rebuilt on every `POST /api/sync_plant`
- Flask is single-process; state is in-memory globals (`_plant`, `_last_result`)
- `from __future__ import annotations` must be **line 1** in every Python file that uses it
- The full theory document (equations, API reference, data paths) is in `SYSTEM_PROMPT.md`