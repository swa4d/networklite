"""
writer.py – HDF5 and CSV output for simulation results.

HDF5 layout matches the project spec exactly:
    /metadata/species_names
    /metadata/reaction_equations
    /metadata/initial_conditions
    /metadata/solver_params
    /metadata/timestamp
    /trajectories/time
    /trajectories/concentrations
    /trajectories/solver_stats
"""

from __future__ import annotations

import json
import time as _time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from chemsim.simulator import SimulationResult


# ─── HDF5 ─────────────────────────────────────────────────────────────────────

def save_hdf5(
    result: "SimulationResult",
    path: str,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save a SimulationResult to an HDF5 file.

    Parameters
    ----------
    result : SimulationResult
    path : str
        Output path (e.g. 'output.h5').
    compression : str
        HDF5 compression filter ('gzip', 'lzf', or None).
    compression_opts : int
        Compression level for gzip (1–9).
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "h5py is required for HDF5 output. Install with: pip install h5py"
        ) from exc

    path = str(path)
    ts   = datetime.now(tz=timezone.utc).isoformat()

    comp_kwargs = {}
    if compression:
        comp_kwargs["compression"]      = compression
        comp_kwargs["compression_opts"] = compression_opts

    with h5py.File(path, "w") as f:
        # ── /metadata ─────────────────────────────────────────────────────────
        meta = f.create_group("metadata")

        # Species names as variable-length UTF-8 strings
        dt_str = h5py.special_dtype(vlen=str)
        sp_ds = meta.create_dataset(
            "species_names",
            data=np.array(result.species_names, dtype=object),
            dtype=dt_str,
        )
        sp_ds.attrs["description"] = "Ordered list of species names"

        # Initial conditions
        ic = result.concentrations[0]  # row 0
        ic_ds = meta.create_dataset("initial_conditions", data=ic)
        ic_ds.attrs["species"] = json.dumps(result.species_names)
        ic_ds.attrs["description"] = "Initial concentrations at t=t_start"

        # Reaction equations (from network if available)
        if result._network is not None:
            eqs = [r.equation for r in result._network.reactions]
            eq_ds = meta.create_dataset(
                "reaction_equations",
                data=np.array(eqs, dtype=object),
                dtype=dt_str,
            )
            eq_ds.attrs["description"] = "Human-readable reaction equations"

        # Solver parameters
        solver_params = {
            "method":       "bdf",
            "converged":    result.converged,
            "message":      result.message,
            "wall_time_s":  result.wall_time_s,
        }
        params_ds = meta.create_dataset(
            "solver_params",
            data=json.dumps(solver_params),
        )
        params_ds.attrs["format"] = "JSON"

        # Timestamp
        meta.create_dataset("timestamp", data=ts)

        # Conservation laws
        if result.conservation_laws:
            meta.create_dataset(
                "conservation_laws",
                data=json.dumps(result.conservation_laws),
            )

        # ── /trajectories ─────────────────────────────────────────────────────
        traj = f.create_group("trajectories")

        t_ds = traj.create_dataset("time", data=result.time, **comp_kwargs)
        t_ds.attrs["description"] = "Simulation time points"
        t_ds.attrs["units"]       = "time"

        c_ds = traj.create_dataset(
            "concentrations",
            data=result.concentrations,
            **comp_kwargs,
        )
        c_ds.attrs["description"] = "Concentrations[time_index, species_index]"
        c_ds.attrs["species"]     = json.dumps(result.species_names)
        c_ds.attrs["shape"]       = f"[{len(result.time)}, {len(result.species_names)}]"

        # Solver stats
        stats = {
            "n_steps":      result.n_steps,
            "n_rhs_evals":  result.n_rhs_evals,
            "n_jac_evals":  result.n_jac_evals,
        }
        stats_ds = traj.create_dataset(
            "solver_stats",
            data=json.dumps(stats),
        )
        stats_ds.attrs["format"] = "JSON"

        # Root attributes
        f.attrs["chemsim_version"] = "0.1.0"
        f.attrs["created"]         = ts
        f.attrs["n_species"]       = len(result.species_names)
        f.attrs["n_timepoints"]    = len(result.time)


def load_hdf5(path: str) -> "SimulationResult":
    """
    Load a SimulationResult from an HDF5 file saved by save_hdf5.

    Returns
    -------
    SimulationResult
    """
    try:
        import h5py
    except ImportError as exc:
        raise ImportError("h5py required. Install: pip install h5py") from exc

    from chemsim.simulator import SimulationResult

    with h5py.File(path, "r") as f:
        meta = f["metadata"]
        traj = f["trajectories"]

        species_names = [s.decode() if isinstance(s, bytes) else s
                         for s in meta["species_names"][:]]
        time          = traj["time"][:]
        concentrations = traj["concentrations"][:]

        stats = json.loads(traj["solver_stats"][()])
        params = json.loads(meta["solver_params"][()])

        conservation_laws = []
        if "conservation_laws" in meta:
            conservation_laws = json.loads(meta["conservation_laws"][()])

    return SimulationResult(
        time              = time,
        concentrations    = concentrations,
        species_names     = species_names,
        n_steps           = stats.get("n_steps", 0),
        n_rhs_evals       = stats.get("n_rhs_evals", 0),
        n_jac_evals       = stats.get("n_jac_evals", 0),
        converged         = params.get("converged", True),
        message           = params.get("message", ""),
        conservation_laws = conservation_laws,
        wall_time_s       = params.get("wall_time_s", 0.0),
    )


# ─── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(
    result: "SimulationResult",
    path: str,
    precision: int = 10,
) -> None:
    """
    Save a SimulationResult as a CSV file.

    The CSV has columns: time, species_1, species_2, ...
    A header comment block contains solver metadata.

    Parameters
    ----------
    result : SimulationResult
    path : str
        Output path (e.g. 'output.csv').
    precision : int
        Number of significant figures.
    """
    import csv

    path = Path(path)
    ts   = datetime.now(tz=timezone.utc).isoformat()

    with path.open("w", newline="", encoding="utf-8") as csvfile:
        # Metadata comment header
        csvfile.write(f"# ChemSim output – {ts}\n")
        csvfile.write(f"# Species: {', '.join(result.species_names)}\n")
        csvfile.write(f"# n_steps={result.n_steps}  "
                      f"n_rhs_evals={result.n_rhs_evals}  "
                      f"converged={result.converged}\n")
        csvfile.write(f"# wall_time={result.wall_time_s:.4f}s\n")
        if result.conservation_laws:
            csvfile.write(f"# conservation_laws={json.dumps(result.conservation_laws)}\n")

        writer = csv.writer(csvfile)
        header = ["time"] + result.species_names
        writer.writerow(header)

        fmt = f"{{:.{precision}g}}"
        for i, t in enumerate(result.time):
            row = [fmt.format(t)] + [fmt.format(c) for c in result.concentrations[i]]
            writer.writerow(row)


def load_csv(path: str) -> "SimulationResult":
    """
    Load a SimulationResult from a CSV saved by save_csv.

    Returns
    -------
    SimulationResult
    """
    import csv
    from chemsim.simulator import SimulationResult

    path   = Path(path)
    times  = []
    concs  = []
    species_names = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header_done = False
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if not header_done:
                species_names = row[1:]
                header_done = True
                continue
            times.append(float(row[0]))
            concs.append([float(v) for v in row[1:]])

    return SimulationResult(
        time           = np.array(times),
        concentrations = np.array(concs),
        species_names  = species_names,
    )
