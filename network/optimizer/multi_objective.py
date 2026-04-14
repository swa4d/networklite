from __future__ import annotations
import sys as _sys
for _p in ["/mnt/project", "/home/claude/networklite"]:
    if _p not in _sys.path: _sys.path.insert(0, _p)
del _sys
"""
network/optimizer/multi_objective.py  –  Multi-objective network optimizer.

Supports:
  - Single-objective (SLSQP, Nelder-Mead, L-BFGS-B)
  - Multi-objective Pareto front (NSGA-II via pymoo, or weighted-sum fallback)
  - Objectives: yield, conversion, selectivity, residence time, energy

Decision variables: flow rates, reactor volumes, temperatures, and
pipeline diameters are all controllable.
"""


import copy
import sys, os
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution

_net_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _net_dir not in sys.path:
    sys.path.insert(0, _net_dir)

from network.plant import PlantNetwork

# ── pymoo for NSGA-II ─────────────────────────────────────────────────────────
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination
    _HAS_PYMOO = True
except ImportError:
    _HAS_PYMOO = False


# ── Decision variable spec ────────────────────────────────────────────────────

@dataclass
class DecisionVariable:
    """
    A single tunable parameter in the optimization.

    Parameters
    ----------
    name : str
        Human-readable name, e.g. "R1.volume_L".
    target_type : str
        "reactor_volume" | "feed_flow" | "temperature" | "pipe_diameter" | "pipe_flow"
    target_name : str
        Name of the reactor or connection.
    sub_target : str, optional
        For "feed_flow": feed name. For "temperature": gradient param.
    lower : float
        Lower bound.
    upper : float
        Upper bound.
    """
    name         : str
    target_type  : str
    target_name  : str
    sub_target   : str  = ""
    lower        : float = 0.0
    upper        : float = 1.0


# ── Objective function spec ───────────────────────────────────────────────────

@dataclass
class Objective:
    """
    A single optimization objective.

    Parameters
    ----------
    name : str
        Identifier, e.g. "maximize_yield_C".
    direction : str
        "minimize" | "maximize"
    fn : callable
        f(result: NetworkSimulationResult) -> float
    weight : float
        Weight in weighted-sum multi-objective (ignored in NSGA-II).
    """
    name      : str
    direction : str                                    # "minimize" | "maximize"
    fn        : Callable                               # result → float
    weight    : float = 1.0


# ── Pre-built objective functions ─────────────────────────────────────────────

def yield_objective(reactor_name: str, species: str, direction="maximize") -> Objective:
    """Maximize/minimize molar fraction of `species` in reactor outlet."""
    def fn(result):
        rr = result.reactor_results.get(reactor_name)
        if rr is None or not rr.outlet_composition:
            return 0.0
        total = sum(rr.outlet_composition.values()) or 1.0
        return rr.outlet_composition.get(species, 0.0) / total
    return Objective(name=f"yield_{species}@{reactor_name}", direction=direction, fn=fn)


def conversion_objective(reactor_name: str, species: str, direction="maximize") -> Objective:
    """Maximize/minimize conversion of `species` in reactor."""
    def fn(result):
        rr = result.reactor_results.get(reactor_name)
        if rr is None:
            return 0.0
        return rr.conversion.get(species, 0.0)
    return Objective(name=f"conv_{species}@{reactor_name}", direction=direction, fn=fn)


def residence_time_objective(reactor_name: str, direction="minimize") -> Objective:
    """Minimize/maximize residence time (proxy for reactor cost)."""
    def fn(result):
        rr = result.reactor_results.get(reactor_name)
        if rr is None:
            return 1e6
        return rr.residence_time_s
    return Objective(name=f"tau@{reactor_name}", direction=direction, fn=fn)


def total_compressor_power_objective(direction="minimize") -> Objective:
    """Minimize total compressor power across all pipelines."""
    def fn(result):
        total_kW = sum(
            d.compressor_kW for d in result.connection_diags.values()
        )
        return total_kW
    return Objective(name="compressor_power_kW", direction=direction, fn=fn)


def selectivity_objective(
    reactor_name: str, product: str, byproduct: str, direction="maximize"
) -> Objective:
    """Maximize selectivity = product_yield / (product_yield + byproduct_yield)."""
    def fn(result):
        rr = result.reactor_results.get(reactor_name)
        if rr is None:
            return 0.0
        c_p  = rr.outlet_composition.get(product,   0.0)
        c_bp = rr.outlet_composition.get(byproduct, 0.0)
        denom = c_p + c_bp
        return c_p / denom if denom > 0 else 0.0
    return Objective(name=f"sel_{product}/{byproduct}@{reactor_name}",
                     direction=direction, fn=fn)


# ── Optimization result ───────────────────────────────────────────────────────

@dataclass
class OptimizationResult:
    """
    Complete result from one optimization run.
    """
    method             : str
    n_objectives       : int
    objectives         : List[str]
    decision_variables : List[str]
    # Single-objective
    optimal_x          : Optional[np.ndarray]     = None
    optimal_objectives : Optional[List[float]]    = None
    # Multi-objective Pareto
    pareto_X           : Optional[np.ndarray]     = None  # (n_pts, n_vars)
    pareto_F           : Optional[np.ndarray]     = None  # (n_pts, n_obj)
    # History
    convergence_history: List[float]              = field(default_factory=list)
    n_evaluations      : int                      = 0
    wall_time_s        : float                    = 0.0
    success            : bool                     = False
    message            : str                      = ""

    def to_dict(self) -> dict:
        d = {
            "method"             : self.method,
            "n_objectives"       : self.n_objectives,
            "objectives"         : self.objectives,
            "decision_variables" : self.decision_variables,
            "n_evaluations"      : self.n_evaluations,
            "wall_time_s"        : self.wall_time_s,
            "success"            : self.success,
            "message"            : self.message,
            "convergence_history": self.convergence_history,
        }
        if self.optimal_x is not None:
            d["optimal_x"]          = self.optimal_x.tolist()
            d["optimal_objectives"] = self.optimal_objectives
        if self.pareto_X is not None:
            d["pareto_X"] = self.pareto_X.tolist()
            d["pareto_F"] = self.pareto_F.tolist()
        return d


# ── Optimizer class ───────────────────────────────────────────────────────────

class NetworkOptimizer:
    """
    Multi-objective optimizer for a PlantNetwork.

    Parameters
    ----------
    plant : PlantNetwork
        The plant to optimize (will be deep-copied internally).
    decision_variables : list of DecisionVariable
        Parameters to tune.
    objectives : list of Objective
        Objective functions to optimize.
    sim_t_end : float
        Simulation end time used in each evaluation.
    sim_n_segments : int
        Temperature segments per evaluation.
    """

    def __init__(
        self,
        plant             : PlantNetwork,
        decision_variables: List[DecisionVariable],
        objectives        : List[Objective],
        sim_t_end         : float = 300.0,
        sim_n_segments    : int   = 10,
    ):
        self.plant              = plant
        self.dvars              = decision_variables
        self.objectives         = objectives
        self.sim_t_end          = sim_t_end
        self.sim_n_segments     = sim_n_segments
        self._n_evals           = 0
        self._history           : List[float] = []

    # ── Parameter application ─────────────────────────────────────────────────

    def _apply_x(self, plant: PlantNetwork, x: np.ndarray) -> None:
        """Apply decision variable vector x to a plant copy."""
        for i, dv in enumerate(self.dvars):
            val = float(x[i])
            if dv.target_type == "reactor_volume":
                plant.get_reactor(dv.target_name).volume_L = max(val, 0.1)
            elif dv.target_type == "feed_flow":
                reactor = plant.get_reactor(dv.target_name)
                reactor.set_feed_flow(dv.sub_target, max(val, 1e-7))
            elif dv.target_type == "temperature":
                reactor = plant.get_reactor(dv.target_name)
                reactor.temperature_gradient.T_initial_K = max(val, 200.0)
                reactor.temperature_gradient.T_final_K   = max(val, 200.0)
            elif dv.target_type == "pipe_flow":
                plant.get_connection(dv.target_name).set_flow(max(val, 1e-7))
            elif dv.target_type == "pipe_diameter":
                conn = plant.get_connection(dv.target_name)
                conn.pipeline.spec.diameter = max(val, 0.005)

    def _evaluate(self, x: np.ndarray) -> Tuple[list, object]:
        """Evaluate all objectives for parameter vector x."""
        plant_copy = copy.deepcopy(self.plant)
        self._apply_x(plant_copy, x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = plant_copy.simulate(
                t_end=self.sim_t_end,
                n_segments=self.sim_n_segments,
            )
        self._n_evals += 1
        obj_vals = []
        for obj in self.objectives:
            v = obj.fn(result)
            if obj.direction == "maximize":
                v = -v   # minimization convention internally
            obj_vals.append(v)
        return obj_vals, result

    # ── Single-objective ──────────────────────────────────────────────────────

    def optimize_single(
        self,
        method  : str = "SLSQP",
        n_restarts: int = 3,
    ) -> OptimizationResult:
        """
        Single-objective optimization using weighted-sum scalarization.

        Parameters
        ----------
        method : str
            scipy.optimize method: 'SLSQP', 'Nelder-Mead', 'L-BFGS-B',
            or 'differential_evolution'.
        n_restarts : int
            Number of random restarts to escape local minima.
        """
        t0 = time.time()
        self._n_evals = 0
        self._history = []

        lb = np.array([dv.lower for dv in self.dvars])
        ub = np.array([dv.upper for dv in self.dvars])
        bounds = list(zip(lb, ub))
        weights = np.array([o.weight for o in self.objectives])
        weights /= weights.sum()

        def scalar_obj(x):
            obj_vals, _ = self._evaluate(x)
            weighted = float(np.dot(weights, obj_vals))
            self._history.append(weighted)
            return weighted

        best_x   = None
        best_val = np.inf

        for restart in range(n_restarts):
            x0 = lb + np.random.rand(len(self.dvars)) * (ub - lb)
            try:
                if method == "differential_evolution":
                    res = differential_evolution(scalar_obj, bounds,
                                                 maxiter=100, tol=1e-4,
                                                 seed=restart * 42)
                else:
                    res = minimize(scalar_obj, x0, method=method,
                                   bounds=bounds,
                                   options={"maxiter": 200, "ftol": 1e-6})
                if res.fun < best_val:
                    best_val = res.fun
                    best_x   = res.x
            except Exception:
                continue

        if best_x is None:
            best_x = lb + np.random.rand(len(self.dvars)) * (ub - lb)
            best_val = float("inf")

        obj_vals, _ = self._evaluate(best_x)
        final_objs = []
        for i, (obj, v) in enumerate(zip(self.objectives, obj_vals)):
            final_objs.append(-v if obj.direction == "maximize" else v)

        return OptimizationResult(
            method             = method,
            n_objectives       = len(self.objectives),
            objectives         = [o.name for o in self.objectives],
            decision_variables = [dv.name for dv in self.dvars],
            optimal_x          = best_x,
            optimal_objectives = final_objs,
            convergence_history= self._history,
            n_evaluations      = self._n_evals,
            wall_time_s        = time.time() - t0,
            success            = True,
            message            = f"Completed {n_restarts} restarts with {method}.",
        )

    # ── Multi-objective (NSGA-II) ─────────────────────────────────────────────

    def optimize_pareto(
        self,
        pop_size  : int = 40,
        n_gen     : int = 30,
        seed      : int = 42,
    ) -> OptimizationResult:
        """
        True multi-objective Pareto optimization via NSGA-II.

        Requires pymoo. Falls back to weighted-sum grid if unavailable.
        """
        t0 = time.time()
        self._n_evals = 0
        self._history = []

        lb = np.array([dv.lower for dv in self.dvars])
        ub = np.array([dv.upper for dv in self.dvars])
        n_obj = len(self.objectives)
        n_var = len(self.dvars)

        if _HAS_PYMOO and n_obj >= 2:
            return self._nsga2_optimize(lb, ub, n_var, n_obj, pop_size, n_gen, seed, t0)
        else:
            return self._weighted_sum_pareto(lb, ub, n_obj, n_var, t0)

    def _nsga2_optimize(
        self, lb, ub, n_var, n_obj, pop_size, n_gen, seed, t0
    ) -> OptimizationResult:
        """NSGA-II via pymoo."""

        opt = self

        class NetworkProblem(Problem):
            def __init__(self_):
                super().__init__(
                    n_var=n_var, n_obj=n_obj,
                    xl=lb, xu=ub, elementwise=False
                )
            def _evaluate(self_, X, out, *args, **kwargs):
                F = np.zeros((len(X), n_obj))
                for i, x in enumerate(X):
                    obj_vals, _ = opt._evaluate(x)
                    F[i] = obj_vals
                    opt._history.append(float(np.mean(obj_vals)))
                out["F"] = F

        algo = NSGA2(pop_size=pop_size)
        term = get_termination("n_gen", n_gen)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pymoo_minimize(NetworkProblem(), algo, term,
                                 seed=seed, verbose=False)

        pareto_X = res.X if res.X is not None else np.zeros((1, n_var))
        pareto_F_raw = res.F if res.F is not None else np.zeros((1, n_obj))

        # Flip sign back for maximize objectives
        pareto_F = pareto_F_raw.copy()
        for j, obj in enumerate(self.objectives):
            if obj.direction == "maximize":
                pareto_F[:, j] = -pareto_F[:, j]

        # Best single solution (min weighted sum on Pareto)
        weights = np.array([o.weight for o in self.objectives])
        weights /= weights.sum()
        scores = pareto_F_raw @ weights
        best_i = int(np.argmin(scores))
        best_x = pareto_X[best_i]
        best_objs = pareto_F[best_i].tolist()

        return OptimizationResult(
            method             = "NSGA-II",
            n_objectives       = n_obj,
            objectives         = [o.name for o in self.objectives],
            decision_variables = [dv.name for dv in self.dvars],
            optimal_x          = best_x,
            optimal_objectives = best_objs,
            pareto_X           = pareto_X,
            pareto_F           = pareto_F,
            convergence_history= self._history,
            n_evaluations      = self._n_evals,
            wall_time_s        = time.time() - t0,
            success            = True,
            message            = f"NSGA-II: {n_gen} generations, pop={pop_size}.",
        )

    def _weighted_sum_pareto(
        self, lb, ub, n_obj, n_var, t0
    ) -> OptimizationResult:
        """
        Weighted-sum scalarization sweep to approximate Pareto front
        when pymoo is unavailable or only 1 objective.
        """
        n_sweep = 20
        pareto_X_list: List[np.ndarray] = []
        pareto_F_list: List[list]        = []

        for i in range(n_sweep):
            # Random weight vector on the simplex
            if n_obj == 1:
                w = np.array([1.0])
            else:
                w = np.random.dirichlet(np.ones(n_obj))

            def wobj(x):
                obj_vals, _ = self._evaluate(x)
                return float(np.dot(w, obj_vals))

            x0     = lb + np.random.rand(n_var) * (ub - lb)
            bounds = list(zip(lb, ub))
            try:
                res = minimize(wobj, x0, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": 100})
                obj_vals, _ = self._evaluate(res.x)
                final_f = []
                for j, obj in enumerate(self.objectives):
                    v = obj_vals[j]
                    final_f.append(-v if obj.direction == "maximize" else v)
                pareto_X_list.append(res.x)
                pareto_F_list.append(final_f)
                self._history.append(float(np.dot(w, obj_vals)))
            except Exception:
                continue

        if not pareto_X_list:
            x_arr = np.zeros((1, n_var))
            f_arr = np.zeros((1, n_obj))
        else:
            x_arr = np.array(pareto_X_list)
            f_arr = np.array(pareto_F_list)

        best_i = 0
        best_x = x_arr[0]
        best_f = f_arr[0].tolist()

        return OptimizationResult(
            method             = "WeightedSum-Pareto",
            n_objectives       = n_obj,
            objectives         = [o.name for o in self.objectives],
            decision_variables = [dv.name for dv in self.dvars],
            optimal_x          = best_x,
            optimal_objectives = best_f,
            pareto_X           = x_arr,
            pareto_F           = f_arr,
            convergence_history= self._history,
            n_evaluations      = self._n_evals,
            wall_time_s        = time.time() - t0,
            success            = True,
            message            = f"Weighted-sum sweep ({n_sweep} points) — install pymoo for NSGA-II.",
        )

    def apply_optimal(self, result: OptimizationResult) -> None:
        """Apply the best found solution back to the plant."""
        if result.optimal_x is not None:
            self._apply_x(self.plant, result.optimal_x)
