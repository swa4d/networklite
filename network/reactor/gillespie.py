"""
gillespie.py – Gillespie Stochastic Simulation Algorithm (SSA) for open CSTRs.

Implements the Gillespie Direct Method (Gillespie 1977) extended to open-boundary
CSTRs via inflow/outflow pseudo-reactions, with Cao et al. (2006) τ-leaping
for tractable performance on macro-scale systems.

Mathematical foundation
-----------------------
System-size parameter Ω (omega) maps macroscopic concentrations [mol/L] to
integer pseudo-molecule counts — controlling the noise-to-speed trade-off:

    N_i  = round(C_i × Ω)          (concentration → molecule count)
    C_i  = N_i / Ω                 (molecule count → concentration)

Stochastic rate constants (Gillespie 1977, eq. 21):

    c_j = k_j / ( Ω^(ord_j − 1) × ∏_i ν_ij! )

where ord_j = Σ_i ν_ij (reaction order) and ν_ij are integer reactant stoichs.

Propensity function (state-dependent combinatorial factor):

    a_j = c_j × ∏_i  N_i! / (N_i − ν_ij)!     [falling factorial]

CSTR open-boundary pseudo-reactions (one inflow + one outflow per species):

    ∅  → X_i :  a_in_i  = D × C_in_i × Ω      [D = Q/V, dilution rate s⁻¹]
    X_i → ∅  :  a_out_i = D × N_i

τ-leaping (Cao et al. 2006)
---------------------------
When E[events in τ] ≥ 10, fire Poisson(a_j × τ) reactions simultaneously
rather than one at a time. τ is chosen so no propensity changes by more than
ε × a_0 in expectation:

    τ = min_i  min( ε max(N_i,1) / |μ_i|,  ε² max(N_i,1)² / σ²_i )

where μ_i = Σ_j ν_ij a_j  and  σ²_i = Σ_j ν_ij² a_j.

If τ < 10 / a_0, fall back to exact Direct Method for that step.

References
----------
Gillespie, D. T. (1977). J. Phys. Chem. 81, 2340–2361.
Gillespie, D. T. (2007). Annu. Rev. Phys. Chem. 58, 35–55.
Cao, Y., Gillespie, D. T., Petzold, L. R. (2006). J. Chem. Phys. 124, 044109.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

R_GAS = 8.314  # J / (mol·K)

# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GillespieResult:
    """
    Ensemble statistics from N_traj independent SSA trajectories for one CSTR.

    Attributes
    ----------
    time : np.ndarray  shape (T,)
    mean : np.ndarray  shape (T, M)   ensemble mean concentration per species
    std  : np.ndarray  shape (T, M)
    p5   : np.ndarray  shape (T, M)   5th-percentile
    p95  : np.ndarray  shape (T, M)   95th-percentile
    species_names : list[str]
    pdf_surfaces : dict
        {species: {"time": [...], "conc_bins": [...], "density": [...]}}
        density[t_idx][bin_idx] = probability density at that time/concentration.
        Ready for Plotly surface3d.
    sample_trajectories : list[list[list[float]]]
        Up to 5 raw trajectories (for spaghetti plot), shape (traj, T, M).
    n_trajectories : int
    omega : int
    converged : bool
    issues : list[str]
    """
    time                : np.ndarray
    mean                : np.ndarray
    std                 : np.ndarray
    p5                  : np.ndarray
    p95                 : np.ndarray
    species_names       : List[str]
    pdf_surfaces        : Dict[str, dict]
    sample_trajectories : List[list]
    n_trajectories      : int
    omega               : int
    converged           : bool = True
    issues              : List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "time"               : self.time.tolist(),
            "mean"               : self.mean.tolist(),
            "std"                : self.std.tolist(),
            "p5"                 : self.p5.tolist(),
            "p95"                : self.p95.tolist(),
            "species_names"      : self.species_names,
            "pdf_surfaces"       : self.pdf_surfaces,
            "sample_trajectories": self.sample_trajectories,
            "n_trajectories"     : self.n_trajectories,
            "omega"              : self.omega,
            "converged"          : self.converged,
            "issues"             : self.issues,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal SSA reaction
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _SSARxn:
    """Pre-compiled SSA reaction for fast inner-loop execution."""
    r_idx   : List[int]    # reactant species indices into N array
    p_idx   : List[int]    # product species indices
    r_nu    : List[int]    # integer reactant stoichs
    p_nu    : List[int]    # integer product stoichs
    c       : float        # stochastic rate constant


def _falling_factorial(n: int, k: int) -> int:
    """n × (n-1) × ... × (n-k+1).  Returns 0 if n < k."""
    if n < k:
        return 0
    result = 1
    for i in range(k):
        result *= (n - i)
    return result


def _propensity(rxn: _SSARxn, N: np.ndarray) -> float:
    h = 1.0
    for idx, nu in zip(rxn.r_idx, rxn.r_nu):
        h *= _falling_factorial(int(N[idx]), nu)
        if h == 0:
            return 0.0
    return rxn.c * h


# ─────────────────────────────────────────────────────────────────────────────
# GillespieSimulator
# ─────────────────────────────────────────────────────────────────────────────

class GillespieSimulator:
    """
    Stochastic simulation of an open CSTR via the Gillespie SSA.

    Parameters
    ----------
    species_names  : list[str]
    reactions_raw  : list[dict]
        Each dict: {reactants, products, reactant_stoich, product_stoich,
                    rate, activation_energy, pre_exponential}
    initial_conc   : dict  {species: concentration_M}
    inlet_conc     : dict  {species: concentration_M}
    dilution_rate  : float   D = Q_Ls / V_L  [s⁻¹]
    temperature_fn : callable   t → T_K
    omega          : int   system-size parameter (default 300)
    """

    def __init__(
        self,
        species_names  : List[str],
        reactions_raw  : List[dict],
        initial_conc   : Dict[str, float],
        inlet_conc     : Dict[str, float],
        dilution_rate  : float,
        temperature_fn : Optional[Callable[[float], float]] = None,
        omega          : int = 300,
    ) -> None:
        self.species_names = list(species_names)
        self.reactions_raw = list(reactions_raw)
        self.initial_conc  = dict(initial_conc)
        self.inlet_conc    = dict(inlet_conc)
        self.D             = float(dilution_rate)   # s⁻¹
        self.temperature_fn = temperature_fn or (lambda t: 298.15)
        self.omega         = int(omega)
        self.M             = len(species_names)
        self._idx          = {sp: i for i, sp in enumerate(species_names)}

    # ── Arrhenius rate at temperature T ──────────────────────────────────────

    def _k_at_T(self, rxn: dict, T: float) -> float:
        k_ref = float(rxn.get("rate", 0.0))
        Ea    = float(rxn.get("activation_energy", 0.0))
        A     = float(rxn.get("pre_exponential", 0.0))
        if Ea > 0:
            if A > 0:
                return A * math.exp(-Ea / (R_GAS * T))
            return k_ref * math.exp(-Ea / R_GAS * (1.0 / T - 1.0 / 298.15))
        return k_ref

    # ── Build SSA reaction list at temperature T ──────────────────────────────

    def _build_rxns(self, T: float) -> List[_SSARxn]:
        """Convert deterministic reactions to SSA format at temperature T."""
        Ω    = self.omega
        rxns : List[_SSARxn] = []
        for raw in self.reactions_raw:
            rnames = list(raw.get("reactants", []))
            pnames = list(raw.get("products",  []))
            r_nus  = [int(round(s)) for s in raw.get("reactant_stoich", [1]*len(rnames))]
            p_nus  = [int(round(s)) for s in raw.get("product_stoich",  [1]*len(pnames))]

            r_idx = [self._idx[n] for n in rnames if n in self._idx]
            p_idx = [self._idx[n] for n in pnames if n in self._idx]
            if len(r_idx) != len(rnames):
                continue  # species unknown — skip

            order = sum(r_nus)
            nu_fact = 1
            for nu in r_nus:
                nu_fact *= math.factorial(nu)

            k_T = self._k_at_T(raw, T)
            c   = k_T / (Ω ** max(0, order - 1) * nu_fact)

            rxns.append(_SSARxn(r_idx=r_idx, p_idx=p_idx, r_nu=r_nus, p_nu=p_nus, c=c))
        return rxns

    # ── Stoichiometry matrix for τ-leaping ────────────────────────────────────

    def _stoich_matrix(self, rxns: List[_SSARxn]) -> np.ndarray:
        """
        Build net-change matrix S shape (n_total, M).

        Row layout:
          [0 .. n_chem-1]      chemical reactions (net change)
          [n_chem .. +M-1]     inflow  (+1 to one species each)
          [n_chem+M .. +2M-1]  outflow (-1 to one species each)
        """
        n_chem  = len(rxns)
        n_total = n_chem + 2 * self.M
        S = np.zeros((n_total, self.M), dtype=np.int64)
        for j, rxn in enumerate(rxns):
            for idx, nu in zip(rxn.r_idx, rxn.r_nu):
                S[j, idx] -= nu
            for idx, nu in zip(rxn.p_idx, rxn.p_nu):
                S[j, idx] += nu
        for i in range(self.M):
            S[n_chem + i,          i] =  1  # inflow
            S[n_chem + self.M + i, i] = -1  # outflow
        return S

    # ── Single trajectory ─────────────────────────────────────────────────────

    def _run_one(
        self,
        sample_times : np.ndarray,
        rng          : np.random.Generator,
        n_segments   : int = 10,
        tau_eps      : float = 0.03,
        exact_thresh : float = 10.0,
        max_steps    : int = 300_000,
    ) -> np.ndarray:
        """
        Run one SSA trajectory; return concentrations at sample_times.

        Returns
        -------
        conc_sampled : np.ndarray  shape (T, M)
        """
        Ω   = self.omega
        D   = self.D
        M   = self.M
        T_s = len(sample_times)

        t_end = float(sample_times[-1])

        # Initial molecule counts
        N = np.array(
            [max(0, int(round(self.initial_conc.get(sp, 0.0) * Ω)))
             for sp in self.species_names],
            dtype=np.int64,
        )
        C_in  = np.array([self.inlet_conc.get(sp, 0.0) for sp in self.species_names])
        a_in_base = D * C_in * Ω      # shape (M,) — inflow propensities (constant)

        conc_sampled = np.zeros((T_s, M))
        s_idx        = 0

        # Split into temperature segments
        seg_edges = np.linspace(0.0, t_end, n_segments + 1)

        for seg in range(n_segments):
            t_seg_start = seg_edges[seg]
            t_seg_end   = seg_edges[seg + 1]
            T_K         = self.temperature_fn(0.5 * (t_seg_start + t_seg_end))

            rxns   = self._build_rxns(T_K)
            S      = self._stoich_matrix(rxns)        # (n_total, M)
            n_chem = len(rxns)
            n_tot  = n_chem + 2 * M

            # Stoich columns for drift / variance (tau selection)
            S_sq   = S ** 2

            t     = t_seg_start
            steps = 0

            while t < t_seg_end and steps < max_steps:
                # ── Build propensity vector ───────────────────────────────────
                a = np.zeros(n_tot)
                for j, rxn in enumerate(rxns):
                    a[j] = _propensity(rxn, N)
                a[n_chem:n_chem + M] = a_in_base             # inflow
                a[n_chem + M:]       = D * N.astype(float)   # outflow

                a_total = a.sum()

                if a_total <= 0.0:
                    t = t_seg_end
                    break

                # ── Record samples at times ≤ t ───────────────────────────────
                while s_idx < T_s and sample_times[s_idx] <= t:
                    conc_sampled[s_idx] = N / Ω
                    s_idx += 1

                # ── τ-leaping step size (Cao 2006) ────────────────────────────
                mu     = S.T @ a                # shape (M,) — expected drift
                sigma2 = S_sq.T @ a             # shape (M,) — variance

                tau_candidates = []
                for i in range(M):
                    denom_mu  = abs(mu[i])
                    denom_s2  = sigma2[i]
                    n_i       = max(float(N[i]), 1.0)
                    if denom_mu > 1e-30:
                        tau_candidates.append(tau_eps * n_i / denom_mu)
                    if denom_s2  > 1e-30:
                        tau_candidates.append(tau_eps * tau_eps * n_i * n_i / denom_s2)

                if tau_candidates:
                    tau_leap = min(tau_candidates)
                else:
                    tau_leap = 1.0 / a_total

                # Clamp to segment boundary
                tau_leap = min(tau_leap, t_seg_end - t)
                tau_leap = max(tau_leap, 1e-14)

                # ── Decide: exact SSA or τ-leap ───────────────────────────────
                if tau_leap * a_total < exact_thresh:
                    # Exact Direct Method (one event)
                    dt   = rng.exponential(1.0 / a_total)
                    dt   = min(dt, t_seg_end - t)

                    # Advance samples for [t, t+dt)
                    while s_idx < T_s and sample_times[s_idx] <= t + dt:
                        conc_sampled[s_idx] = N / Ω
                        s_idx += 1

                    t     += dt
                    steps += 1

                    # Select event
                    cumsum = np.cumsum(a)
                    j      = int(np.searchsorted(cumsum, rng.random() * a_total))
                    j      = min(j, n_tot - 1)
                    dN     = S[j].copy()
                    N      = np.maximum(N + dN, 0)

                else:
                    # τ-leaping: fire Poisson(a_j * τ) for each channel
                    k     = rng.poisson(np.maximum(a * tau_leap, 0.0))  # (n_tot,)
                    dN    = S.T @ k                                       # (M,)
                    N     = np.maximum(N + dN, 0)
                    t    += tau_leap
                    steps += 1

                    # Record samples advanced over τ
                    while s_idx < T_s and sample_times[s_idx] <= t:
                        conc_sampled[s_idx] = N / Ω
                        s_idx += 1

        # Fill any remaining sample slots with final state
        while s_idx < T_s:
            conc_sampled[s_idx] = N / Ω
            s_idx += 1

        return conc_sampled

    # ── Ensemble run ──────────────────────────────────────────────────────────

    def run(
        self,
        t_end          : float,
        n_trajectories : int   = 30,
        n_samples      : int   = 80,
        n_segments     : int   = 10,
        tau_eps        : float = 0.03,
        seed           : Optional[int] = None,
        n_bins         : int   = 30,
    ) -> GillespieResult:
        """
        Run n_trajectories independent SSA realisations.

        Parameters
        ----------
        t_end          : simulation end time
        n_trajectories : number of independent runs
        n_samples      : evenly-spaced output time points
        n_segments     : temperature-segment count (Arrhenius piecewise)
        tau_eps        : τ-leaping error parameter ε (0.01–0.05 typical)
        seed           : RNG seed for reproducibility
        n_bins         : concentration histogram bins for PDF surface

        Returns
        -------
        GillespieResult
        """
        sample_times = np.linspace(0.0, t_end, n_samples)
        rng          = np.random.default_rng(seed)
        issues: List[str] = []

        # Stack: (n_traj, n_samples, M)
        all_trajs = np.zeros((n_trajectories, n_samples, self.M))
        for traj_i in range(n_trajectories):
            try:
                all_trajs[traj_i] = self._run_one(
                    sample_times, rng, n_segments=n_segments, tau_eps=tau_eps
                )
            except Exception as exc:
                issues.append(f"Trajectory {traj_i}: {exc}")

        # ── Ensemble statistics ───────────────────────────────────────────────
        mean = all_trajs.mean(axis=0)                          # (T, M)
        std  = all_trajs.std(axis=0)
        p5   = np.percentile(all_trajs,  5, axis=0)
        p95  = np.percentile(all_trajs, 95, axis=0)

        # ── PDF surface (per species, for Plotly surface3d) ───────────────────
        pdf_surfaces: Dict[str, dict] = {}
        for i, sp in enumerate(self.species_names):
            vals    = all_trajs[:, :, i]   # (n_traj, n_samples)
            c_min   = float(vals.min())
            c_max   = float(vals.max())
            if c_max - c_min < 1e-12:
                c_max = c_min + 1e-10
            edges       = np.linspace(c_min, c_max, n_bins + 1)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])
            density     = np.zeros((n_samples, n_bins))
            for t_idx in range(n_samples):
                hist, _ = np.histogram(vals[:, t_idx], bins=edges, density=True)
                density[t_idx] = hist
            pdf_surfaces[sp] = {
                "time"      : sample_times.tolist(),
                "conc_bins" : bin_centers.tolist(),
                "density"   : density.tolist(),   # [t][bin]
            }

        # ── Sample trajectories (max 5 for spaghetti plot) ───────────────────
        n_samp_traj = min(5, n_trajectories)
        sample_trajs = all_trajs[:n_samp_traj].tolist()

        return GillespieResult(
            time                = sample_times,
            mean                = mean,
            std                 = std,
            p5                  = p5,
            p95                 = p95,
            species_names       = self.species_names,
            pdf_surfaces        = pdf_surfaces,
            sample_trajectories = sample_trajs,
            n_trajectories      = n_trajectories,
            omega               = self.omega,
            converged           = len(issues) == 0,
            issues              = issues,
        )