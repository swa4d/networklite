"""
network.py – ReactionNetwork and associated domain objects.

This module is pure Python: no C++ dependency, fully testable in isolation.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ─── Domain types ──────────────────────────────────────────────────────────────


@dataclass
class Species:
    """A chemical species with a name and initial concentration."""

    name: str
    initial: float = 0.0
    unit: str = "M"

    def __post_init__(self) -> None:
        if not self.name or not re.match(r"^[A-Za-z_][A-Za-z0-9_*\-\[\]]*$", self.name):
            raise ValueError(
                f"Invalid species name '{self.name}'. "
                "Must start with a letter or underscore and contain only "
                "alphanumeric characters, underscores, hyphens, or brackets."
            )
        if self.initial < 0:
            raise ValueError(f"Initial concentration of '{self.name}' must be ≥ 0")


@dataclass
class Reaction:
    """
    A single chemical reaction.

    Rate law: r = k(T) · ∏ [reactant_i]^stoich_i

    If activation_energy > 0, the rate constant follows Arrhenius:
        k(T) = pre_exponential · exp(-activation_energy / (R · T))
    """

    reactants: List[str] = field(default_factory=list)
    products: List[str] = field(default_factory=list)
    rate: float = 1.0
    reactant_stoich: List[float] = field(default_factory=list)
    product_stoich: List[float] = field(default_factory=list)
    activation_energy: float = 0.0   # J/mol
    pre_exponential: float = 0.0     # A in Arrhenius; 0 = derive from rate
    label: str = ""

    def __post_init__(self) -> None:
        # Fill default stoichiometries
        if not self.reactant_stoich:
            self.reactant_stoich = [1.0] * len(self.reactants)
        if not self.product_stoich:
            self.product_stoich = [1.0] * len(self.products)

        if len(self.reactant_stoich) != len(self.reactants):
            raise ValueError("reactant_stoich length must match reactants")
        if len(self.product_stoich) != len(self.products):
            raise ValueError("product_stoich length must match products")
        if self.rate < 0:
            raise ValueError("Reaction rate must be ≥ 0")
        if self.activation_energy < 0:
            raise ValueError("Activation energy must be ≥ 0")

    @property
    def equation(self) -> str:
        """Human-readable equation string, e.g. '2·A + B → C'."""
        def fmt_side(names: List[str], stoichs: List[float]) -> str:
            parts = []
            for name, s in zip(names, stoichs):
                parts.append(f"{s:.4g}·{name}" if s != 1.0 else name)
            return " + ".join(parts) if parts else "∅"

        return f"{fmt_side(self.reactants, self.reactant_stoich)} → {fmt_side(self.products, self.product_stoich)}"

    @property
    def order(self) -> float:
        """Reaction order (sum of reactant stoichiometries)."""
        return sum(self.reactant_stoich)

    def to_core_dict(self) -> dict:
        """Serialize to the dict format expected by _chemsim_core.run_simulation."""
        return {
            "reactants":        self.reactants,
            "products":         self.products,
            "reactant_stoich":  self.reactant_stoich,
            "product_stoich":   self.product_stoich,
            "rate":             self.rate,
            "activation_energy": self.activation_energy,
            "pre_exponential":  self.pre_exponential,
        }


# ─── Temperature specification helpers ─────────────────────────────────────────


def constant_temperature(T: float) -> dict:
    """Return a temperature spec for a constant temperature *T* (Kelvin)."""
    if T <= 0:
        raise ValueError("Temperature must be positive (Kelvin)")
    return {"type": "constant", "T": float(T)}


def step_temperature(segments: List[Tuple[float, float, float]],
                     T_default: float = 298.15) -> dict:
    """
    Piecewise-constant temperature profile.

    Parameters
    ----------
    segments : list of (t_start, t_end, T)
        Time intervals and their temperatures.
    T_default : float
        Temperature outside all segments.
    """
    for seg in segments:
        if len(seg) != 3:
            raise ValueError("Each step segment must be (t_start, t_end, T)")
        if seg[0] >= seg[1]:
            raise ValueError(f"Segment t_start ({seg[0]}) must be < t_end ({seg[1]})")
        if seg[2] <= 0:
            raise ValueError("Temperature must be positive (Kelvin)")
    return {
        "type": "step",
        "segments": [[float(t0), float(t1), float(T)] for t0, t1, T in segments],
        "T_default": float(T_default),
    }


def ramp_temperature(segments: List[Tuple[float, float, float, float]],
                     T_default: float = 298.15) -> dict:
    """
    Piecewise-linear temperature ramp.

    Parameters
    ----------
    segments : list of (t_start, t_end, T_start, T_end)
    T_default : float
        Temperature outside all segments.
    """
    for seg in segments:
        if len(seg) != 4:
            raise ValueError("Each ramp segment must be (t_start, t_end, T_start, T_end)")
        if seg[0] >= seg[1]:
            raise ValueError("Segment t_start must be < t_end")
    return {
        "type": "ramp",
        "segments": [[float(t0), float(t1), float(T0), float(T1)]
                     for t0, t1, T0, T1 in segments],
        "T_default": float(T_default),
    }


# ─── ReactionNetwork ───────────────────────────────────────────────────────────


class ReactionNetwork:
    """
    Container for a chemical reaction network definition.

    This class is the primary user-facing API for building networks
    programmatically. It holds species, reactions, and temperature specs
    but performs no numerical computation itself.

    Examples
    --------
    >>> net = ReactionNetwork(name="Simple A→B")
    >>> net.add_species("A", initial=100.0)
    >>> net.add_species("B", initial=0.0)
    >>> net.add_reaction(["A"], ["B"], rate=0.5)
    >>> net.set_temperature(310.0)
    >>> print(net)
    ReactionNetwork 'Simple A→B' | 2 species, 1 reactions, T=310.0 K
    """

    def __init__(self, name: str = "Untitled", description: str = "") -> None:
        self.name: str = name
        self.description: str = description
        self._species: Dict[str, Species] = {}
        self._species_order: List[str] = []  # insertion order
        self._reactions: List[Reaction] = []
        self._temperature_spec: Union[float, dict] = 298.15  # default 25°C

    # ── Species ───────────────────────────────────────────────────────────────

    def add_species(
        self,
        name: str,
        initial: float = 0.0,
        unit: str = "M",
    ) -> "ReactionNetwork":
        """Add a species. Returns self for chaining."""
        if name in self._species:
            raise ValueError(f"Species '{name}' already defined")
        sp = Species(name=name, initial=float(initial), unit=unit)
        self._species[name] = sp
        self._species_order.append(name)
        return self

    def set_initial(self, name: str, value: float) -> "ReactionNetwork":
        """Update initial concentration of an existing species."""
        if name not in self._species:
            raise KeyError(f"Unknown species '{name}'")
        if value < 0:
            raise ValueError("Initial concentration must be ≥ 0")
        self._species[name].initial = float(value)
        return self

    @property
    def species_names(self) -> List[str]:
        return list(self._species_order)

    @property
    def n_species(self) -> int:
        return len(self._species_order)

    # ── Reactions ──────────────────────────────────────────────────────────────

    def add_reaction(
        self,
        reactants: Sequence[str],
        products: Sequence[str],
        rate: float,
        reactant_stoich: Optional[Sequence[float]] = None,
        product_stoich: Optional[Sequence[float]] = None,
        activation_energy: float = 0.0,
        pre_exponential: float = 0.0,
        label: str = "",
    ) -> "ReactionNetwork":
        """
        Add a reaction.

        Parameters
        ----------
        reactants : list of str
            Reactant species names (may be empty for spontaneous creation).
        products : list of str
            Product species names (may be empty for degradation).
        rate : float
            Rate constant k at reference temperature (298.15 K).
        reactant_stoich : list of float, optional
            Stoichiometric coefficients for reactants (default 1.0 each).
        product_stoich : list of float, optional
            Stoichiometric coefficients for products (default 1.0 each).
        activation_energy : float
            Activation energy Ea [J/mol]. 0 = temperature-independent.
        pre_exponential : float
            Pre-exponential factor A in Arrhenius. 0 = derive from rate.
        label : str
            Human-readable label for this reaction.
        """
        # Validate species references (auto-add if not present would be surprising)
        all_mentioned = list(reactants) + list(products)
        for sp in all_mentioned:
            if sp not in self._species:
                raise KeyError(
                    f"Species '{sp}' referenced in reaction but not defined. "
                    f"Call add_species('{sp}') first."
                )

        rxn = Reaction(
            reactants=list(reactants),
            products=list(products),
            rate=float(rate),
            reactant_stoich=list(reactant_stoich) if reactant_stoich else [],
            product_stoich=list(product_stoich)   if product_stoich  else [],
            activation_energy=float(activation_energy),
            pre_exponential=float(pre_exponential),
            label=label,
        )
        self._reactions.append(rxn)
        return self

    @property
    def reactions(self) -> List[Reaction]:
        return list(self._reactions)

    @property
    def n_reactions(self) -> int:
        return len(self._reactions)

    # ── Temperature ────────────────────────────────────────────────────────────

    def set_temperature(self, spec: Union[float, dict]) -> "ReactionNetwork":
        """
        Set the temperature specification.

        Parameters
        ----------
        spec : float or dict
            Float = constant temperature in Kelvin.
            Dict = structured temperature profile (use helper functions:
            ``constant_temperature``, ``step_temperature``, ``ramp_temperature``).
        """
        if isinstance(spec, (int, float)):
            if float(spec) <= 0:
                raise ValueError("Temperature must be positive (Kelvin)")
            self._temperature_spec = float(spec)
        elif isinstance(spec, dict):
            if "type" not in spec:
                raise ValueError("Temperature dict must have a 'type' key")
            self._temperature_spec = spec
        else:
            raise TypeError("Temperature spec must be a float or a dict")
        return self

    @property
    def temperature_spec(self) -> Union[float, dict]:
        return self._temperature_spec

    # ── Serialization ──────────────────────────────────────────────────────────

    def initial_conditions_array(self) -> np.ndarray:
        """Return initial concentrations as a float64 NumPy array (insertion order)."""
        return np.array(
            [self._species[n].initial for n in self._species_order],
            dtype=np.float64,
        )

    def reactions_as_core_dicts(self) -> List[dict]:
        """Serialize all reactions to the format expected by _chemsim_core."""
        return [r.to_core_dict() for r in self._reactions]

    def validate(self) -> None:
        """
        Validate the network for common issues.
        Raises ValueError with a descriptive message if invalid.
        """
        if self.n_species == 0:
            raise ValueError("Network has no species")
        if self.n_reactions == 0:
            raise ValueError("Network has no reactions")

        # All species referenced in reactions must exist
        for i, rxn in enumerate(self._reactions):
            for sp in rxn.reactants + rxn.products:
                if sp not in self._species:
                    raise ValueError(
                        f"Reaction {i} ('{rxn.equation}'): "
                        f"unknown species '{sp}'"
                    )

        # Check for zero-rate reactions (warn, not error)
        zero_rate = [i for i, r in enumerate(self._reactions) if r.rate == 0.0]
        if zero_rate:
            import warnings
            warnings.warn(
                f"Reactions at indices {zero_rate} have rate=0.0 and will have no effect.",
                stacklevel=2,
            )

    def copy(self) -> "ReactionNetwork":
        """Return a deep copy."""
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        """Serialize the entire network to a JSON-compatible dict."""
        return {
            "name": self.name,
            "description": self.description,
            "species": [
                {"name": n, "initial": self._species[n].initial, "unit": self._species[n].unit}
                for n in self._species_order
            ],
            "reactions": [
                {
                    "reactants": r.reactants,
                    "products": r.products,
                    "rate": r.rate,
                    "reactant_stoich": r.reactant_stoich,
                    "product_stoich": r.product_stoich,
                    "activation_energy": r.activation_energy,
                    "pre_exponential": r.pre_exponential,
                    "label": r.label,
                }
                for r in self._reactions
            ],
            "temperature": self._temperature_spec,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReactionNetwork":
        """Deserialize from a dict (as produced by to_dict or load_json)."""
        net = cls(name=d.get("name", "Untitled"), description=d.get("description", ""))
        for sp in d.get("species", []):
            net.add_species(sp["name"], initial=sp.get("initial", 0.0),
                            unit=sp.get("unit", "M"))
        for r in d.get("reactions", []):
            net.add_reaction(
                reactants=r.get("reactants", []),
                products=r.get("products", []),
                rate=r.get("rate", 1.0),
                reactant_stoich=r.get("reactant_stoich"),
                product_stoich=r.get("product_stoich"),
                activation_energy=r.get("activation_energy", 0.0),
                pre_exponential=r.get("pre_exponential", 0.0),
                label=r.get("label", ""),
            )
        temp = d.get("temperature", 298.15)
        net.set_temperature(temp)
        return net

    # ── Display ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        T_str = (
            f"{self._temperature_spec:.2f} K"
            if isinstance(self._temperature_spec, float)
            else self._temperature_spec.get("type", "?")
        )
        return (
            f"ReactionNetwork('{self.name}' | "
            f"{self.n_species} species, {self.n_reactions} reactions, T={T_str})"
        )

    def __str__(self) -> str:
        lines = [f"ReactionNetwork: '{self.name}'"]
        if self.description:
            lines.append(f"  {self.description}")
        lines.append(f"\nSpecies ({self.n_species}):")
        for n in self._species_order:
            sp = self._species[n]
            lines.append(f"  {n:20s}  [{sp.initial:.4g} {sp.unit}]")
        lines.append(f"\nReactions ({self.n_reactions}):")
        for i, r in enumerate(self._reactions):
            lbl = f"  [{r.label}]" if r.label else ""
            ea_str = f", Ea={r.activation_energy:.0f} J/mol" if r.activation_energy else ""
            lines.append(f"  {i:3d}. {r.equation:<40s}  k={r.rate:.4g}{ea_str}{lbl}")
        T_str = (
            f"{self._temperature_spec:.2f} K"
            if isinstance(self._temperature_spec, float)
            else str(self._temperature_spec)
        )
        lines.append(f"\nTemperature: {T_str}")
        return "\n".join(lines)
