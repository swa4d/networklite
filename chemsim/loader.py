"""
loader.py – Load reaction networks from JSON configuration files.

Supports the ChemSim JSON schema (see Appendix A of the project scope).
Unknown keys are silently ignored so configs remain forward-compatible.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Union

from chemsim.network import ReactionNetwork

# ─── JSON Schema ──────────────────────────────────────────────────────────────

_NETWORK_SCHEMA = {
    "type": "object",
    "required": ["species", "reactions"],
    "additionalProperties": True,
    "properties": {
        "name":        {"type": "string"},
        "description": {"type": "string"},
        "species": {
            "type": "array",
            "minItems": 1,
            "items": {
                "oneOf": [
                    {"type": "string"},
                    {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name":    {"type": "string"},
                            "initial": {"type": "number", "minimum": 0},
                            "unit":    {"type": "string"},
                        },
                    },
                ]
            },
        },
        "reactions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "reactants":          {"type": "array", "items": {"type": "string"}},
                    "products":           {"type": "array", "items": {"type": "string"}},
                    "reactant_stoich":    {"type": "array", "items": {"type": "number"}},
                    "product_stoich":     {"type": "array", "items": {"type": "number"}},
                    "rate":               {"type": "number", "minimum": 0},
                    "activation_energy":  {"type": "number", "minimum": 0},
                    "pre_exponential":    {"type": "number", "minimum": 0},
                    "equation":           {"type": "string"},
                    "label":              {"type": "string"},
                    "__comment":          {"type": "string"},
                },
            },
        },
        "initial_conditions": {
            "type": "object",
            "additionalProperties": {"type": "number"},
        },
        "temperature": {},  # any
        "simulation": {
            "type": "object",
            "properties": {
                "duration":            {"type": "number", "minimum": 0},
                "t_start":             {"type": "number"},
                "t_end":               {"type": "number"},
                "timestep":            {"type": "number"},
                "output_interval":     {"type": "number"},
                "temperature":         {},
                "solver":              {"type": "string"},
                "tolerance_relative":  {"type": "number"},
                "tolerance_absolute":  {"type": "number"},
                "method":              {"type": "string"},
            },
        },
    },
}


def _parse_species_list(raw_species, initial_conditions_override=None):
    """
    Accept either:
      ["A", "B"]                         – names only, IC = 0
      [{"name": "A", "initial": 100}]    – full spec
    plus an optional override dict.
    """
    species = []
    for item in raw_species:
        if isinstance(item, str):
            species.append({"name": item, "initial": 0.0, "unit": "M"})
        elif isinstance(item, dict):
            species.append({
                "name":    item["name"],
                "initial": float(item.get("initial", 0.0)),
                "unit":    item.get("unit", "M"),
            })
        else:
            raise ValueError(f"Invalid species entry: {item!r}")

    # Apply initial_conditions override (flat dict format)
    if initial_conditions_override:
        name_to_idx = {s["name"]: i for i, s in enumerate(species)}
        for name, value in initial_conditions_override.items():
            if name in name_to_idx:
                species[name_to_idx[name]]["initial"] = float(value)

    return species


def _parse_reaction(raw_rxn: dict) -> dict:
    """
    Parse one reaction dict.

    Supports two formats:
      1. Explicit lists:  {"reactants": ["A","B"], "products": ["C"], "rate": 1.5}
      2. Equation string: {"equation": "A + B -> C", "rate": 1.5}
    """
    rxn = {}

    if "equation" in raw_rxn and "reactants" not in raw_rxn:
        # Parse equation string: "2*A + B -> C + D"
        reactants, products = _parse_equation_string(raw_rxn["equation"])
        rxn["reactants"]       = [r[1] for r in reactants]
        rxn["products"]        = [p[1] for p in products]
        rxn["reactant_stoich"] = [r[0] for r in reactants]
        rxn["product_stoich"]  = [p[0] for p in products]
    else:
        rxn["reactants"] = raw_rxn.get("reactants", [])
        rxn["products"]  = raw_rxn.get("products", [])
        if "reactant_stoich" in raw_rxn:
            rxn["reactant_stoich"] = [float(s) for s in raw_rxn["reactant_stoich"]]
        if "product_stoich" in raw_rxn:
            rxn["product_stoich"]  = [float(s) for s in raw_rxn["product_stoich"]]

    rxn["rate"]               = float(raw_rxn.get("rate", 1.0))
    rxn["activation_energy"]  = float(raw_rxn.get("activation_energy", 0.0))
    rxn["pre_exponential"]    = float(raw_rxn.get("pre_exponential", 0.0))
    rxn["label"]              = raw_rxn.get("label", raw_rxn.get("__comment", ""))
    return rxn


def _parse_equation_string(eq: str):
    """
    Parse "2*Prey + B -> C + 3*D" into
    ([(2.0, 'Prey'), (1.0, 'B')], [(1.0, 'C'), (3.0, 'D')]).
    Supports '→', '->', '->'. Sink '∅' or '0' is treated as empty list.
    """
    eq = eq.replace("→", "->").replace("=>", "->")
    if "->" not in eq:
        raise ValueError(f"Equation missing arrow: '{eq}'")
    lhs, rhs = eq.split("->", 1)

    def parse_side(side: str):
        side = side.strip()
        if side in ("∅", "0", ""):
            return []
        terms = []
        for term in side.split("+"):
            term = term.strip()
            if not term:
                continue
            if "*" in term:
                coeff_str, name = term.split("*", 1)
                coeff = float(coeff_str.strip())
                name  = name.strip()
            elif term[0].isdigit():
                # "2Prey" format
                i = 0
                while i < len(term) and (term[i].isdigit() or term[i] == "."):
                    i += 1
                coeff = float(term[:i]) if i > 0 else 1.0
                name  = term[i:].strip()
            else:
                coeff = 1.0
                name  = term
            terms.append((coeff, name))
        return terms

    return parse_side(lhs), parse_side(rhs)


def _parse_temperature(raw_temp) -> Union[float, dict]:
    """Interpret the 'temperature' field from JSON."""
    if raw_temp is None:
        return 298.15
    if isinstance(raw_temp, (int, float)):
        return float(raw_temp)
    if isinstance(raw_temp, dict):
        return raw_temp
    raise ValueError(f"Invalid temperature specification: {raw_temp!r}")


def load_json_string(json_str: str) -> tuple[ReactionNetwork, dict]:
    """
    Parse a JSON string into a (ReactionNetwork, simulation_params) tuple.

    Returns
    -------
    network : ReactionNetwork
    sim_params : dict
        Keys from the 'simulation' block (t_end, method, tolerances, etc.)
        ready to pass to ``Simulator.run(**sim_params)``.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc

    # Optional schema validation
    try:
        import jsonschema
        jsonschema.validate(data, _NETWORK_SCHEMA)
    except ImportError:
        pass  # jsonschema not installed: skip validation
    except Exception as exc:
        raise ValueError(f"JSON schema validation failed: {exc}") from exc

    # ── Species ───────────────────────────────────────────────────────────────
    ic_override = data.get("initial_conditions")
    if isinstance(ic_override, dict):
        pass  # handled in _parse_species_list
    else:
        ic_override = None

    species_list = _parse_species_list(data["species"], ic_override)

    # ── Build network ─────────────────────────────────────────────────────────
    net = ReactionNetwork(
        name=data.get("name", "Untitled"),
        description=data.get("description", ""),
    )
    for sp in species_list:
        net.add_species(sp["name"], initial=sp["initial"], unit=sp.get("unit", "M"))

    for raw_rxn in data.get("reactions", []):
        rxn = _parse_reaction(raw_rxn)
        net.add_reaction(
            reactants         = rxn["reactants"],
            products          = rxn["products"],
            rate              = rxn["rate"],
            reactant_stoich   = rxn.get("reactant_stoich"),
            product_stoich    = rxn.get("product_stoich"),
            activation_energy = rxn["activation_energy"],
            pre_exponential   = rxn["pre_exponential"],
            label             = rxn.get("label", ""),
        )

    # ── Temperature ────────────────────────────────────────────────────────────
    # Prefer simulation.temperature, then top-level temperature
    sim_block = data.get("simulation", {})
    raw_temp  = sim_block.get("temperature", data.get("temperature", 298.15))
    net.set_temperature(_parse_temperature(raw_temp))

    # ── Simulation params ──────────────────────────────────────────────────────
    sim_params: dict = {}

    t_end = sim_block.get("duration") or sim_block.get("t_end")
    if t_end is not None:
        sim_params["t_end"] = float(t_end)

    if "t_start" in sim_block:
        sim_params["t_start"] = float(sim_block["t_start"])

    if "output_interval" in sim_block:
        sim_params["output_interval"] = float(sim_block["output_interval"])
    elif "timestep" in sim_block:
        sim_params["output_interval"] = float(sim_block["timestep"])

    solver = sim_block.get("solver", "").lower()
    if solver in ("cvode_bdf", "bdf"):
        sim_params["method"] = "bdf"
    elif solver in ("cvode_adams", "adams"):
        sim_params["method"] = "adams"
    elif solver == "rk4":
        sim_params["method"] = "rk4"

    if "tolerance_relative" in sim_block:
        sim_params["rel_tol"] = float(sim_block["tolerance_relative"])
    if "tolerance_absolute" in sim_block:
        sim_params["abs_tol"] = float(sim_block["tolerance_absolute"])

    return net, sim_params


def load_json(path: Union[str, Path]) -> tuple[ReactionNetwork, dict]:
    """
    Load a reaction network from a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON configuration file.

    Returns
    -------
    (ReactionNetwork, sim_params dict)

    Examples
    --------
    >>> net, params = load_json("lotka_volterra.json")
    >>> result = Simulator(net).run(**params)
    """
    path = Path(path)
    if path.suffix.lower() not in (".json",):
        raise ValueError(f"Expected a .json file, got: {path.suffix}")
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")
    return load_json_string(text)
