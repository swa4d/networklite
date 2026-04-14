"""
transportsim/flow_regimes.py  –  Flow regime classification helpers (v2).

Thin Python wrapper over the C++ classify_regime function, plus
utility functions for generating regime diagnostic strings suitable
for display to process engineers.
"""

from __future__ import annotations

import sys
import os

_ts_dir = os.path.dirname(os.path.abspath(__file__))
if _ts_dir not in sys.path:
    sys.path.insert(0, _ts_dir)

try:
    import _transportsim_core as _core
    _HAS_CORE = True
except ImportError:
    _HAS_CORE = False


# ── Constants ─────────────────────────────────────────────────────────────────

RE_LAMINAR_MAX      = 2300.0
RE_TURBULENT_MIN    = 4000.0


# ── Regime classification ─────────────────────────────────────────────────────

def classify_regime(Re: float) -> dict:
    """
    Classify flow regime and return an engineer-facing diagnostic dict.

    Parameters
    ----------
    Re : float  Reynolds number

    Returns
    -------
    dict with keys:
        regime        : str   "laminar" | "transitional" | "turbulent"
        reynolds      : float
        is_turbulent  : bool
        is_laminar    : bool
        is_transitional : bool
        diagnostic    : str  human-readable explanation
        recommendation: str  actionable note
        color_hint    : str  "#hex" for UI colour coding
    """
    if _HAS_CORE:
        s = _core.classify_regime(Re)
        return {
            "regime"          : s.regime,
            "reynolds"        : s.reynolds_number,
            "is_turbulent"    : s.is_turbulent,
            "is_laminar"      : s.is_laminar,
            "is_transitional" : s.is_transitional,
            "diagnostic"      : s.diagnostic,
            "recommendation"  : s.recommendation,
            "color_hint"      : _regime_color(s.regime),
        }
    return _classify_python(Re)


def _classify_python(Re: float) -> dict:
    if Re < RE_LAMINAR_MAX:
        return {
            "regime"          : "laminar",
            "reynolds"        : Re,
            "is_turbulent"    : False,
            "is_laminar"      : True,
            "is_transitional" : False,
            "diagnostic"      : f"Re={Re:.0f} < 2300: Laminar (Hagen-Poiseuille). "
                                 "Darcy-Weisbach may overestimate ΔP.",
            "recommendation"  : "Increase flow rate or reduce pipe diameter for turbulent regime.",
            "color_hint"      : "#e84040",
        }
    elif Re < RE_TURBULENT_MIN:
        return {
            "regime"          : "transitional",
            "reynolds"        : Re,
            "is_turbulent"    : False,
            "is_laminar"      : False,
            "is_transitional" : True,
            "diagnostic"      : f"Re={Re:.0f} in 2300–4000: Transitional. Friction factor uncertain (±40%).",
            "recommendation"  : "Avoid this regime. Target Re > 10000 for reliable ΔP calculation.",
            "color_hint"      : "#f0a030",
        }
    else:
        note = ""
        if Re > 1e6:
            note = " Very high Re — check erosion limits."
        return {
            "regime"          : "turbulent",
            "reynolds"        : Re,
            "is_turbulent"    : True,
            "is_laminar"      : False,
            "is_transitional" : False,
            "diagnostic"      : f"Re={Re:.0f}: Turbulent. Colebrook-White friction is reliable." + note,
            "recommendation"  : note.strip(),
            "color_hint"      : "#18d8b0",
        }


def _regime_color(regime: str) -> str:
    return {"laminar": "#e84040", "transitional": "#f0a030", "turbulent": "#18d8b0"}.get(regime, "#18d8b0")


def regime_summary(hydraulic_state_dict: dict) -> str:
    """
    Return a short human-readable summary for a hydraulic state dict.

    Parameters
    ----------
    hydraulic_state_dict : dict with keys "regime", "reynolds_number",
                           "velocity_m_s", "friction_factor" at minimum.

    Returns
    -------
    str
    """
    Re      = hydraulic_state_dict.get("reynolds_number", 0)
    regime  = hydraulic_state_dict.get("regime", "unknown")
    v       = hydraulic_state_dict.get("velocity_m_s", 0)
    f       = hydraulic_state_dict.get("friction_factor_darcy", 0)
    dP      = hydraulic_state_dict.get("pipe_pressure_drop_kPa", 0)
    warning = hydraulic_state_dict.get("warning", "")

    lines = [
        f"Regime: {regime.upper()}  (Re = {Re:,.0f})",
        f"Velocity: {v:.2f} m/s  |  Darcy f = {f:.5f}  |  Pipe ΔP = {dP:.2f} kPa",
    ]
    if warning:
        lines.append(f"⚠ {warning}")
    return "\n".join(lines)


def fanning_to_darcy(f_fanning: float) -> float:
    """Convert Fanning friction factor to Darcy-Weisbach: f_D = 4 · f_F."""
    return 4.0 * f_fanning


def darcy_to_fanning(f_darcy: float) -> float:
    """Convert Darcy-Weisbach friction factor to Fanning: f_F = f_D / 4."""
    return f_darcy / 4.0
