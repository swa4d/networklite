#pragma once
/*
 * transportsim.hpp  –  High-performance pipeline hydraulics (v2)
 *
 * Models single-phase flow through pipelines connecting CSTRs and
 * Source/Sink nodes. Implements Darcy-Weisbach pressure drop with
 * Colebrook-White friction factor.
 *
 * v2 additions:
 *   - PumpSolution / solve_pump_delta_p  → auto-balance engine
 *   - PumpCurveSweepResult / pump_curve_sweep → power-vs-flow curves
 *   - fanning_friction_factor  → Fanning f = Darcy f / 4
 *   - FanningResult  → separate ΔP calculation using Fanning friction
 *   - FlowRegimeSummary / classify_regime → regime diagnostics
 *
 * Design assumptions (per spec):
 *   - Every pipe has a virtual pump at its inlet.
 *   - CSTRs and Source/Sink nodes operate at constant pressure.
 *   - Default density = 1000 kg/m³ (water) for unspecified species.
 *   - Roughness default: commercial steel ε = 4.6e-5 m.
 *   - Pump efficiency is per-pipe, default 0.75.
 *   - Pressure units in user-facing output: kPa.
 */

#include <cmath>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace transportsim {

// ── Physical constants ──────────────────────────────────────────────────────

constexpr double DENSITY_WATER    = 1000.0;   // kg/m³
constexpr double VISCOSITY_WATER  = 1.002e-3;  // Pa·s at 20°C
constexpr double P_ATM            = 101325.0;  // Pa
constexpr double GRAVITY          = 9.80665;   // m/s²
constexpr double ROUGHNESS_STEEL  = 4.6e-5;    // m, commercial steel

// ── Data structures ─────────────────────────────────────────────────────────

struct FluidProps {
    double density    = DENSITY_WATER;
    double viscosity  = VISCOSITY_WATER;
    std::string phase = "liquid";
};

struct PipelineGeometry {
    double length           = 10.0;
    double diameter         = 0.05;
    double roughness        = ROUGHNESS_STEEL;
    double elevation_change = 0.0;
    int    n_fittings_K     = 0;
};

struct FlowConditions {
    double flow_rate_m3s  = 1e-4;
    double inlet_pressure = P_ATM;
    double temperature_K  = 298.15;
};

struct HydraulicResult {
    double outlet_pressure_Pa  = 0.0;
    double pressure_drop_Pa    = 0.0;
    double pressure_drop_kPa   = 0.0;   // convenience
    double velocity_m_s        = 0.0;
    double reynolds_number     = 0.0;
    double friction_factor     = 0.0;   // Darcy-Weisbach f
    double fanning_factor      = 0.0;   // f_Fanning = f_Darcy / 4
    double head_loss_m         = 0.0;
    double minor_loss_Pa       = 0.0;
    double gravity_loss_Pa     = 0.0;
    bool   flow_is_turbulent   = true;
    bool   needs_pump          = true;  // always true in v2 model
    double required_delta_p_Pa = 0.0;
    double required_delta_p_kPa= 0.0;
    std::string regime         = "turbulent";
    std::string warning        = "";
};

// ── Pump structures (v2) ─────────────────────────────────────────────────────

struct PumpSolution {
    double required_delta_p_Pa  = 0.0;   // Pa the pump must produce
    double required_delta_p_kPa = 0.0;   // kPa (display units)
    double shaft_power_kW       = 0.0;   // kW at given efficiency
    double efficiency           = 0.75;  // user-settable per pipe
    double flow_rate_m3s        = 0.0;
    bool   feasible             = true;  // false if ΔP > 5000 kPa (unreasonably high)
    std::string message         = "";
};

struct PumpCurveSweepResult {
    std::vector<double> flow_rates;        // m³/s
    std::vector<double> pressure_drops;    // Pa
    std::vector<double> pressure_drops_kPa;
    std::vector<double> pump_delta_p;      // Pa (= pressure_drop for auto-balance)
    std::vector<double> pump_delta_p_kPa;
    std::vector<double> pump_power_kW;     // kW
    std::vector<double> reynolds_numbers;
    std::vector<double> friction_factors;
    std::vector<double> velocities;
    std::vector<bool>   flow_is_turbulent;
};

// Legacy sweep (kept for compatibility)
struct PipelineSweepResult {
    std::vector<double> flow_rates;
    std::vector<double> pressure_drops;
    std::vector<double> reynolds_numbers;
    std::vector<double> friction_factors;
    std::vector<double> velocities;
    std::vector<bool>   needs_compressor;
};

struct FlowRegimeSummary {
    std::string regime;          // "laminar" | "transitional" | "turbulent"
    double      reynolds_number;
    bool        is_turbulent;
    bool        is_laminar;
    bool        is_transitional;
    std::string diagnostic;      // human-readable string for engineer
    std::string recommendation;
};

// ── Friction factor solvers ─────────────────────────────────────────────────

/**
 * Colebrook-White (Darcy friction factor, iterative Newton):
 *   1/√f = -2 log10( ε/(3.7·D) + 2.51/(Re·√f) )
 */
inline double colebrook_white(double Re, double relative_roughness,
                               int max_iter = 20, double tol = 1e-8) {
    if (Re < 1.0) throw std::invalid_argument("Reynolds number must be > 0");
    if (Re < 2300.0) return 64.0 / Re;  // Hagen-Poiseuille

    double eps_D = relative_roughness;
    // Swamee-Jain initial guess
    double f = std::pow(-1.8 * std::log10((eps_D / 3.7) + (6.9 / Re)), -2.0);
    if (f <= 0.0 || std::isnan(f)) f = 0.02;

    for (int i = 0; i < max_iter; ++i) {
        double sqrt_f = std::sqrt(f);
        double rhs    = -2.0 * std::log10(eps_D / 3.7 + 2.51 / (Re * sqrt_f));
        double f_new  = 1.0 / (rhs * rhs);
        if (std::abs(f_new - f) < tol) return f_new;
        f = f_new;
    }
    return f;
}

/**
 * Churchill correlation – single smooth formula across all regimes.
 */
inline double churchill_friction(double Re, double relative_roughness) {
    if (Re < 1.0) throw std::invalid_argument("Re must be > 0");
    double eps_D = relative_roughness;
    double A = std::pow(
        2.457 * std::log(1.0 / (std::pow(7.0 / Re, 0.9) + 0.27 * eps_D)), 16.0);
    double B  = std::pow(37530.0 / Re, 16.0);
    double f8 = std::pow(8.0 / Re, 12.0) + 1.0 / std::pow(A + B, 1.5);
    return 8.0 * std::pow(f8, 1.0 / 12.0);
}

/**
 * Fanning friction factor = Darcy / 4.
 *
 * Used in the alternative ΔP formulation:
 *   ΔP_Fanning = 4 · f_F · (L/D) · (ρ v² / 2)
 *
 * Note: 4 · f_F · (L/D) = f_D · (L/D), so the ΔP value is identical.
 * The distinction is relevant for correlations expressed in Fanning form.
 */
inline double fanning_friction_factor(double Re, double relative_roughness) {
    return colebrook_white(Re, relative_roughness) / 4.0;
}

// ── Flow regime classifier ──────────────────────────────────────────────────

inline FlowRegimeSummary classify_regime(double Re) {
    FlowRegimeSummary s;
    s.reynolds_number  = Re;
    s.is_turbulent     = (Re >= 4000.0);
    s.is_laminar       = (Re < 2300.0);
    s.is_transitional  = (!s.is_laminar && !s.is_turbulent);

    if (s.is_laminar) {
        s.regime = "laminar";
        s.diagnostic = "Re=" + std::to_string((int)Re) +
                       " < 2300: Hagen-Poiseuille flow. Darcy-Weisbach assumes turbulence; "
                       "calculated ΔP may overestimate actual loss.";
        s.recommendation = "Consider increasing flow rate or pipe diameter to reach "
                           "turbulent regime for more predictable pressure behaviour.";
    } else if (s.is_transitional) {
        s.regime = "transitional";
        s.diagnostic = "Re=" + std::to_string((int)Re) +
                       " in 2300–4000: Transitional regime. Flow is unstable; "
                       "friction factor is uncertain (±40%).";
        s.recommendation = "Design should avoid this regime. Adjust velocity or "
                           "pipe diameter to establish fully turbulent flow (Re > 10000 preferred).";
    } else {
        s.regime = "turbulent";
        s.diagnostic = "Re=" + std::to_string((int)Re) +
                       " ≥ 4000: Turbulent flow. Colebrook-White friction is reliable.";
        s.recommendation = "";
        if (Re > 1e6) {
            s.recommendation = "Very high Re — verify erosion limits (v > 5 m/s?).";
        }
    }
    return s;
}

// ── Core hydraulic calculator ───────────────────────────────────────────────

/**
 * Compute full hydraulic analysis.
 *
 * ΔP_total = ΔP_friction + ΔP_minor + ΔP_gravity
 * Darcy-Weisbach: ΔP_f = f_D · (L/D) · (ρv²/2)
 * Minor:          ΔP_m = K_total · (ρv²/2)
 * Gravity:        ΔP_g = ρ·g·Δz
 *
 * In v2, required_delta_p_Pa = ΔP_total (pump must supply this).
 * We no longer use sub-atmospheric as the pump trigger; instead,
 * the pump always provides ΔP = ΔP_total + (P_outlet_node - P_inlet_node).
 * This function computes ΔP_total only; node pressure correction
 * is done by solve_pump_delta_p.
 */
inline HydraulicResult compute_hydraulics(
    const PipelineGeometry& geom,
    const FlowConditions&   cond,
    const FluidProps&       fluid
) {
    HydraulicResult res;

    double A_pipe = M_PI * std::pow(geom.diameter / 2.0, 2.0);
    if (A_pipe <= 0.0) throw std::invalid_argument("Pipe diameter must be > 0");

    res.velocity_m_s    = cond.flow_rate_m3s / A_pipe;
    res.reynolds_number = fluid.density * res.velocity_m_s * geom.diameter / fluid.viscosity;
    if (res.reynolds_number < 0.0) res.reynolds_number = 0.0;

    auto regime_s       = classify_regime(res.reynolds_number);
    res.flow_is_turbulent = regime_s.is_turbulent;
    res.regime          = regime_s.regime;

    double rel_rough     = geom.roughness / geom.diameter;
    res.friction_factor  = colebrook_white(std::max(res.reynolds_number, 1.0), rel_rough);
    res.fanning_factor   = res.friction_factor / 4.0;

    double q_dyn         = 0.5 * fluid.density * res.velocity_m_s * res.velocity_m_s;
    double dP_friction   = res.friction_factor * (geom.length / geom.diameter) * q_dyn;
    res.minor_loss_Pa    = geom.n_fittings_K * q_dyn;
    res.gravity_loss_Pa  = fluid.density * GRAVITY * geom.elevation_change;
    res.pressure_drop_Pa = dP_friction + res.minor_loss_Pa + res.gravity_loss_Pa;
    res.pressure_drop_kPa= res.pressure_drop_Pa / 1000.0;
    res.head_loss_m      = res.pressure_drop_Pa / (fluid.density * GRAVITY);

    res.outlet_pressure_Pa     = cond.inlet_pressure - res.pressure_drop_Pa;
    res.needs_pump             = true;   // always in v2
    res.required_delta_p_Pa    = res.pressure_drop_Pa;
    res.required_delta_p_kPa   = res.pressure_drop_kPa;

    // Regime warning
    if (!regime_s.is_turbulent) {
        res.warning = regime_s.diagnostic;
    }
    if (res.velocity_m_s > 5.0) {
        res.warning += (res.warning.empty() ? "" : " | ") +
                       std::string("High velocity ") +
                       std::to_string(res.velocity_m_s).substr(0,5) +
                       " m/s — check erosion limits.";
    }

    return res;
}

// ── Auto-balance pump solver ────────────────────────────────────────────────

/**
 * Solve for the pump ΔP required to achieve target_flow_m3s through a
 * pipeline connecting two nodes at known operating pressures.
 *
 * ΔP_pump = ΔP_friction(Q) + ΔP_minor(Q) + ΔP_gravity
 *           + (P_outlet_node - P_inlet_node)
 *
 * Since CSTRs/SourceSinks operate at constant pressure, the node
 * pressures are fixed and the pump simply needs to overcome the sum.
 *
 * If P_outlet_node < P_inlet_node, the pressure gradient assists flow
 * and ΔP_pump may be smaller. If ΔP_pump ≤ 0, no pump is needed but
 * we still size a zero-power pump for consistency.
 *
 * @param efficiency  Per-pipe pump efficiency (0–1). Default 0.75.
 */
inline PumpSolution solve_pump_delta_p(
    const PipelineGeometry& geom,
    double                  target_flow_m3s,
    double                  node_pressure_inlet_Pa,
    double                  node_pressure_outlet_Pa,
    const FluidProps&       fluid,
    double                  efficiency = 0.75
) {
    PumpSolution sol;
    sol.efficiency    = efficiency;
    sol.flow_rate_m3s = target_flow_m3s;

    FlowConditions cond;
    cond.flow_rate_m3s  = target_flow_m3s;
    cond.inlet_pressure = node_pressure_inlet_Pa;

    HydraulicResult hyd = compute_hydraulics(geom, cond, fluid);

    // Node pressure difference (outlet node pressure - inlet node pressure)
    // Positive means outlet is at higher pressure → pump must work harder
    double node_dP = node_pressure_outlet_Pa - node_pressure_inlet_Pa;

    // Total pump ΔP
    double total_dP = hyd.pressure_drop_Pa + node_dP;

    sol.required_delta_p_Pa  = std::max(total_dP, 0.0);
    sol.required_delta_p_kPa = sol.required_delta_p_Pa / 1000.0;

    // Shaft power: W = Q·ΔP / η  (Watts), then /1000 for kW
    if (efficiency > 0.0 && sol.required_delta_p_Pa > 0.0) {
        sol.shaft_power_kW = (target_flow_m3s * sol.required_delta_p_Pa) / (efficiency * 1000.0);
    } else {
        sol.shaft_power_kW = 0.0;
    }

    // Feasibility check: >5000 kPa is unrealistically high — flag for user
    constexpr double MAX_FEASIBLE_kPa = 5000.0;
    sol.feasible = (sol.required_delta_p_kPa <= MAX_FEASIBLE_kPa);

    if (!sol.feasible) {
        sol.message = "Pump ΔP = " + std::to_string(sol.required_delta_p_kPa).substr(0,8) +
                      " kPa exceeds 5000 kPa — check pipe geometry (length/diameter) "
                      "or reduce target flow rate.";
    } else if (total_dP <= 0.0) {
        sol.message = "Pressure gradient assists flow — no active pumping required. "
                      "Pump sized at zero power.";
    } else {
        sol.message = "Pump ΔP = " + std::to_string(sol.required_delta_p_kPa).substr(0,6) +
                      " kPa, Power = " + std::to_string(sol.shaft_power_kW).substr(0,5) +
                      " kW (η=" + std::to_string((int)(efficiency*100)) + "%).";
    }

    return sol;
}

// ── Pump curve sweep (power vs flow) ───────────────────────────────────────

/**
 * Sweep volumetric flow rate and compute both pressure drop and
 * pump power at each point. Used to render the pump operating curve
 * overlaid with the system resistance curve.
 *
 * node_pressure_inlet_Pa / node_pressure_outlet_Pa are the constant
 * pressures of the connected CSTR/SourceSink nodes.
 */
inline PumpCurveSweepResult pump_curve_sweep(
    const PipelineGeometry& geom,
    double                  q_min,
    double                  q_max,
    int                     n_points,
    const FluidProps&       fluid,
    double                  node_pressure_inlet_Pa  = P_ATM,
    double                  node_pressure_outlet_Pa = P_ATM,
    double                  efficiency              = 0.75
) {
    if (n_points < 2) throw std::invalid_argument("n_points must be >= 2");

    PumpCurveSweepResult sweep;
    sweep.flow_rates.resize(n_points);
    sweep.pressure_drops.resize(n_points);
    sweep.pressure_drops_kPa.resize(n_points);
    sweep.pump_delta_p.resize(n_points);
    sweep.pump_delta_p_kPa.resize(n_points);
    sweep.pump_power_kW.resize(n_points);
    sweep.reynolds_numbers.resize(n_points);
    sweep.friction_factors.resize(n_points);
    sweep.velocities.resize(n_points);
    sweep.flow_is_turbulent.resize(n_points);

    for (int i = 0; i < n_points; ++i) {
        double Q = q_min + i * (q_max - q_min) / (n_points - 1);

        FlowConditions fc;
        fc.flow_rate_m3s  = Q;
        fc.inlet_pressure = node_pressure_inlet_Pa;

        auto r = compute_hydraulics(geom, fc, fluid);

        double node_dP     = node_pressure_outlet_Pa - node_pressure_inlet_Pa;
        double pump_dP     = std::max(r.pressure_drop_Pa + node_dP, 0.0);
        double power_kW    = (efficiency > 0.0 && pump_dP > 0.0)
                             ? Q * pump_dP / (efficiency * 1000.0) : 0.0;

        sweep.flow_rates[i]         = Q;
        sweep.pressure_drops[i]     = r.pressure_drop_Pa;
        sweep.pressure_drops_kPa[i] = r.pressure_drop_kPa;
        sweep.pump_delta_p[i]       = pump_dP;
        sweep.pump_delta_p_kPa[i]   = pump_dP / 1000.0;
        sweep.pump_power_kW[i]      = power_kW;
        sweep.reynolds_numbers[i]   = r.reynolds_number;
        sweep.friction_factors[i]   = r.friction_factor;
        sweep.velocities[i]         = r.velocity_m_s;
        sweep.flow_is_turbulent[i]  = r.flow_is_turbulent;
    }
    return sweep;
}

// ── Legacy flow sweep (kept for backward compatibility) ─────────────────────

inline PipelineSweepResult flow_sweep(
    const PipelineGeometry& geom,
    double                  q_min,
    double                  q_max,
    int                     n_points,
    const FluidProps&       fluid,
    double                  inlet_pressure = P_ATM
) {
    if (n_points < 2) throw std::invalid_argument("n_points must be >= 2");
    PipelineSweepResult sweep;
    sweep.flow_rates.resize(n_points);
    sweep.pressure_drops.resize(n_points);
    sweep.reynolds_numbers.resize(n_points);
    sweep.friction_factors.resize(n_points);
    sweep.velocities.resize(n_points);
    sweep.needs_compressor.resize(n_points);

    for (int i = 0; i < n_points; ++i) {
        double Q = q_min + i * (q_max - q_min) / (n_points - 1);
        FlowConditions fc;
        fc.flow_rate_m3s  = Q;
        fc.inlet_pressure = inlet_pressure;
        auto r = compute_hydraulics(geom, fc, fluid);
        sweep.flow_rates[i]       = Q;
        sweep.pressure_drops[i]   = r.pressure_drop_Pa;
        sweep.reynolds_numbers[i] = r.reynolds_number;
        sweep.friction_factors[i] = r.friction_factor;
        sweep.velocities[i]       = r.velocity_m_s;
        sweep.needs_compressor[i] = (r.outlet_pressure_Pa < P_ATM);
    }
    return sweep;
}

} // namespace transportsim
