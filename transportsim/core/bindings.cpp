/*
 * bindings.cpp  –  pybind11 Python bindings for TransportSim v2
 *
 * Exposes all hydraulic calculation functions and data structures
 * to Python as the `_transportsim_core` extension module.
 *
 * v2 additions:
 *   - PumpSolution, solve_pump_delta_p
 *   - PumpCurveSweepResult, pump_curve_sweep
 *   - FlowRegimeSummary, classify_regime
 *   - fanning_friction_factor
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "transportsim.hpp"

namespace py = pybind11;
using namespace transportsim;

PYBIND11_MODULE(_transportsim_core, m) {
    m.doc() = R"(
TransportSim C++ Core v2
========================
High-performance pipeline hydraulics for chemical plant networks.

v2 model: every pipe has a virtual pump at its inlet. CSTRs and
Source/Sink nodes operate at constant pressure. The pump solves for
the ΔP needed to achieve the user-specified volumetric flow rate.

All user-facing pressure values are in kPa.
)";

    // ── FluidProps ──────────────────────────────────────────────────────────
    py::class_<FluidProps>(m, "FluidProps")
        .def(py::init<>())
        .def_readwrite("density",   &FluidProps::density)
        .def_readwrite("viscosity", &FluidProps::viscosity)
        .def_readwrite("phase",     &FluidProps::phase);

    // ── PipelineGeometry ───────────────────────────────────────────────────
    py::class_<PipelineGeometry>(m, "PipelineGeometry")
        .def(py::init<>())
        .def_readwrite("length",            &PipelineGeometry::length)
        .def_readwrite("diameter",          &PipelineGeometry::diameter)
        .def_readwrite("roughness",         &PipelineGeometry::roughness)
        .def_readwrite("elevation_change",  &PipelineGeometry::elevation_change)
        .def_readwrite("n_fittings_K",      &PipelineGeometry::n_fittings_K);

    // ── FlowConditions ─────────────────────────────────────────────────────
    py::class_<FlowConditions>(m, "FlowConditions")
        .def(py::init<>())
        .def_readwrite("flow_rate_m3s",  &FlowConditions::flow_rate_m3s)
        .def_readwrite("inlet_pressure", &FlowConditions::inlet_pressure)
        .def_readwrite("temperature_K",  &FlowConditions::temperature_K);

    // ── HydraulicResult ────────────────────────────────────────────────────
    py::class_<HydraulicResult>(m, "HydraulicResult")
        .def(py::init<>())
        .def_readonly("outlet_pressure_Pa",   &HydraulicResult::outlet_pressure_Pa)
        .def_readonly("pressure_drop_Pa",     &HydraulicResult::pressure_drop_Pa)
        .def_readonly("pressure_drop_kPa",    &HydraulicResult::pressure_drop_kPa)
        .def_readonly("velocity_m_s",         &HydraulicResult::velocity_m_s)
        .def_readonly("reynolds_number",      &HydraulicResult::reynolds_number)
        .def_readonly("friction_factor",      &HydraulicResult::friction_factor)
        .def_readonly("fanning_factor",       &HydraulicResult::fanning_factor)
        .def_readonly("head_loss_m",          &HydraulicResult::head_loss_m)
        .def_readonly("minor_loss_Pa",        &HydraulicResult::minor_loss_Pa)
        .def_readonly("gravity_loss_Pa",      &HydraulicResult::gravity_loss_Pa)
        .def_readonly("flow_is_turbulent",    &HydraulicResult::flow_is_turbulent)
        .def_readonly("needs_pump",           &HydraulicResult::needs_pump)
        .def_readonly("required_delta_p_Pa",  &HydraulicResult::required_delta_p_Pa)
        .def_readonly("required_delta_p_kPa", &HydraulicResult::required_delta_p_kPa)
        .def_readonly("regime",               &HydraulicResult::regime)
        .def_readonly("warning",              &HydraulicResult::warning);

    // ── PumpSolution (v2) ──────────────────────────────────────────────────
    py::class_<PumpSolution>(m, "PumpSolution", R"(
        Result of auto-balance pump sizing for a pipeline.

        Attributes
        ----------
        required_delta_p_Pa  : float  Pa the pump must supply
        required_delta_p_kPa : float  kPa (display units)
        shaft_power_kW       : float  kW at given efficiency
        efficiency           : float  pump efficiency (0-1)
        flow_rate_m3s        : float  design flow rate
        feasible             : bool   False if ΔP > 5000 kPa
        message              : str    human-readable summary
    )")
        .def(py::init<>())
        .def_readonly("required_delta_p_Pa",  &PumpSolution::required_delta_p_Pa)
        .def_readonly("required_delta_p_kPa", &PumpSolution::required_delta_p_kPa)
        .def_readonly("shaft_power_kW",       &PumpSolution::shaft_power_kW)
        .def_readonly("efficiency",           &PumpSolution::efficiency)
        .def_readonly("flow_rate_m3s",        &PumpSolution::flow_rate_m3s)
        .def_readonly("feasible",             &PumpSolution::feasible)
        .def_readonly("message",              &PumpSolution::message);

    // ── PumpCurveSweepResult (v2) ──────────────────────────────────────────
    py::class_<PumpCurveSweepResult>(m, "PumpCurveSweepResult")
        .def(py::init<>())
        .def_readonly("flow_rates",          &PumpCurveSweepResult::flow_rates)
        .def_readonly("pressure_drops",      &PumpCurveSweepResult::pressure_drops)
        .def_readonly("pressure_drops_kPa",  &PumpCurveSweepResult::pressure_drops_kPa)
        .def_readonly("pump_delta_p",        &PumpCurveSweepResult::pump_delta_p)
        .def_readonly("pump_delta_p_kPa",    &PumpCurveSweepResult::pump_delta_p_kPa)
        .def_readonly("pump_power_kW",       &PumpCurveSweepResult::pump_power_kW)
        .def_readonly("reynolds_numbers",    &PumpCurveSweepResult::reynolds_numbers)
        .def_readonly("friction_factors",    &PumpCurveSweepResult::friction_factors)
        .def_readonly("velocities",          &PumpCurveSweepResult::velocities)
        .def_readonly("flow_is_turbulent",   &PumpCurveSweepResult::flow_is_turbulent);

    // ── PipelineSweepResult (legacy) ───────────────────────────────────────
    py::class_<PipelineSweepResult>(m, "PipelineSweepResult")
        .def(py::init<>())
        .def_readonly("flow_rates",        &PipelineSweepResult::flow_rates)
        .def_readonly("pressure_drops",    &PipelineSweepResult::pressure_drops)
        .def_readonly("reynolds_numbers",  &PipelineSweepResult::reynolds_numbers)
        .def_readonly("friction_factors",  &PipelineSweepResult::friction_factors)
        .def_readonly("velocities",        &PipelineSweepResult::velocities)
        .def_readonly("needs_compressor",  &PipelineSweepResult::needs_compressor);

    // ── FlowRegimeSummary (v2) ─────────────────────────────────────────────
    py::class_<FlowRegimeSummary>(m, "FlowRegimeSummary")
        .def(py::init<>())
        .def_readonly("regime",          &FlowRegimeSummary::regime)
        .def_readonly("reynolds_number", &FlowRegimeSummary::reynolds_number)
        .def_readonly("is_turbulent",    &FlowRegimeSummary::is_turbulent)
        .def_readonly("is_laminar",      &FlowRegimeSummary::is_laminar)
        .def_readonly("is_transitional", &FlowRegimeSummary::is_transitional)
        .def_readonly("diagnostic",      &FlowRegimeSummary::diagnostic)
        .def_readonly("recommendation",  &FlowRegimeSummary::recommendation);

    // ── Free functions ─────────────────────────────────────────────────────
    m.def("compute_hydraulics", &compute_hydraulics,
          py::arg("geometry"), py::arg("conditions"), py::arg("fluid"));

    m.def("solve_pump_delta_p", &solve_pump_delta_p, R"(
        Solve for pump ΔP required to achieve target flow between two nodes.

        Parameters
        ----------
        geometry              : PipelineGeometry
        target_flow_m3s       : float  desired volumetric flow (m³/s)
        node_pressure_inlet_Pa: float  constant pressure of source node (Pa)
        node_pressure_outlet_Pa:float  constant pressure of target node (Pa)
        fluid                 : FluidProps
        efficiency            : float  pump efficiency 0-1 (default 0.75)

        Returns
        -------
        PumpSolution
    )", py::arg("geometry"), py::arg("target_flow_m3s"),
        py::arg("node_pressure_inlet_Pa"), py::arg("node_pressure_outlet_Pa"),
        py::arg("fluid"), py::arg("efficiency") = 0.75);

    m.def("pump_curve_sweep", &pump_curve_sweep, R"(
        Sweep flow rate and compute system curve + pump power at each point.

        Returns
        -------
        PumpCurveSweepResult
    )", py::arg("geometry"), py::arg("q_min"), py::arg("q_max"),
        py::arg("n_points"), py::arg("fluid"),
        py::arg("node_pressure_inlet_Pa")  = P_ATM,
        py::arg("node_pressure_outlet_Pa") = P_ATM,
        py::arg("efficiency")              = 0.75);

    m.def("flow_sweep", &flow_sweep,
          py::arg("geometry"), py::arg("q_min"), py::arg("q_max"),
          py::arg("n_points"), py::arg("fluid"),
          py::arg("inlet_pressure") = P_ATM);

    m.def("colebrook_white", &colebrook_white,
          py::arg("Re"), py::arg("relative_roughness"),
          py::arg("max_iter") = 20, py::arg("tol") = 1e-8);

    m.def("fanning_friction_factor", &fanning_friction_factor, R"(
        Fanning friction factor = Darcy friction factor / 4.

        Parameters
        ----------
        Re                : float  Reynolds number
        relative_roughness: float  ε/D ratio

        Returns
        -------
        float
    )", py::arg("Re"), py::arg("relative_roughness"));

    m.def("classify_regime", &classify_regime, R"(
        Classify flow regime and return engineer-facing diagnostic.

        Parameters
        ----------
        Re : float  Reynolds number

        Returns
        -------
        FlowRegimeSummary
    )", py::arg("Re"));

    // Constants
    m.attr("DENSITY_WATER")   = DENSITY_WATER;
    m.attr("VISCOSITY_WATER") = VISCOSITY_WATER;
    m.attr("P_ATM")           = P_ATM;
    m.attr("ROUGHNESS_STEEL") = ROUGHNESS_STEEL;
    m.attr("__version__")     = "2.0.0";
}
