"""Diagnostics for inspecting musculoskeletal OCP solutions."""

import casadi as ca
import numpy as np


def evaluate_constraint_violations(ocp, result=None, *, task_parameters=None,
                                   task_param_rates=None, include_dynamics=True):
    """Return maximum constraint violations and fatigue diagnostics for a result."""
    result = ocp.ocp_result if result is None else result
    if result.states.size == 0:
        raise ValueError("No OCP result is available to evaluate.")

    def max_abs(value):
        return float(np.max(np.abs(value))) if np.size(value) else 0.0

    def bound_violation(values, lower, upper):
        values = np.asarray(values, dtype=float)
        return float(max(np.max(np.maximum(lower - values, 0.0)),
                         np.max(np.maximum(values - upper, 0.0))))

    n_joint = ocp.joint_states.size
    q_rows = result.pos_indices
    joint_positions = result.states[q_rows, :]
    fatigue_states = result.states[n_joint:, :]
    report = {
        "joint_position_limit": bound_violation(
            joint_positions, ocp.joint_states.angle_limits[:, 0, None],
            ocp.joint_states.angle_limits[:, 1, None]),
        "control_limit": bound_violation(
            result.controls, -ocp.conf.control_lim[:, None], ocp.conf.control_lim[:, None]),
        "initial_fatigue_state": max_abs(fatigue_states[:, 0]),
        "fatigue_state_min": float(np.min(fatigue_states)),
        "fatigue_state_max": float(np.max(fatigue_states)),
        "fatigue_beyond_unit_interval": float(max(
            np.max(np.maximum(-fatigue_states, 0.0)),
            np.max(np.maximum(fatigue_states - 1.0, 0.0)))),
        "objective": float(result.objective),
        "fatigue_objective": float(ocp.conf.w_fatigue * np.sum(np.mean(fatigue_states, axis=0) ** 2)),
    }

    parameters = task_parameters if task_parameters is not None else ocp.task_parameters_solution
    rates = task_param_rates if task_param_rates is not None else ocp.task_param_rates_solution
    if parameters is not None and rates is not None:
        p_lo, p_hi = np.asarray(ocp.task.parameter_bounds).T
        r_lo, r_hi = np.asarray(ocp.task.rate_bounds).T
        report["task_parameter_bounds"] = bound_violation(parameters, p_lo[:, None], p_hi[:, None])
        report["task_rate_bounds"] = bound_violation(rates, r_lo[:, None], r_hi[:, None])
        report["initial_task_parameters"] = max_abs(parameters[:, 0] - ocp.task.initial_parameters)
        report["initial_task_rates"] = max_abs(rates[:, 0] - ocp.task.initial_rates)
        report["task_parameter_integration"] = max_abs(
            parameters[:, 1:] - parameters[:, :-1] - rates * ocp.conf.dt)
        task_points = np.column_stack([
            np.asarray(ocp.task.evaluate(parameters[:, k]), dtype=float).reshape(3)
            for k in range(parameters.shape[1])
        ])
        tracking_distance = np.linalg.norm(result.endpoint_positions - task_points, axis=0)
        if hasattr(ocp.task, "radius"):
            path_distances = tracking_distance[:-1]
            report["task_region_violation"] = float(max(
                np.max(np.maximum(path_distances - ocp.task.radius, 0.0)) if path_distances.size else 0.0,
                max(0.0, tracking_distance[-1] - ocp.task.radius)))
        else:
            report["rms_task_tracking_error"] = float(np.sqrt(np.mean(tracking_distance ** 2)))
    else:
        report["task_constraints"] = "unavailable: task parameter trajectories were not supplied or retained"

    if include_dynamics:
        joint_residuals, fatigue_residuals = [], []
        for k in range(result.controls.shape[1]):
            xk, uk = result.states[:n_joint, k], result.controls[:, k]
            f = lambda state: np.asarray(ocp.f_dynamics(ca.DM(np.r_[state, uk]))).reshape(-1)
            dt = ocp.conf.dt
            k1 = f(xk)
            k2 = f(xk + dt * k1 / 2)
            k3 = f(xk + dt * k2 / 2)
            k4 = f(xk + dt * k3)
            predicted = xk + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            joint_residuals.append(result.states[:n_joint, k + 1] - predicted)
            df = np.asarray(ocp.fatigue.muscle_state_derivative(
                ca.DM(joint_positions[:, k]), ca.DM(fatigue_states[:, k]),
                from_interpolants=True)).reshape(-1)
            fatigue_residuals.append(fatigue_states[:, k + 1] - fatigue_states[:, k] - dt * df)
        report["joint_dynamics_equality"] = max_abs(joint_residuals)
        report["fatigue_dynamics_equality"] = max_abs(fatigue_residuals)

    if result.val_states.size and result.val_states.shape == result.states.shape:
        validation_fatigue = result.val_states[n_joint:, :]
        report["fatigue_interpolant_vs_nearest_neighbor_max_difference"] = max_abs(
            fatigue_states - validation_fatigue)
        report["nearest_neighbor_fatigue_min"] = float(np.min(validation_fatigue))
        report["nearest_neighbor_fatigue_max"] = float(np.max(validation_fatigue))
    return report


def print_constraint_violations(ocp, result=None, *, task_parameters=None,
                                task_param_rates=None, include_dynamics=True):
    """Print diagnostics in readable groups; return nothing to avoid dict echo."""
    report = evaluate_constraint_violations(
        ocp, result, task_parameters=task_parameters,
        task_param_rates=task_param_rates, include_dynamics=include_dynamics)
    groups = {
        "Constraint violations (max)": (
            "joint_position_limit", "control_limit", "task_parameter_bounds",
            "task_rate_bounds", "task_region_violation"),
        "Equality residuals (max abs)": (
            "initial_fatigue_state", "initial_task_parameters", "initial_task_rates",
            "task_parameter_integration", "joint_dynamics_equality", "fatigue_dynamics_equality"),
        "Fatigue impact": (
            "objective", "fatigue_objective", "fatigue_state_min", "fatigue_state_max",
            "fatigue_beyond_unit_interval", "fatigue_interpolant_vs_nearest_neighbor_max_difference",
            "nearest_neighbor_fatigue_min", "nearest_neighbor_fatigue_max"),
    }
    grouped_keys = {key for keys in groups.values() for key in keys}
    print("\nOCP constraint and fatigue diagnostics")
    for title, keys in groups.items():
        rows = [(key, report[key]) for key in keys if key in report]
        if rows:
            print(f"\n{title}")
            width = max(len(key.replace("_", " ")) for key, _ in rows)
            for key, value in rows:
                print(f"  {key.replace('_', ' ').capitalize():<{width}} : {value:.3e}")
    for key, value in report.items():
        if key not in grouped_keys:
            print(f"  {key.replace('_', ' ').capitalize()} : {value}")
