"""Separate trajectory optimizer using the original 3CC fatigue dynamics (r=1).

Run: python -m code.msk_optimization_3cc --model tsm_3DOF

Uses the same posture-dependent force maps as the 3CC visual. Target load is
clip(force / (reference_scale * maximum_isometric_force), 0, 1). This is a
muscle-level load proxy, not a dynamic muscle recruitment calculation.
Defaults match the current msk_optimization module run: 600 nodes, dt=0.01,
two muscles, the same translating sphere, and 0.1*dt*sum(mf**2).
This compares fatigue models, not just the hard force threshold: 3CC adds an
active compartment, different rates and integration, and a piecewise controller.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp

from .msk_data import Interpolants, JointStates, MuscleStates, OCPResult, MODEL_CONFIG
from .msk_optimization import MusculoskeletalOCP, OCPconfig, SecondOrderIntegrator
from .msk_fatiguemodel import FatigueModel
from .sk_kinematics import Kinematics
from .tasks import SphericalRegion


COMPARISON_MUSCLES = ("DeltoideusScapula_M", "Infraspinatus_S")


@dataclass(frozen=True)
class ThreeCCParameters:
    """3CC parameters with fatigue production scaled to the threshold model.

    F = Fmax / capacity = -log(1 - 0.993) / 10 for the current dataset.
    This matches its active-branch production coefficient per unit target load;
    the compartment equations and recovery law remain those of 3CC.
    """
    fatigue_rate: float = -np.log(1 - 0.993) / 10.0
    recovery_rate: float = 0.00094
    development: float = 10.0
    relaxation: float = 10.0
    reference_scale: float = 1.0

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("3CC parameters must be finite.")
        if min(self.fatigue_rate, self.recovery_rate) < 0:
            raise ValueError("Fatigue and recovery rates must be nonnegative.")
        if min(self.development, self.relaxation, self.reference_scale) <= 0:
            raise ValueError("Controller gains and force reference scale must be positive.")


class ThreeCCModel:
    def __init__(self, interpolants, muscle_states, parameters=None):
        self.parameters = parameters or ThreeCCParameters()
        self.interpolants = interpolants
        self.muscle_states = muscle_states
        self.columns = [interpolants.muscle_names.index(name) for name in muscle_states.names]
        if np.any(np.asarray(muscle_states.max_forces) <= 0):
            raise ValueError("Maximum muscle forces must be positive.")
        self.force_map = interpolants.interpolant("muscle_force")

    def target_load(self, q):
        force = self.force_map(q)[self.columns]
        reference = self.muscle_states.max_forces * self.parameters.reference_scale
        return ca.fmin(ca.fmax(force / reference, 0), 1)

    def derivative(self, q, fatigued, active):
        """Return [dmf/dt, dma/dt]; mr = 1 - ma - mf.

        The implied dmr/dt is -C + R*mf, exactly the third 3CC equation.
        Algebraic elimination avoids redundant conservation constraints in IPOPT.
        """
        p = self.parameters
        resting = 1 - fatigued - active
        deficit = self.target_load(q) - active
        controller = ca.if_else(deficit > 0,
                                p.development * ca.fmin(deficit, resting),
                                p.relaxation * deficit)
        return ca.vertcat(p.fatigue_rate * active - p.recovery_rate * fatigued,
                         controller - p.fatigue_rate * active)


@dataclass
class ThreeCCResult(OCPResult):
    """states/val_states retain [joint states; mf] for existing plotting tools.

    Validation uses RK45 along linearly interpolated optimized joint positions,
    with the same interpolated force map (not a nearest-neighbor validation).
    """
    active_states: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    resting_states: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    val_active_states: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    val_resting_states: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    task_parameters: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    task_rates: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    parameters_3cc: dict = field(default_factory=dict)
    endpoint_force: np.ndarray = field(default_factory=lambda: np.empty(0))
    target_loads: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    joint_names: tuple = ()
    control_label: str = "Control"
    integration_substeps: int = 1
    initial_joint_positions: np.ndarray = field(default_factory=lambda: np.empty(0))
    validation_method: str = "RK45; interpolated force map; linear joint-position interpolation"


class Musculoskeletal3CCOCP(MusculoskeletalOCP):
    def __init__(self, kinematics, f_dynamics, i_dynamics, fatigue, task,
                 joint_states, muscle_states, conf):
        if conf.N < 2 or not np.isfinite(conf.dt) or conf.dt <= 0 or not muscle_states.names:
            raise ValueError("Need at least two nodes, positive dt, and at least one muscle.")
        super().__init__(kinematics, f_dynamics, i_dynamics, fatigue, task,
                         joint_states, muscle_states, conf)
        self.active = self.opti.variable(len(muscle_states.names), conf.N)
        self.resting = 1 - self.active - self.X[joint_states.size:, :]
        self.opti.set_initial(self.active, 0)
        p = fatigue.parameters
        # Resolve controller transients even when the trajectory nodes are coarse.
        max_step = min(0.02, 0.2 / max(p.development, p.relaxation,
                                     p.fatigue_rate, p.recovery_rate))
        self.integration_substeps = max(1, int(np.ceil(conf.dt / max_step)))
        self._step = self._make_step()
        self._built = False

    @classmethod
    def from_model(cls, model, interpolants, task, horizon=600, timestep=0.01,
                   muscles=None, parameters=None, f_dynamics_func=None,
                   i_dynamics_func=None):
        maps = Interpolants.from_npz(interpolants) if isinstance(interpolants, (str, Path)) else interpolants
        joint = JointStates.from_model(model)
        if tuple(maps.joint_coordinate_names) != tuple(joint.names):
            raise ValueError("Interpolant and model joint coordinate orders must match.")
        muscle = MuscleStates.from_model(model, muscles=maps.muscle_names if muscles is None else muscles)
        forward = (ca.Function.load(str(f_dynamics_func)) if f_dynamics_func
                   else SecondOrderIntegrator(joint.size // 2))
        inverse = ca.external("F", str(i_dynamics_func)) if i_dynamics_func else None
        if inverse is None:
            # The parent initializer uses inverse dynamics only for its control guess.
            inverse = lambda state: np.zeros(joint.torques.size)
        return cls(Kinematics(model), forward, inverse, ThreeCCModel(maps, muscle, parameters),
                   task, joint, muscle, OCPconfig.from_model(model, horizon, timestep))

    def _make_step(self):
        n, m = self.joint_states.size, len(self.muscle_states.names)
        z = ca.MX.sym("state", n + 2*m)
        u = ca.MX.sym("control", self.U.size1())
        rhs = ca.Function("rhs_3cc", [z, u], [ca.vertcat(
            self.f_dynamics(ca.vertcat(z[:n], u)),
            self.fatigue.derivative(z[:n:2], z[n:n+m], z[n+m:]))])
        value = z
        h = self.conf.dt / self.integration_substeps
        for _ in range(self.integration_substeps):
            k1 = rhs(value, u)
            k2 = rhs(value + h*k1/2, u)
            k3 = rhs(value + h*k2/2, u)
            k4 = rhs(value + h*k3, u)
            value += h*(k1 + 2*k2 + 2*k3 + k4)/6
        return ca.Function("step_3cc", [z, u], [value])

    def solve(self):
        n, m, N = self.joint_states.size, len(self.muscle_states.names), self.conf.N
        positions = np.arange(0, n, 2)
        endpoint = self.kinematics.endpoint_function
        if not self._built:
            self.opti.subject_to(self.X[n:, 0] == 0)
            self.opti.subject_to(self.active[:, 0] == 0)  # initially fully rested
            self.opti.subject_to(self.opti.bounded(0, self.X[n:, :], 1))
            self.opti.subject_to(self.opti.bounded(0, self.active, 1))
            self.opti.subject_to(ca.vec(self.resting) >= 0)
            for i, row in enumerate(positions):
                lo, hi = self.joint_states.angle_limits[i]
                self.opti.subject_to(self.opti.bounded(lo, self.X[row, :], hi))
            for i, limit in enumerate(self.conf.control_lim):
                self.opti.subject_to(self.opti.bounded(-limit, self.U[i, :], limit))
            self.task.add_boundary_constraints(self.opti, self.task_params, self.task_param_rates,
                                              endpoint=endpoint(self.X[positions, -1]))
            for k in range(N - 1):
                self.task.add_path_constraints(self.opti, self.task_params, self.task_param_rates,
                                              k, self.conf.dt, endpoint=endpoint(self.X[positions, k]))
                z = ca.vertcat(self.X[:, k], self.active[:, k])
                self.opti.subject_to(ca.vertcat(self.X[:, k+1], self.active[:, k+1])
                                     == self._step(z, self.U[:, k]))
            # Match the baseline's sum of individual squared fatigue states,
            # including its time-step scaling and weight.
            self.opti.minimize(0.1 * self.conf.dt * ca.sumsqr(self.X[n:, :]))
            self.opti.solver("ipopt", self.conf.p_opts, self.conf.s_opts)
            self._built = True
        initial_joint_positions = np.asarray(self.opti.debug.value(
            self.X[positions, 0], self.opti.initial())).reshape(-1)
        solution = self.opti.solve()
        states = np.asarray(solution.value(self.X)).reshape(n+m, N)
        active = np.asarray(solution.value(self.active)).reshape(m, N)
        self.task_parameters_solution = np.asarray(solution.value(self.task_params)).reshape(-1, N)
        self.task_param_rates_solution = np.asarray(solution.value(self.task_param_rates)).reshape(-1, N-1)
        time = np.arange(N) * self.conf.dt
        validated = self._validate(time, states[positions], np.r_[states[n:, 0], active[:, 0]])
        self.ocp_result = ThreeCCResult(
            time=time, states=states, val_states=np.vstack((states[:n], validated[:m])),
            controls=np.asarray(solution.value(self.U)).reshape(self.U.size1(), N-1),
            pos_indices=positions, endpoint_positions=np.asarray(endpoint(states[positions])),
            task=np.column_stack([np.asarray(self.task.evaluate(p)).reshape(3)
                                  for p in self.task_parameters_solution.T]),
            objective=float(solution.value(self.opti.f)), muscle_names=self.muscle_states.names,
            active_states=active, resting_states=1-active-states[n:],
            val_active_states=validated[m:], val_resting_states=1-validated[m:]-validated[:m],
            task_parameters=self.task_parameters_solution, task_rates=self.task_param_rates_solution,
            parameters_3cc=asdict(self.fatigue.parameters),
            endpoint_force=np.asarray(self.fatigue.interpolants.force).copy(),
            target_loads=np.column_stack([np.asarray(self.fatigue.target_load(q)).ravel()
                                         for q in states[positions].T]),
            joint_names=tuple(self.joint_states.names),
            control_label=("Acceleration [rad/s²]" if isinstance(self.f_dynamics, SecondOrderIntegrator)
                           else "Control (forward-dynamics input)"),
            integration_substeps=self.integration_substeps,
            initial_joint_positions=initial_joint_positions,
        )
        return self.ocp_result

    def _validate(self, time, q, initial):
        m = len(self.muscle_states.names)
        def rhs(t, state):
            position = np.array([np.interp(t, time, row) for row in q])
            return np.asarray(self.fatigue.derivative(position, state[:m], state[m:])).ravel()
        solution = solve_ivp(rhs, (time[0], time[-1]), initial, t_eval=time,
                             max_step=self.conf.dt, rtol=1e-7, atol=1e-10)
        if not solution.success:
            raise RuntimeError(f"3CC forward validation failed: {solution.message}")
        return solution.y

    def save_result(self, model_name, *, dir=None, tag=None):
        directory = dir or Path(__file__).resolve().parent.parent / "results" / model_name / "ocp_3cc" / "results"
        return super().save_result(model_name, dir=directory, tag=tag)

    def print_diagnostics(self):
        r = self.ocp_result
        n = self.joint_states.size
        z = np.vstack((r.states, r.active_states))
        defects = np.column_stack([z[:, k+1] - np.asarray(self._step(z[:, k], r.controls[:, k])).ravel()
                                   for k in range(self.conf.N-1)])
        compartments = np.vstack((r.states[n:], r.active_states, r.resting_states))
        print("\n3CC optimization diagnostics")
        for name, value in {
            "Objective": r.objective,
            "Max joint dynamics residual": np.max(np.abs(defects[:n])),
            "Max 3CC dynamics residual": np.max(np.abs(defects[n:])),
            "Max compartment bound violation": max(0, -compartments.min(), compartments.max()-1),
            "Maximum fatigued fraction": np.max(r.states[n:]),
            "Max mf difference from RK45 replay": np.max(np.abs(r.states[n:]-r.val_states[n:])),
        }.items():
            print(f"  {name:<40}: {value:.3e}")


def plot_ocp_3cc(result, *, show=True, block=True, output_dir=None):
    """Plot a result or trusted saved NPZ without loading a model or force maps.

    Returns three Matplotlib figures: compartments/derivatives, joint motion,
    and endpoint tracking. Optional PNG exports use fixed descriptive names.
    Dashed compartment curves are the independent RK45 replay.
    """
    import matplotlib.pyplot as plt

    if isinstance(result, (str, Path)):
        result = OCPResult.from_npz(result)
    if not isinstance(result, ThreeCCResult):
        raise ValueError("Expected a 3CC optimization result containing all compartments.")
    t = np.asarray(result.time)
    nq = len(result.pos_indices)
    n_joint = 2 * nq
    mf = result.states[n_joint:]
    ma, mr = result.active_states, result.resting_states
    if mf.shape != ma.shape or mf.shape != mr.shape or t.size < 2:
        raise ValueError("Invalid or empty 3CC compartment trajectories.")
    muscles = result.muscle_names
    loads = getattr(result, "target_loads", np.empty((0, 0)))
    has_loads = loads.shape == mf.shape
    parameters = result.parameters_3cc
    dmf = parameters["fatigue_rate"] * ma - parameters["recovery_rate"] * mf
    derivatives = None
    if has_loads:
        deficit = loads - ma
        drive = np.where(deficit > 0, parameters["development"] * np.minimum(deficit, mr),
                         parameters["relaxation"] * deficit)
        derivatives = (drive - parameters["fatigue_rate"] * ma,
                       -drive + parameters["recovery_rate"] * mf, dmf)

    compartments, axs = plt.subplots(len(muscles), 2, squeeze=False,
                                      figsize=(13, max(4, 3.2 * len(muscles))), sharex=True)
    compartments.canvas.manager.set_window_title("3CC compartments and derivatives")
    colors = ("tab:blue", "tab:orange", "tab:green")
    validation = (result.val_active_states, result.val_resting_states, result.val_states[n_joint:])
    for i, name in enumerate(muscles):
        state_ax, rate_ax = axs[i]
        for label, values, replay, color in zip(("Active ma", "Resting mr", "Fatigued mf"),
                                              (ma, mr, mf), validation, colors):
            state_ax.plot(t, values[i], color=color, label=label)
            if replay.shape == mf.shape:
                state_ax.plot(t, replay[i], "--", color=color, alpha=0.65,
                              label=f"{label.split()[0]} RK45")
        if has_loads:
            state_ax.plot(t, loads[i], ":", color="black", label="Target load TL")
        state_ax.set(title=name, ylabel="Compartment fraction / target load", ylim=(-0.03, 1.03))
        if derivatives is not None:
            for label, values, color in zip(("dma/dt", "dmr/dt", "dmf/dt"), derivatives, colors):
                rate_ax.plot(t, values[i], color=color, label=label)
        else:
            rate_ax.plot(t, dmf[i], color=colors[2], label="dmf/dt (saved TL unavailable)")
        rate_ax.axhline(0, color="gray", linewidth=0.7)
        rate_ax.set(title=f"{name}: instantaneous derivatives", ylabel="Rate [s⁻¹]")
        for ax in (state_ax, rate_ax):
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, ncol=2)
    for ax in axs[-1]:
        ax.set_xlabel("Time [s]")
    compartments.suptitle("3CC solution — solid: optimized; dashed: RK45 replay")
    compartments.tight_layout()

    motion, axs = plt.subplots(nq, 3, squeeze=False, figsize=(14, max(3.5, 2.8 * nq)), sharex=True)
    motion.canvas.manager.set_window_title("3CC joint motion and controls")
    names = getattr(result, "joint_names", ()) or tuple(f"Joint {i}" for i in range(nq))
    for i, row in enumerate(result.pos_indices):
        axs[i, 0].plot(t, result.states[row])
        axs[i, 1].plot(t, result.states[row + 1])
        # Controls apply over [time[k], time[k+1]).
        axs[i, 2].step(t, np.r_[result.controls[i], result.controls[i, -1]], where="post")
        for ax, label in zip(axs[i], ("Position [rad]", "Velocity [rad/s]",
                                     getattr(result, "control_label", "Control"))):
            ax.set(title=names[i], ylabel=label)
            ax.grid(alpha=0.25)
    for ax in axs[-1]:
        ax.set_xlabel("Time [s]")
    motion.tight_layout()

    tracking = plt.figure(figsize=(12, 5))
    tracking.canvas.manager.set_window_title("3CC endpoint tracking")
    path_ax = tracking.add_subplot(121, projection="3d")
    error_ax = tracking.add_subplot(122)
    path_ax.plot(*result.endpoint_positions, label="Endpoint")
    path_ax.plot(*result.task, "--", label="Task")
    path_ax.scatter(*result.endpoint_positions[:, 0], marker="o", label="Start")
    path_ax.scatter(*result.endpoint_positions[:, -1], marker="x", label="End")
    path_ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="Endpoint trajectory")
    path_ax.legend()
    error = np.linalg.norm(result.endpoint_positions - result.task, axis=0)
    error_ax.plot(t, error)
    error_ax.set(xlabel="Time [s]", ylabel="Endpoint-to-task distance [m]",
                 title=f"Tracking distance (RMS {np.sqrt(np.mean(error**2)):.4g} m)")
    error_ax.grid(alpha=0.25)
    tracking.tight_layout()
    figures = {"compartments": compartments, "joint_motion": motion, "tracking": tracking}
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            fig.savefig(directory / f"3cc_{name}.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show(block=block)
    return figures


def solve_ocp_3cc(model_name="tsm_3DOF", *, task=None, horizon=600, timestep=0.01,
                   muscles=COMPARISON_MUSCLES, parameters=None, interpolants=None, tag="test", save_motion=True,
                   visualize=True, plot_dir=None):
    import opensim as osim
    config = MODEL_CONFIG[model_name]
    model = osim.Model(config["model_path"])
    task = task if task is not None else SphericalRegion(
        center=np.array([-0.22686198, 0.39980641, 0.28600239]), radius=0.1)
    if interpolants is None:
        # Rebuild from the baseline dataset and force_index=0 instead of relying
        # on an interpolants archive that another run may have overwritten.
        selected = MuscleStates.from_model(model, muscles=muscles)
        maps = FatigueModel.from_npz(config["dataset_path"], selected).interpolants
    else:
        maps = interpolants
    ocp = Musculoskeletal3CCOCP.from_model(model, maps, task, horizon, timestep,
                                          muscles=muscles, parameters=parameters,
                                          i_dynamics_func=config["i_dynamics_func_path"])
    result = ocp.solve()
    ocp.print_diagnostics()
    print(f"Saved 3CC result: {ocp.save_result(model_name, tag=tag)}")
    if save_motion:
        from .visuals.visualizer import Visualizer
        root = Path(__file__).resolve().parent.parent
        Visualizer(model).save_motion_sto(root / "results" / model_name / "ocp_3cc" / "motions" /
                                         f"{result.tag}.sto", result.time, result.states[:ocp.joint_states.size])
    if visualize or plot_dir is not None:
        plot_ocp_3cc(result, show=visualize, output_dir=plot_dir)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tsm_3DOF", choices=MODEL_CONFIG)
    parser.add_argument("--horizon", type=int, default=600)
    parser.add_argument("--timestep", type=float, default=0.01)
    parser.add_argument("--muscles", nargs="+", default=list(COMPARISON_MUSCLES))
    parser.add_argument("--interpolants", type=Path)
    parser.add_argument("--tag", default="test")
    parser.add_argument("--save-motion", dest="save_motion", action="store_true", default=True)
    parser.add_argument("--no-save-motion", dest="save_motion", action="store_false")
    parser.add_argument("--no-plots", action="store_true", help="Do not open plot windows")
    parser.add_argument("--plot-dir", type=Path, help="Export the plots as PNGs")
    parser.add_argument("--result", type=Path, help="Plot a trusted saved 3CC NPZ without solving again")
    for name, default in asdict(ThreeCCParameters()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=default)
    args = parser.parse_args()
    if args.result is not None:
        plot_ocp_3cc(args.result, show=not args.no_plots, output_dir=args.plot_dir)
        return
    parameters = ThreeCCParameters(**{key: getattr(args, key) for key in asdict(ThreeCCParameters())})
    solve_ocp_3cc(args.model, horizon=args.horizon, timestep=args.timestep, muscles=args.muscles,
                  parameters=parameters, interpolants=args.interpolants, tag=args.tag,
                  save_motion=args.save_motion, visualize=not args.no_plots, plot_dir=args.plot_dir)


if __name__ == "__main__":
    # Pickled result classes must retain an importable module name when run via -m.
    from .msk_optimization_3cc import main as run_main
    run_main()
