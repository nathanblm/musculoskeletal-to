from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import casadi as ca
import numpy as np

import opensim as osim
from .sk_kinematics import Kinematics
from .visuals.visualizer import Visualizer
from .msk_fatiguemodel import FatigueModel, ThreeCompartment, Thresholded
from .msk_data import ModelInfo, OCPResult, MSKdataset, MODEL_CONFIG
from .tasks import *
from .msk_data import JointStates, MuscleInfo

def rk4_step(derivative, state, dt):
    k1 = derivative(state)
    k2 = derivative(state + dt * k1 / 2)
    k3 = derivative(state + dt * k2 / 2)
    k4 = derivative(state + dt * k3)
    return state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

@dataclass
class OCPconfig:
    N: int
    dt: float
    control_lim: np.ndarray
    w_control: np.ndarray
    w_tracking: np.ndarray
    w_fatigue: np.ndarray

    p_opts = {
        "expand": False,
        "print_time": False,
    }
    s_opts = {
        "print_level": 5,
        "tol": 1e-4,
        "acceptable_tol": 1e-3,
        "acceptable_constr_viol_tol": 1e-4,
        "acceptable_dual_inf_tol": 1e-3,
        "acceptable_compl_inf_tol": 1e-3,
        "acceptable_iter": 5,
    }


    @classmethod
    def from_model(cls, model, horizon, timestep, *, control_lim=None, w_control=None, w_tracking=None, w_fatigue=None):
        info = ModelInfo.from_model(model)

        if control_lim is None:
            control_lim = np.full(len(info.joint_coordinate_names), 100)
        if w_control is None:
            w_control = np.full(len(info.joint_coordinate_names), 1e-3)
        if w_tracking is None:
            w_tracking = 1.0
        if w_fatigue is None:
            w_fatigue = 1e-3

        return cls(
            N=horizon,
            dt=timestep,
            control_lim=control_lim,
            w_control=w_control,
            w_tracking=w_tracking,
            w_fatigue=w_fatigue,
        )

class SecondOrderIntegrator:
    def __init__(self, n_coordinates):
        self.n_coordinates = n_coordinates

    def __call__(self, state_and_control):
        n_states = 2 * self.n_coordinates
        x = state_and_control[:n_states]
        u = state_and_control[n_states:]

        dx = []
        for i in range(self.n_coordinates):
            dx.extend((x[2*i + 1], u[i]))
        return ca.vertcat(*dx)

class MusculoskeletalOCP:
    def __init__(self, kinematics, f_dynamics, i_dynamics, fatigue_model, task, joint_states, conf, N=None):
        self.kinematics = kinematics
        self.f_dynamics = f_dynamics
        self.i_dynamics = i_dynamics
        self.fatigue_model = fatigue_model
        self.task = task
        self.conf = conf
        self.ocp_result = OCPResult.empty()

        self.joint_states = joint_states

        self.opti = ca.Opti()

        self.X = self.opti.variable(
            self.joint_states.x.size + self.fatigue_model.n_states,
            self.conf.N,
        )
        self.U = self.opti.variable(self.joint_states.torques.size, self.conf.N - 1)       # [a1, a2, ..., an] * (N-1)

        self.task_params = self.opti.variable(
            len(self.task.initial_parameters), self.conf.N
        )
        self.task_param_rates = self.opti.variable(
            len(self.task.initial_rates), self.conf.N - 1
        )

        self.x_initial = self.opti.parameter(self.X.shape[0])
        self.task_initial = self.opti.parameter(self.task_params.shape[0])
        self.task_rate_initial = self.opti.parameter(self.task_param_rates.shape[0])

        self.set_initial_trajectory()
        self.build_ocp()

    @classmethod
    def from_model(
        cls,
        model,
        f_dynamics_func: Path,
        i_dynamics_func: Path,
        dataset: Path,
        task: Task,
        horizon,
        timestep,
        muscles: list[str] = None,
        force_index: int = 0,
        fatigue_model_type: type[FatigueModel] = Thresholded,
    ):
        kinematics = Kinematics(model)
        joint_states = JointStates.from_model(model)
        muscle_info = MuscleInfo.from_model(model, muscles=muscles)
        f_dynamics = ca.Function.load(str(f_dynamics_func)) if f_dynamics_func else SecondOrderIntegrator(joint_states.size // 2)
        i_dynamics = ca.external("F", str(i_dynamics_func))
        fatigue_model = fatigue_model_type(
            MSKdataset.from_npz(dataset), muscle_info, force_index=force_index
        )
        conf = OCPconfig.from_model(model, horizon, timestep)

        return cls(kinematics, f_dynamics, i_dynamics, fatigue_model, task, joint_states, conf)

    def build_ocp(self):
        endpoint_FK = self.kinematics.endpoint_function
        n = self.joint_states.x.size
        u = self.joint_states.torques.size
        dt = self.conf.dt

        position_indices = np.arange(0, n, 2)

        # always use the "measured" state from the plant as the initial state for the OCP
        self.opti.subject_to(self.X[:, 0] == self.x_initial)
        self.opti.subject_to(self.opti.bounded(0, self.X[n:, :], 1))
        if isinstance(self.fatigue_model, ThreeCompartment):
            _, resting, _ = self.fatigue_model.split_state(self.X[n:, :])
            self.opti.subject_to(ca.vec(resting) >= 0)

                # joint limits
        for i, row in enumerate(position_indices):
            lower, upper = self.joint_states.angle_limits[i]
            self.opti.subject_to(
                self.opti.bounded(lower, self.X[row, :], upper)
            )

        # control limits
        for i in range(u):
            lower = -self.conf.control_lim[i]
            upper = self.conf.control_lim[i]
            self.opti.subject_to(self.opti.bounded(lower, self.U[i, :], upper))


        terminal_endpoint = endpoint_FK(self.X[position_indices, -1])
        self.task.add_boundary_constraints(
            self.opti, self.task_params, self.task_param_rates,
            endpoint=terminal_endpoint,
        )

        cost = 0
        for k in range(self.conf.N):
            cost += 0.1 * dt * ca.sumsqr(
                self.fatigue_model.fatigue(self.X[n:, k])
            )

        # path constraints and dynamics
        for k in range(self.conf.N - 1):
            qk = self.X[position_indices, k]
            endpoint = endpoint_FK(qk)

            self.task.add_path_constraints(
                self.opti, self.task_params, self.task_param_rates, k, dt,
                endpoint=endpoint,
            )

            # Integrate joint and fatigue states together, evaluating muscle
            # force at each RK4 substep posture.
            xk = self.X[:, k]
            uk = self.U[:, k]
            f = lambda state: ca.vertcat(
                self.f_dynamics(ca.vertcat(state[:n], uk)),
                self.fatigue_model.derivative(state[position_indices], state[n:]),
            )
            self.opti.subject_to(
                self.X[:, k + 1] == rk4_step(f, xk, dt)
            )

        self.opti.minimize(cost)
        self.opti.solver("ipopt", self.conf.p_opts, self.conf.s_opts)

    def solve(
        self,
        initial_state,
        task_parameters,
        task_rates,
        initial_guess=None,
        *, 
        diagnostics=False,
        validate=False,
    ):
        self.opti.set_value(self.x_initial, initial_state)
        self.opti.set_value(self.task_initial, task_parameters)
        self.opti.set_value(self.task_rate_initial, task_rates)

        if initial_guess is not None:
            for variable, name in (
                (self.X, "states"),
                (self.U, "controls"),
                (self.task_params, "task_parameters"),
                (self.task_param_rates, "task_rates"),
            ):
                self.opti.set_initial(variable, initial_guess[name])

        solution = self.opti.solve()

        n = self.joint_states.x.size
        position_indices = np.arange(0, n, 2)
        endpoint_FK = self.kinematics.endpoint_function
        endpoint_positions = np.asarray(endpoint_FK(solution.value(self.X[position_indices, :])))
        task_values = np.asarray(solution.value(self.task_params), dtype=float).reshape(
            len(self.task.initial_parameters), self.conf.N)
        task_rate_values = np.asarray(solution.value(self.task_param_rates), dtype=float).reshape(
            len(self.task.initial_rates), self.conf.N - 1)
        self.task_parameters_solution = task_values
        self.task_param_rates_solution = task_rate_values
        task_positions = np.column_stack([
            np.asarray(self.task.evaluate(task_values[:, k]), dtype=float).reshape(3)
            for k in range(self.conf.N)
        ])
    
        self.ocp_result = OCPResult(
            time=np.arange(self.conf.N) * self.conf.dt,
            states=solution.value(self.X),
            val_states=np.zeros_like(solution.value(self.X)),
            pos_indices=position_indices,
            controls=solution.value(self.U),
            endpoint_positions=endpoint_positions,
            task=task_positions,
            objective=solution.value(self.opti.f),
            muscle_names=self.fatigue_model.muscles.names,
            fatigue_model=("3cc" if isinstance(self.fatigue_model, ThreeCompartment)
                           else "thresholded" if isinstance(self.fatigue_model, Thresholded) else ""),
        )

        # forward simulation of muscle states using nearest neighbor evaluation of discrete rate and force maps
        # for comparison with the interpolated fatigue dynamics used in the OCP
        if validate:
            self.ocp_result.val_states[:n, :] = self.ocp_result.states[:n, :]
            self.ocp_result.val_states[n:, 0] = np.asarray(initial_state)[n:]
            self.ocp_result.val_states[n:, :] = self.fatigue_model.simulate_muscle_states(
                self.ocp_result.states[position_indices, :].T,
                self.ocp_result.val_states[n:, :],
                dt=self.conf.dt,
                N=self.conf.N,
            )
        else:
            self.ocp_result.val_states = np.empty((0, 0))

        if diagnostics:
            from .msk_opti_debug import print_constraint_violations
            print_constraint_violations(
                self, self.ocp_result,
                task_parameters=task_values,
                task_param_rates=task_rate_values,
            )

        return self.ocp_result

    def set_initial_trajectory(self, trajectory=None):
        if trajectory is not None:
            pass

        from .tasks import initial_state_from_task
        init_state, joint_state, acc = initial_state_from_task(self.task, self.kinematics, self.fatigue_model)
        self.initial_state = init_state.copy()
        init_state_trajectory = np.tile(init_state.reshape(-1, 1), (1, self.conf.N))            # shape (2*n_coordinates + n_muscles, N)
        torques = self.i_dynamics(
            np.concatenate((joint_state, acc)))[:len(acc)]                      # shape (n_coordinates,)
        init_control_trajectory = np.tile(
            np.asarray(torques).reshape(-1, 1),
            (1, self.conf.N - 1)
        )                                                                                       # shape (n_coordinates, N-1)
        self.opti.set_initial(self.X, init_state_trajectory)
        self.opti.set_initial(self.U, init_control_trajectory)
        self.opti.set_initial(
            self.task_params,
            np.tile(self.task.initial_parameters.reshape(-1, 1), (1, self.conf.N)),
        )
        self.opti.set_initial(
            self.task_param_rates,
            np.tile(self.task.initial_rates.reshape(-1, 1), (1, self.conf.N - 1)),
        )

    def save_result(self, model_name: str, *, dir=None, tag=None):
        if dir is None:
            ROOT = Path(__file__).parent.parent
            dir = ROOT / "results" / model_name / "ocp" / "results"
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        if tag is None:
            tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ocp_result.tag = tag

        filename = dir / f"{tag}.npz"
        return self.ocp_result.save_npz(filename)


def solve_ocp(model_name: str, force_index: int = 0, task=None, horizon=100, timestep=0.1, save_motion=False, tag=None, muscles: list[str] = None, *, fatigue_model="thresholded"):

    model_types = {"thresholded": Thresholded, "3cc": ThreeCompartment}
    if fatigue_model not in model_types:
        raise ValueError("fatigue_model must be 'thresholded' or '3cc'.")

    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])

    if task is None:
        #task = CircularPath1(
        #    r1=0.3,
        #    tz=0.2,
        #    theta=0.0,
        #    h=0.6,
        #    r2=0.1
        #)
        #task = CircularPath2(
        #    r1=0.3,
        #    tz=0.2,
        #    h=0.6,
        #    r2=0.1
        #)
        task = SphericalRegion(
            center=np.array([-0.22686198,  0.39980641,  0.28600239]),
            radius=0.1,
        )

    ocp = MusculoskeletalOCP.from_model(
        model=model,
        f_dynamics_func=None,
        i_dynamics_func=MODEL_CONFIG[model_name]["i_dynamics_func_path"],
        dataset=MODEL_CONFIG[model_name]["dataset_path"],
        task=task,
        horizon=horizon,
        timestep=timestep,
        muscles=muscles,
        force_index=force_index,
        fatigue_model_type=model_types[fatigue_model],
    )

    result = ocp.solve(
        initial_state=ocp.initial_state,
        task_parameters=task.initial_parameters,
        task_rates=task.initial_rates,
        diagnostics=True,
        validate=True,
    )
    ocp.save_result(model_name=model_name, tag=tag)

    if save_motion:
        visualizer = Visualizer(model)
        root = Path(__file__).parent.parent
        filename = f"{model.getName()}_{datetime.now():%Y%m%d_%H%M%S}.sto"
        motion_file = visualizer.save_motion_sto(
            root / "results" / f"{model.getName()}/motions/{filename}",
            result.time,
            result.states[:ocp.joint_states.size, :],
        )

    
    return result

def visualize_ocp_result(model_name: str, ocp_result_path=None, *, block=True):
    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])
    ocp_result = OCPResult.from_npz(ocp_result_path or MODEL_CONFIG[model_name]["ocp_result_path"])
    dataset = MSKdataset.from_npz(MODEL_CONFIG[model_name]["dataset_path"])

    visualizer = Visualizer(model, data=dataset)
    visualizer.visualize_ocp_result(
        ocp_result,
        slice_axis="y",
        slice_value=-0.25,
        reference_marker=MODEL_CONFIG[model_name]["reference_marker"],
        tolerance=0.05,
        opacity=0.9,
        radius=0.008,
        block=block,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimize a thresholded or three-compartment fatigue trajectory.")
    parser.add_argument("--model", default="tsm_3DOF", choices=MODEL_CONFIG)
    parser.add_argument("--fatigue-model", choices=("thresholded", "3cc"), default="3cc")
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--timestep", type=float, default=0.01)
    parser.add_argument("--force-index", type=int, default=0)
    parser.add_argument("--muscles", nargs="+", default=["DeltoideusScapula_M", "Infraspinatus_S"])
    parser.add_argument("--tag", help="Result filename stem; defaults to test_<fatigue-model>.")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--no-save-motion", action="store_true")
    args = parser.parse_args()
    tag = args.tag or f"test_{args.fatigue_model}"
    print(f"Fatigue model: {args.fatigue_model}; result tag: {tag}")
    solve_ocp(args.model, force_index=args.force_index, horizon=args.horizon,
              timestep=args.timestep, save_motion=not args.no_save_motion,
              tag=tag, muscles=args.muscles, fatigue_model=args.fatigue_model)
    result_path = Path(__file__).resolve().parent.parent / "results" / args.model / "ocp" / "results" / f"{tag}.npz"
    print(f"Saved result: {result_path}")
    if not args.no_show:
        visualize_ocp_result(args.model, ocp_result_path=result_path, block=True)


if __name__ == "__main__":
    main()
