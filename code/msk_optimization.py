from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import casadi as ca
import numpy as np

import opensim as osim
from .sk_kinematics import Kinematics
from .visualizer import Visualizer
from .msk_fatiguemodel import FatigueModel
from .msk_data import ModelInfo, OCPResult
from .references import CircularPath

from .msk_data import JointStates, MuscleStates

@dataclass
class OCPconfig:
    N: int
    dt: float
    control_lim: np.ndarray
    thetadot_bounds: tuple  # minimum and maximum speed for traversing the circular path

    w_control: np.ndarray
    w_tracking: np.ndarray

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
    def from_model(cls, model, horizon, timestep, *, control_lim=None, thetadot_bounds=None, w_control=None, w_tracking=None):
        info = ModelInfo.from_model(model)

        if control_lim is None:
            control_lim = np.full(len(info.joint_coordinate_names), 100)
        if thetadot_bounds is None:
            thetadot_bounds = (0.01, 5.0)
        if w_control is None:
            w_control = np.full(len(info.joint_coordinate_names), 1e-3)
        if w_tracking is None:
            w_tracking = np.full(3, 1e4)

        return cls(
            N=horizon,
            dt=timestep,
            control_lim=control_lim,
            thetadot_bounds=thetadot_bounds,
            w_control=w_control,
            w_tracking=w_tracking
        )

class MusculoskeletalOCP:
    def __init__(self, kinematics, dynamics, fatigue, reference, joint_states, muscle_states, x, u, conf, N=None, x_init=None):
        self.kinematics = kinematics
        self.dynamics = dynamics
        self.fatigue = fatigue
        self.reference = reference
        self.conf = conf
        self.ocp_result = OCPResult.empty()

        self.muscle_states = muscle_states
        self.joint_states = joint_states

        self.x = x                          # [q1, v1, q2, v2, ..., qn, vn, f1, f2, ..., fm]
        self.u = u                          # [a1, a2, ..., an]
        self.x_init = x_init

        self.opti = ca.Opti()

        self.X = self.opti.variable(self.x.size, self.conf.N)
        self.U = self.opti.variable(self.u.size, self.conf.N - 1)

        self.theta = self.opti.variable(1, self.conf.N)             # progress state for circular path
        self.thetadot = self.opti.variable(1, self.conf.N - 1)      # progress state derivative

    @classmethod
    def from_model(
        cls,
        model,
        dynamics_func: Path,
        fatigue: FatigueModel,
        reference: CircularPath,
        horizon,
        timestep,
        x_init=None,
    ):
        kinematics = Kinematics(model)
        joint_states = JointStates.from_model(model)
        muscle_states = MuscleStates.from_model(model)
        dynamics_func = Path(dynamics_func)
        dynamics = ca.Function.load(str(dynamics_func))
        conf = OCPconfig.from_model(model, horizon, timestep)

        x = np.concatenate((joint_states.x, muscle_states.fatigue))
        u = joint_states.accelerations

        if x_init is None:
            path_start = np.asanyarray(reference.evaluate(0.0)).reshape((3,))
            positions, _ = kinematics.inverse_kinematics(path_start)
            velocities = np.zeros_like(positions)
            muscle_state = np.zeros_like(muscle_states.fatigue)
            joint_state = np.column_stack((positions, velocities)).ravel()
            x_init = np.concatenate((joint_state, muscle_state))
            print(f"Initial state not provided. Using default: {x_init}")

        return cls(kinematics, dynamics, fatigue, reference, joint_states, muscle_states, x, u, conf, x_init=x_init)

    def solve(self):
        endpoint_FK = self.kinematics.endpoint_function
        n = self.joint_states.x.size
        m = self.muscle_states.fatigue.size
        position_indices = np.arange(0, n, 2)

        # intial state constraint
        self.opti.subject_to(self.X[:, 0] == self.x_init)
        self.opti.subject_to(self.theta[0, 0] == 0.0)

        # joint limits
        for i, row in enumerate(position_indices):
            lower, upper = self.joint_states.position_limits[i]
            self.opti.subject_to(
                self.opti.bounded(lower, self.X[row, :], upper)
            )

        # control limits
        for i in range(len(self.u)):
            lower = -self.conf.control_lim[i]
            upper = self.conf.control_lim[i]
            self.opti.subject_to(self.opti.bounded(lower, self.U[i, :], upper))

        # progress velocity limits
        self.opti.subject_to(self.opti.bounded(self.conf.thetadot_bounds[0], self.thetadot, self.conf.thetadot_bounds[1]))

        # cost function
        cost = 0
        for k in range(self.conf.N):
            endpoint = endpoint_FK(self.X[position_indices, k])
            path_point = self.reference.evaluate(self.theta[0, k])
            error = endpoint - path_point
            cost += ca.sumsqr(error)


        for k in range(self.conf.N - 1):
            # dynamics rk4 integration
            xk = self.X[:n, k]
            uk = self.U[:, k]

            f = lambda state: self.dynamics(ca.vertcat(state, uk))
            k1 = f(xk)
            k2 = f(xk + self.conf.dt * k1 / 2)
            k3 = f(xk + self.conf.dt * k2 / 2)
            k4 = f(xk + self.conf.dt * k3)

            self.opti.subject_to(
                self.X[:n, k + 1]
                == xk + self.conf.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            )

            # path progression
            self.opti.subject_to(
                self.theta[0, k + 1] == self.theta[0, k] + self.conf.dt * self.thetadot[0, k]
            )

            # keep fatigue states constant as not yet implemented
            self.opti.subject_to(self.X[n:, k + 1] == self.X[n:, k])

        # TODO: fatigue dynamics

        self.opti.minimize(cost)
        self.opti.solver("ipopt", self.conf.p_opts, self.conf.s_opts)
        solution = self.opti.solve()

        endpoint_positions = np.asarray(endpoint_FK(solution.value(self.X[position_indices, :])))
        theta_values = np.asarray(solution.value(self.theta), dtype=float).reshape(-1)
        reference_positions = np.column_stack([
            np.asarray(self.reference.evaluate(float(theta)), dtype=float).reshape(3)
            for theta in theta_values
        ])
    
        self.ocp_result = OCPResult(
            time=np.arange(self.conf.N) * self.conf.dt,
            states=solution.value(self.X),
            pos_indices=position_indices,
            controls=solution.value(self.U),
            endpoint_positions=endpoint_positions,
            reference=reference_positions,
            objective=solution.value(self.opti.f)
        )

        # simulate muscle states because not yet implemented in the OCP
        self.ocp_result.states[n:, :] = self.fatigue.simulate_muscle_states(
            self.ocp_result.states[position_indices, :],
            self.ocp_result.states[n:, :],
            dt=self.conf.dt,
            N=self.conf.N,
        )

        return self.ocp_result

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


MODEL_CONFIG = {
    "arm26_3DOF": {
        "model_path": "musculoskeletal_models/arm26_3DOF/arm26_3DOF.osim",
        "dataset_npz_path": "results/arm26_3DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "dynamics_func_path": "musculoskeletal_models/arm26_3DOF/fd_dynamics/F.casadi",
        "ocp_result_npz_path": "results/arm26_3DOF/ocp/results/20261017_140446.npz",
        "reference_marker": "r_acromion",
    },
    "arm26_4DOF": {
        "model_path": "musculoskeletal_models/arm26_4DOF/arm26_4DOF.osim",
        "dataset_npz_path": "results/arm26_4DOF/datasets/msk_dataset_p1e5_m1_d1_fx-20_fy0_fz0.npz",
        "dynamics_func_path": "musculoskeletal_models/arm26_4DOF/fd_dynamics/F.casadi",
        "ocp_result_npz_path": "results/arm26_4DOF/ocp/results/20261017_112917.npz",
        "reference_marker": "r_acromion",

    },
    "tsm_4DOF": {
        "model_path": "musculoskeletal_models/tsm_4DOF/tsm_4DOF.osim",
        "dataset_npz_path": "results/tsm_4DOF/datasets/msk_dataset_p1e5_m1_d1_fx-26.8905_fy-7.10053_fz-10.2189.npz",
        "dynamics_func_path": "musculoskeletal_models/tsm_4DOF/fd_dynamics/F.casadi",
        "reference_marker": "ac",
    },
    "tsm_3DOF": {
        "model_path": "musculoskeletal_models/tsm_3DOF/tsm_3DOF.osim",
        "dataset_npz_path": "results/tsm_3DOF/datasets/msk_dataset_p1e4_m1_d1_fx-20_fy0_fz0.npz",
        "dynamics_func_path": "musculoskeletal_models/tsm_3DOF/fd_dynamics/F.casadi",
        "ocp_result_npz_path": "results/tsm_3DOF/ocp/results/20261021_162148.npz",
        "reference_marker": "ac",
    },
}


def solve_ocp(model_name: str, force_index: int = 0, reference=None, horizon=100, timestep=0.1, save_motion=False, tag=None):
    fatigue_model = FatigueModel.from_npz(MODEL_CONFIG[model_name]["dataset_npz_path"], force_index=force_index)

    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])

    if reference is None:
        reference = CircularPath(
            r1=0.16,
            tz=0.2,
            theta1=0.0,
            h=0.6,
            r2=0.1
        )

    ocp = MusculoskeletalOCP.from_model(
        model=model,
        dynamics_func=MODEL_CONFIG[model_name]["dynamics_func_path"],
        fatigue=fatigue_model,
        reference=reference,
        horizon=horizon,
        timestep=timestep,
    )

    result = ocp.solve()
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

def visualize_ocp_result(model_name: str, ocp_result_path: str, *, reference=None, block=True):
    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])
    ocp_result = OCPResult.from_npz(ocp_result_path)

    fatigue_model = FatigueModel.from_npz(MODEL_CONFIG[model_name]["dataset_npz_path"])
    visualizer = Visualizer(model, data=fatigue_model.data)
    visualizer.visualize_ocp_result(
        ocp_result,
        slice_axis="y",
        slice_value=-0.25,
        reference_marker=MODEL_CONFIG[model_name]["reference_marker"],
        tolerance=0.05,
        opacity=0.9,
        radius=0.008,
        circular_path=reference,
        block=block,
    )


if __name__ == "__main__":

    model_name = "tsm_3DOF"

    solve_ocp(model_name, force_index=0, horizon=100, timestep=0.01, save_motion=True, tag="test")

    #ocp_result_path = MODEL_CONFIG[model_name]["ocp_result_npz_path"]
    #visualize_ocp_result(model_name, ocp_result_path)
