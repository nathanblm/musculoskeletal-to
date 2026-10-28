from dataclasses import dataclass
from datetime import datetime
from typing import Union
from pathlib import Path

import casadi as ca
import numpy as np

import opensim as osim
from .sk_kinematics import Kinematics
from .visualizer import Visualizer
from .msk_fatiguemodel import FatigueModel
from .msk_data import ModelInfo, OCPResult, MSKdataset, MODEL_CONFIG
from .references import Circular2DOF, Circular1DOF, CartesianPath, JointPath, FreeJointPath

from .msk_data import JointStates, MuscleStates

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

class MusculoskeletalOCP:
    def __init__(self, kinematics, f_dynamics, i_dynamics, fatigue, reference, joint_states, muscle_states, x, u, conf, N=None):
        self.kinematics = kinematics
        self.f_dynamics = f_dynamics
        self.i_dynamics = i_dynamics
        self.fatigue = fatigue
        self.reference = reference
        self.conf = conf
        self.ocp_result = OCPResult.empty()

        self.muscle_states = muscle_states
        self.joint_states = joint_states

        self.x = x                          # [q1, v1, q2, v2, ..., qn, vn, f1, f2, ..., fm]
        self.u = u                          # [a1, a2, ..., an]

        self.opti = ca.Opti()

        self.X = self.opti.variable(self.x.size, self.conf.N)
        self.U = self.opti.variable(self.u.size, self.conf.N - 1)

        
        self.path_parameters = self.opti.variable(
            len(self.reference.initial_parameters), self.conf.N
        )
        self.path_rates = self.opti.variable(
            len(self.reference.initial_rates), self.conf.N - 1
        )

        self.set_initial_trajectory()

    @classmethod
    def from_model(
        cls,
        model,
        f_dynamics_func: Path,
        i_dynamics_func: Path,
        dataset: Path,
        reference: Union[CartesianPath, JointPath],
        horizon,
        timestep,
        muscles: list[str] = None,
    ):
        kinematics = Kinematics(model)
        joint_states = JointStates.from_model(model)
        muscle_states = MuscleStates.from_model(model, muscles=muscles)
        f_dynamics = ca.Function.load(str(f_dynamics_func))
        i_dynamics = ca.external("F", str(i_dynamics_func))
        fatigue = FatigueModel.from_npz(dataset, muscle_states)
        conf = OCPconfig.from_model(model, horizon, timestep)

        x = np.concatenate((joint_states.x, muscle_states.fatigue))
        u = joint_states.torques

        return cls(kinematics, f_dynamics, i_dynamics, fatigue, reference, joint_states, muscle_states, x, u, conf)

    def solve(self):
        endpoint_FK = self.kinematics.endpoint_function
        n = self.joint_states.x.size
        m = self.muscle_states.fatigue.size
        position_indices = np.arange(0, n, 2)

        # initial muscle state
        self.opti.subject_to(self.X[n:, 0] == np.zeros(m))

        # joint limits
        for i, row in enumerate(position_indices):
            lower, upper = self.joint_states.angle_limits[i]
            self.opti.subject_to(
                self.opti.bounded(lower, self.X[row, :], upper)
            )

        # control limits
        for i in range(len(self.u)):
            lower = -self.conf.control_lim[i]
            upper = self.conf.control_lim[i]
            self.opti.subject_to(self.opti.bounded(lower, self.U[i, :], upper))

        self.reference.add_constraints(
            self.opti, self.path_parameters, self.path_rates, self.conf.dt
        )

        # cost function
        cost = 0
        for k in range(self.conf.N):
            qk = self.X[position_indices, k]
            endpoint = endpoint_FK(qk)
            path_point = self.reference.evaluate(self.path_parameters[:, k])
            error = endpoint - path_point
            cost += self.conf.w_tracking * ca.sumsqr(error)

            fatigue_mean = ca.sum1(self.X[n:, k]) / m
            cost += self.conf.w_fatigue * ca.sumsqr(fatigue_mean)

        for k in range(self.conf.N - 1):
            # dynamics rk4 integration
            xk = self.X[:n, k]
            uk = self.U[:, k]

            f = lambda state: self.f_dynamics(ca.vertcat(state, uk))
            k1 = f(xk)
            k2 = f(xk + self.conf.dt * k1 / 2)
            k3 = f(xk + self.conf.dt * k2 / 2)
            k4 = f(xk + self.conf.dt * k3)

            self.opti.subject_to(
                self.X[:n, k + 1]
                == xk + self.conf.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            )

            # fatigue dynamics
            qk = self.X[position_indices, k]
            mk = self.X[n:, k]

            df = self.fatigue.muscle_state_derivative(qk, mk, from_interpolants=True)
            self.opti.subject_to(self.X[n:, k + 1] == self.X[n:, k] + self.conf.dt * df)

        self.opti.minimize(cost)
        self.opti.solver("ipopt", self.conf.p_opts, self.conf.s_opts)
        solution = self.opti.solve()

        endpoint_positions = np.asarray(endpoint_FK(solution.value(self.X[position_indices, :])))
        path_values = np.atleast_2d(
            np.asarray(solution.value(self.path_parameters), dtype=float)
        )
        reference_positions = np.column_stack([
            np.asarray(self.reference.evaluate(path_values[:, k]), dtype=float).reshape(3)
            for k in range(self.conf.N)
        ])
    
        self.ocp_result = OCPResult(
            time=np.arange(self.conf.N) * self.conf.dt,
            states=solution.value(self.X),
            val_states=np.zeros_like(solution.value(self.X)),
            pos_indices=position_indices,
            controls=solution.value(self.U),
            endpoint_positions=endpoint_positions,
            reference=reference_positions,
            objective=solution.value(self.opti.f),
            muscle_names=self.muscle_states.names,
        )

        # forward simulation of muscle states using nearest neighbor evaluation of discrete rate and force maps
        # for comparison with the interpolated fatigue dynamics used in the OCP
        self.ocp_result.val_states[:n, :] = self.ocp_result.states[:n, :]
        self.ocp_result.val_states[n:, :] = self.fatigue.simulate_muscle_states(
            self.ocp_result.states[position_indices, :],
            self.ocp_result.val_states[n:, :],
            dt=self.conf.dt,
            N=self.conf.N,
        )

        return self.ocp_result

    def solve_joint_path(self):
        """ Optimize a joint-space trajectory with fatigue but without human dynamics. """
        if not isinstance(self.reference, JointPath):
            raise ValueError("Reference path must be a JointPath for this method")

        self.endpoint_FK = self.kinematics.endpoint_function

        n = self.joint_states.x.size
        m = self.muscle_states.fatigue.size
        position_indices = np.arange(0, n, 2)

        self.opti.subject_to(self.X[:n, 0] == np.zeros(n))
        self.opti.subject_to(self.U[:, 0] == np.zeros(self.u.size))

        # path in terms of joint angles so set joint limits as bounds on path parameters
        for i in range(self.path_parameters.shape[0]):
            lower, upper = self.joint_states.angle_limits[i]
            self.opti.subject_to(
                self.opti.bounded(lower, self.path_parameters[i, :], upper)
            )

        # initial muscle state
        self.opti.subject_to(self.X[n:, 0] == np.zeros(m))

        self.reference.add_constraints(
            self.opti, self.path_parameters, self.path_rates, self.conf.dt
        ) 
    
        # cost function
        cost = 0
        for k in range(self.conf.N):
            fatigue_mean = ca.sum1(self.X[n:, k]) / m
            cost += self.conf.w_fatigue * ca.sumsqr(fatigue_mean)

        for k in range(self.conf.N - 1):
            # fatigue dynamics
            qk = self.path_parameters[:, k]
            mk = self.X[n:, k]

            df = self.fatigue.muscle_state_derivative(qk, mk, from_interpolants=True)
            self.opti.subject_to(self.X[n:, k + 1] == self.X[n:, k] + self.conf.dt * df)

        self.opti.minimize(cost)
        self.opti.solver("ipopt", self.conf.p_opts, self.conf.s_opts)
        solution = self.opti.solve()


        path_values = np.atleast_2d(
            np.asarray(solution.value(self.path_parameters), dtype=float)
        )

        endpoint_positions = np.asarray(
            self.endpoint_FK(solution.value(self.X[position_indices, :]))
        )

        # convert joint-space path to Cartesian positions for visualization
        reference_positions = np.zeros((3, self.conf.N))
        for k in range(self.conf.N):
            reference_positions[:, k] = np.asarray(
                self.endpoint_FK(path_values[:, k])
            ).reshape(3)

        self.ocp_result = OCPResult(
            time=np.arange(self.conf.N) * self.conf.dt,
            states=solution.value(self.X),
            val_states=np.zeros_like(solution.value(self.X)),
            pos_indices=position_indices,
            controls=solution.value(self.U),
            endpoint_positions=endpoint_positions,
            reference=reference_positions,
            objective=solution.value(self.opti.f),
            muscle_names=self.muscle_states.names,
        )

        return self.ocp_result

    def set_initial_trajectory(self):

        if isinstance(self.reference, JointPath):
            return

        path_start = np.asanyarray(
            self.reference.evaluate(self.reference.initial_parameters)
        ).reshape((3,))
        positions, _ = self.kinematics.inverse_kinematics(path_start)                           # shape (n_coordinates,)
        velocities = np.zeros_like(positions)                                                   # shape (n_coordinates,)
        accelerations = np.zeros_like(positions)                                                # shape (n_coordinates,)
        muscle_state = np.zeros_like(self.muscle_states.fatigue)                                # shape (n_muscles,)
        joint_state = np.column_stack((positions, velocities)).ravel()                          # shape (2*n_coordinates,)
        torques = self.i_dynamics(
            np.concatenate((joint_state, accelerations)))[:len(positions)]                      # shape (n_coordinates,)

        init_state = np.concatenate((joint_state, muscle_state))                                # shape (2*n_coordinates + n_muscles,)
        init_state_trajectory = np.tile(init_state.reshape(-1, 1), (1, self.conf.N))            # shape (2*n_coordinates + n_muscles, N)
        init_control_trajectory = np.tile(
            np.asarray(torques).reshape(-1, 1),
            (1, self.conf.N - 1)
        )                                                                                       # shape (n_coordinates, N-1)

        self.opti.set_initial(self.X, init_state_trajectory)
        self.opti.set_initial(self.U, init_control_trajectory)
        self.opti.set_initial(
            self.path_parameters,
            np.tile(self.reference.initial_parameters.reshape(-1, 1), (1, self.conf.N)),
        )
        self.opti.set_initial(
            self.path_rates,
            np.tile(self.reference.initial_rates.reshape(-1, 1), (1, self.conf.N - 1)),
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

def solve_ocp(model_name: str, force_index: int = 0, reference=None, horizon=100, timestep=0.1, save_motion=False, tag=None, muscles: list[str] = None):

    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])

    if reference is None:
        reference = Circular1DOF(
            r1=0.3,
            tz=0.2,
            theta=0.0,
            h=0.6,
            r2=0.1
        )

    ocp = MusculoskeletalOCP.from_model(
        model=model,
        f_dynamics_func=MODEL_CONFIG[model_name]["f_dynamics_func_path"],
        i_dynamics_func=MODEL_CONFIG[model_name]["i_dynamics_func_path"],
        dataset=MODEL_CONFIG[model_name]["dataset_path"],
        reference=reference,
        horizon=horizon,
        timestep=timestep,
        muscles=muscles
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

def solve_joint_path(model_name: str, horizon=100, timestep=0.1, tag=None, muscles: list[str] = None):
    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])
    joint_states = JointStates.from_model(model)
    reference = FreeJointPath(joint_states.names)

    ocp = MusculoskeletalOCP.from_model(
        model=model,
        f_dynamics_func=MODEL_CONFIG[model_name]["f_dynamics_func_path"],
        i_dynamics_func=MODEL_CONFIG[model_name]["i_dynamics_func_path"],
        dataset=MODEL_CONFIG[model_name]["dataset_path"],
        reference=reference,
        horizon=horizon,
        timestep=timestep,
        muscles=muscles
    )

    result = ocp.solve_joint_path()
    ocp.save_result(model_name=model_name, tag=tag)

    return result

def visualize_ocp_result(model_name: str, ocp_result_path=None, *, reference=None, block=True):
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
        circular_path=reference,
        block=block,
    )


if __name__ == "__main__":

    model_name = "tsm_4DOF"





    #solve_ocp(model_name, horizon=300, timestep=0.01, save_motion=True, tag="test", muscles=["TeresMinor", "Supraspinatus_A"])  #"Supraspinatus_P"

    solve_joint_path(model_name, horizon=500, timestep=0.01, tag="joint_path", muscles=["TeresMinor", "Supraspinatus_A", "Supraspinatus_P"])  #
    path = Path("/home/nathan/Robotics/THESIS/Project/musculoskeletal-to/results/tsm_4DOF/ocp/results/joint_path.npz")
    visualize_ocp_result(model_name, block=True, ocp_result_path=path)
