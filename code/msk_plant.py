from abc import ABC, abstractmethod
from typing import Union as U
from pathlib import Path
from kinematics import Kinematics
from msk_data import MODEL_CONFIG, JointStates, MSKdataset, MuscleInfo, ModelInfo, RMRconfig
from msk_fatiguemodel import ThreeCompartment, Thresholded
import casadi as ca
from msk_optimization import rk4_step
from rmr_solver import RMRsolver, ActSquared

import numpy as np

class Plant(ABC):
    def __init__(self, 
            f_dynamics,
            x_init: np.ndarray, 
            joint_states: JointStates, 
            fatigue_model: U[ThreeCompartment, Thresholded],
            rmr_solver,
        ):
        self.f_dynamics = f_dynamics
        self.joint_states = joint_states
        self.fatigue_model = fatigue_model
        self.rmr_solver = rmr_solver

        self.x = x_init
        self.time = 0.0

    @classmethod
    def from_model(
            cls, 
            model, 
            dataset: Path, 
            muscles: list[str], 
            fatigue_model_type: str, 
            *, 
            task=None, 
            initial_state=None,
            f_dynamics: Path = None,
        ):
        
        joint_states = JointStates.from_model(model)
        muscle_info = MuscleInfo.from_model(model, muscles=muscles)
        dataset = MSKdataset.from_npz(dataset)

        model_types = {"3cc": ThreeCompartment, "thresholded": Thresholded}
        if fatigue_model_type not in model_types:
            raise ValueError(f"Unknown fatigue model type: {fatigue_model_type}")

        fatigue_model = model_types[fatigue_model_type](dataset, muscle_info)

        from tasks import initial_state_from_task
        kinematics = Kinematics(model)

        if task is not None:
            init_state, _, _ = initial_state_from_task(task, kinematics, fatigue_model)
        elif initial_state is not None:
            init_state = initial_state
        else:
            raise ValueError("Either task or initial_state must be provided")

        if f_dynamics is None:
            from msk_optimization import SecondOrderIntegrator
            f_dynamics = SecondOrderIntegrator(joint_states.size // 2)
        else:
            f_dynamics = ca.Function.load(str(f_dynamics))

        model_info = ModelInfo.from_model(model)
        rmr_solver = RMRsolver(
            model=model,
            visualize=False,
            constrainActDyn=True,
            prescribedForceIndex=model_info.prescribed_force_indices,
        )
        config = RMRconfig.from_model_info(model_info)
        rmr_solver.setObjective(ActSquared(config.weights))

        return cls(f_dynamics, init_state, joint_states, fatigue_model, rmr_solver)

    @property
    def state(self):
        return self.x

    @abstractmethod
    def step(self, u: np.ndarray, dt: float):
        raise NotImplementedError

class O2Integrator(Plant):
    def step(self, u: np.ndarray, dt: float):
        n = self.joint_states.size
        u = np.asarray(u, dtype=float).reshape(-1)
        joints = self.x[:n]
        fatigue_model = self.fatigue_model

        endpoint_force = fatigue_model.dataset.samples.forces[fatigue_model.force_index]

        _, _, forces, info = self.rmr_solver.solve(
            time=self.time,
            position=joints[0::2],
            speed=joints[1::2],
            acceleration=np.asarray(
                self.f_dynamics(ca.vertcat(joints, u)), dtype=float,
            ).reshape(-1)[1::2],
            values_prescribed_forces=(
                np.r_[endpoint_force, np.zeros(3)]
            ),
        )

        if not info["success"]:
            raise RuntimeError(f"RMR solve failed: {info.get('message', '')}")

        forces = np.asarray(forces)[fatigue_model.muscles.indexes]

        f = lambda state: np.concatenate((
            np.asarray(
                self.f_dynamics(ca.vertcat(state[:n], u)), dtype=float,
            ).reshape(-1),
            fatigue_model.derivative_from_forces(state[n:], forces),
        ))
        self.x = rk4_step(f, self.x, dt)
        self.time += dt
        return self.state
