import numpy as np
import opensim as osim

import rmr_solver

from .msk_data import MSKdataset, MuscleInfo, ModelInfo, RMRconfig
from .msk_fatiguemodel import Thresholded
from rmr_solver import ActSquared, RMRsolver


class MuscleStateEstimator:

    def __init__(self, model, dataset_path, *, muscles=None, force_index=0):
        self.dataset = MSKdataset.from_npz(dataset_path)
        self.muscles = MuscleInfo.from_model(model, muscles=muscles)
        self.fatigue_model = Thresholded(
            self.dataset,
            self.muscles,
            force_index=force_index,
        )
        self.muscle_state = self.fatigue_model.initial_state()
        self.previous_timestamp_ns = None

        model_info = ModelInfo.from_model(model)
        model = osim.Model(model)
        model.finalizeFromProperties()
        self.rmr_solver = RMRsolver(
            model=model,
            visualize=False,
            constrainActDyn=True,
            prescribedForceIndex=model_info.prescribed_force_indices,
        )
        config = RMRconfig.from_model_info(model_info)
        self.rmr_solver.setObjective(ActSquared(config.weights))
        self.endpoint_force = self.dataset.samples.forces[force_index]

    def update(self, positions, timestamp_ns):
        activation, _, forces, info = self.rmr_solver.solve(
            time=timestamp_ns * 1e-9,
            position=positions,
            speed=np.zeros_like(positions),
            acceleration=np.zeros_like(positions),
            values_prescribed_forces=np.r_[self.endpoint_force, np.zeros(3)],
        )
        if not info["success"]:
            raise RuntimeError(f"RMR solve failed: {info['message']}")

        forces = np.asarray(forces)[self.muscles.indexes]
        if self.previous_timestamp_ns is not None:
            dt = (timestamp_ns - self.previous_timestamp_ns) * 1e-9
            if dt <= 0.0:
                raise ValueError('Muscle-state timestamps must increase')
            derivative = self.fatigue_model.derivative_from_forces(
                self.muscle_state,
                forces,
            )
            self.muscle_state = np.clip(
                self.muscle_state + dt * derivative,
                0.0,
                1.0,
            )
        self.previous_timestamp_ns = timestamp_ns
        return activation[self.muscles.indexes], forces, self.muscle_state.copy()
