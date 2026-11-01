
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import tqdm
import numpy as np
import casadi as ca
from dataclasses import dataclass

from .msk_data import MSKdataset, ModelInfo, OCPResult, JointStates, \
        MuscleStates, MODEL_CONFIG, Interpolants, MuscleInfo, ThresholdedParams, \
        ThreeCompartmentParams  

from rmr_solver import RMRsolver, ActSquared
import os
import sys

os.environ.setdefault("SIMBODY_HOME", str(Path(sys.prefix) / "simbody"))

class FatigueModel(ABC):
    def __init__(self, dataset: MSKdataset, muscles: MuscleInfo, *, force_index: int = 0):
        self.dataset = dataset
        self.muscles = muscles
        self.force_index = force_index

        self.interpolants = Interpolants.empty()
        self.__init_interpolant_grids(force_index=force_index, filter_success=True)

    @property
    def n_muscles(self) -> int:
        return self.muscles.size

    @property
    def valid_mask(self):
        samples, results = self.dataset.samples, self.dataset.results
        mask = (
            results.sample_success[:, self.force_index]
            & np.isfinite(samples.joint_coordinates).all(axis=1)
            & np.isfinite(results.sample_fatigue_rates[:, self.force_index, :]).all(axis=1)
            & np.isfinite(results.sample_muscle_forces[:, self.force_index, :]).all(axis=1)
        ) 
        return mask

    @cached_property
    def rate_interpolant(self):
        return self.interpolants.interpolant("fatigue")

    @cached_property
    def force_interpolant(self):
        return self.interpolants.interpolant("muscle_force")

    def _init_interpolant_grids(self, *, force_index: int = 0, filter_success=False):
        joint_coordinates = self.data.samples.joint_coordinates[self.valid_mask if filter_success else slice(None)]
        n_coordinates = joint_coordinates.shape[1]

        # extracts unique joint coordinate values for interpolation grid
        self.axes = [
            np.unique(joint_coordinates[:, i])
            for i in range(n_coordinates)
        ]
        self.axes_list = [axis.tolist() for axis in self.axes]
        self.interpolants.axes_list = self.axes_list
        self.interpolants.joint_coordinate_names = tuple(self.data.info.joint_coordinate_names)
        self.interpolants.muscle_names = self.muscles.names
        self.interpolants.force = self.data.samples.forces[force_index].copy()

        # the amount of grid points per joint coordinate axis
        grid_shape = tuple(len(axis) for axis in self.axes)          # shape = (n_q1, ...,  n_qN)

        muscle_forces = self.data.results.sample_muscle_forces[
            :, force_index, :][
            :, self.muscle_states.indexes]                          # shape (n_samples, n_muscles)

        def _build_reversed_grid(sample_values: np.ndarray):
            grid = sample_values.reshape(*grid_shape, len(self.muscle_states.indexes), order="C")       # shape (n_q1, ...,  n_qN, n_muscles)
            n_coordinates = len(grid_shape)
            grid_order = tuple(reversed(range(n_coordinates))) + (n_coordinates,)                       # shape (n_qN, ...,  n_q1, n_muscles)
            return grid.transpose(grid_order).ravel(order="C").tolist()
        
        muscle_force_grid = _build_reversed_grid(muscle_forces)                                     # shape (n_qN, ...,  n_q1, n_muscles) 
        self.interpolants.muscle_force_grid = muscle_force_grid
        self.interpolants.force_rate_grid = muscle_force_grid / self.params.capacity                # shape (n_qN, ...,  n_q1, n_muscles)

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Total number of internal states."""

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        """Return a fully initialized vector of shape (n_states,)."""

    @abstractmethod
    def derivative(self, q, state):
        """Symbolic derivative used by the OCP."""

    def _nearest_neighbor_batch(self, q: np.ndarray):                                           # shape (N, n_coordinates)
        qs = self.data.samples.joint_coordinates[self.valid_indices]                            # shape (n_valid_samples, n_coordinates)
        diffs = qs[:, :, None] - q[None, :, :]                                                  # shape (n_valid_samples, n_coordinates, N)
        dists = np.linalg.norm(diffs, axis=1)                                                   # shape (n_valid_samples, N)
        nn_indices = self.valid_indices[np.argmin(dists, axis=0)]                               # shape (N,)

        muscle_indexes = self.muscle_states.indexes

        rates = self.data.results.sample_fatigue_rates[
            nn_indices, self.force_index, :][:, muscle_indexes]         # shape (N, n_muscles)
        forces = self.data.results.sample_muscle_forces[
            nn_indices, self.force_index, :][:, muscle_indexes]        # shape (N, n_muscles)
        
        return rates, forces

    def simulate_muscle_states(self, q: np.ndarray, m: np.ndarray, dt: float, N: int, method="nearest"):
        """Numerical trajectory simulation."""
        if method == "nearest":
            rates, forces = self._nearest_neighbor_batch(q)         # shape (N, n_muscles)
        if method == "interpolant":
            for k in range(N):
                rates[k] = self.rate_interpolant(q[k, :])
                forces[k] = self.force_interpolant(q[k, :])

        muscle_indexes = self.muscle_states.indexes
        params = self.data.fatigue_params
        recovery_rates = params.recovery / params.capacity

        for k in range(N - 1):
            previous = m[:, k]

            active = forces[k] >= params.threshold[muscle_indexes]

            derivative = np.where(
                active,
                (1.0 - previous) * rates[k],
                -previous * recovery_rates[muscle_indexes],
            )

            m[:, k + 1] = np.clip(
                previous + dt * derivative,
                0.0,
                1.0,
            )           

        return m           # shape (n_muscles, N)


class Thresholded(FatigueModel):
    def __init__(self, dataset: MSKdataset, muscles: MuscleInfo, *, force_index: int = 0):
        super().__init__(dataset, muscles, force_index=force_index)

        self.force_index = force_index
        self.params = ThresholdedParams.from_max_forces(muscles.max_forces)
    
    def initial_state(self):
        return np.zeros((self.n_muscles,))

    def derivative(self, q: ca.MX, fatigue: ca.MX) -> ca.MX:
        rates = self.rate_interpolant(q)
        forces = self.force_interpolant(q)
        active = forces >= self.thresholds

        return ca.if_else(
            active,
            (1 - fatigue) * rates,
            -fatigue * self.recovery_rates,
        )


class ThreeCompartment(FatigueModel):
    def __init__(self, dataset: MSKdataset, muscles: MuscleInfo, *, force_index: int = 0):
        super().__init__(dataset, muscles, force_index=force_index)

        self.force_index = force_index
        self.params = ThreeCompartmentParams
    
    @property
    def n_states(self) -> int:
        return 3 * self.n_muscles

    @property
    def fatigue(self, state):
        _, _, mf = self.split_state(state)
        return mf

    def initial_state(self) -> np.ndarray:
        m = self.n_muscles
        return np.concatenate((
            np.zeros(m),  # Ma
            np.ones(m),   # Mr
            np.zeros(m),  # Mf
        ))

    def split_state(self, state):
        m = self.n_muscles
        return (
            state[:m],
            state[m:2 * m],
            state[2 * m:3 * m],
        )

    def derivative(self, q: ca.MX, m: ca.MX) -> ca.MX:
        forces = self.force_interpolant(q)
        ma, mr, mf = self.split_state(m)
        p = self.params

        # Normalized muscle-demand approximation.
        target_load = forces / self.max_forces

        recruitment = p.LD * ca.fmin(target_load - ma, mr)
        relaxation = p.LR * (target_load - ma)

        control = ca.if_else(
            ma < target_load,
            recruitment,
            relaxation,
        )

        fatigue_flow = p.F * ma
        recovery_flow = p.R * mf

        dma = control - fatigue_flow
        dmr = -control + recovery_flow
        dmf = fatigue_flow - recovery_flow

        return ca.vertcat(dma, dmr, dmf)

class FatigueModel2:
    def __init__(self, dataset: MSKdataset, muscle_states: MuscleStates, *, force_index: int = 0):
        self.data = dataset
        self.muscle_states = muscle_states

        # currently only supports nearest-neighbor lookup for joint coordinates.
        # would provide a very rough estimate if endpoint force not exactly the same as the dataset force.
        self.force_index = force_index

        samples, results = dataset.samples, dataset.results
        valid = (
            results.sample_success[:, self.force_index]
            & np.isfinite(samples.joint_coordinates).all(axis=1)
            & np.isfinite(results.sample_fatigue_rates[:, self.force_index, :]).all(axis=1)
            & np.isfinite(results.sample_muscle_forces[:, self.force_index, :]).all(axis=1)
        )
        self.valid_indices = np.flatnonzero(valid)

        self.interpolants = Interpolants.empty()

        self._init_grids_for_casadi_interpolants(force_index=force_index, filter_success=True)

        self.interpolants.save_npz(MODEL_CONFIG["tsm_3DOF"]["interpolants_path"])

    @property
    def rate_interpolant(self):
        return self.interpolants.interpolant("fatigue")

    @property
    def mf_interpolant(self):
        return self.interpolants.interpolant("muscle_force")

    @classmethod
    def from_npz(cls, npz_path: Path, muscle_states: MuscleStates, *, force_index: int = 0):
        return cls(MSKdataset.from_npz(npz_path), muscle_states=muscle_states, force_index=force_index)

    def _init_grids_for_casadi_interpolants(self, *, force_index: int = 0, filter_success=False):
        indices = (
            self.valid_indices
            if filter_success
            else np.arange(len(self.data.samples.joint_coordinates))
        )

        joint_coordinates = self.data.samples.joint_coordinates[indices]
        n_coordinates = joint_coordinates.shape[1]

        # extracts unique joint coordinate values for interpolation grid
        self.axes = [
            np.unique(joint_coordinates[:, i])
            for i in range(n_coordinates)
        ]
        self.axes_list = [axis.tolist() for axis in self.axes]
        self.interpolants.axes_list = self.axes_list
        self.interpolants.joint_coordinate_names = tuple(self.data.info.joint_coordinate_names)
        self.interpolants.muscle_names = tuple(
            self.data.info.muscle_names[i] for i in self.muscle_states.indexes
        )
        self.interpolants.force = self.data.samples.forces[force_index].copy()

        # the amount of grid points per joint coordinate axis
        grid_shape = tuple(len(axis) for axis in self.axes)          # shape = (n_q1, ...,  n_qN)

        fatigue_rates = self.data.results.sample_fatigue_rates[
            :, force_index, :][
            :, self.muscle_states.indexes]                          # shape (n_samples, n_muscles)
        muscle_forces = self.data.results.sample_muscle_forces[
            :, force_index, :][
            :, self.muscle_states.indexes]                          # shape (n_samples, n_muscles)

        def _build_reversed_grid(sample_values: np.ndarray):
            grid = sample_values.reshape(*grid_shape, len(self.muscle_states.indexes), order="C")       # shape (n_q1, ...,  n_qN, n_muscles)
            n_coordinates = len(grid_shape)
            grid_order = tuple(reversed(range(n_coordinates))) + (n_coordinates,)                       # shape (n_qN, ...,  n_q1, n_muscles)
            return grid.transpose(grid_order).ravel(order="C").tolist()
        
        self.interpolants.fatigue_rate_grid = _build_reversed_grid(fatigue_rates)                                    # shape (n_qN, ...,  n_q1, n_muscles)
        self.interpolants.muscle_force_grid = _build_reversed_grid(muscle_forces)                                    # shape (n_qN, ...,  n_q1, n_muscles) 

    def muscle_state_derivative(self, qk: ca.MX, mk: ca.MX, from_interpolants: bool = True):
        if from_interpolants:
            return self.from_interpolants(qk, mk)
        # more implementations...

    def from_interpolants(self, qk: ca.MX, mk: ca.MX):  
        rates_k = self.rate_interpolant(qk)
        forces_k = self.mf_interpolant(qk)

        params = self.data.fatigue_params

        recovery_rates = params.recovery / params.capacity[self.muscle_states.indexes]
        thresholds = params.threshold[self.muscle_states.indexes]

        active = forces_k >= thresholds

        df = ca.if_else(
            active,
            (1.0 - mk) * rates_k,
            -mk * recovery_rates,
        )

        return df


    def simulate_muscle_states(self, X_joints, X_muscles, dt, N, rates=None, forces=None):
        if rates is None and forces is None:
            rates, forces = self.evaluate_nn_batch(X_joints)         # shape (N, n_muscles)

        muscle_indexes = self.muscle_states.indexes
        params = self.data.fatigue_params
        recovery_rates = params.recovery / params.capacity

        for k in range(N - 1):
            previous = X_muscles[:, k]

            active = forces[k] >= params.threshold[muscle_indexes]

            derivative = np.where(
                active,
                (1.0 - previous) * rates[k],
                -previous * recovery_rates[muscle_indexes],
            )

            X_muscles[:, k + 1] = np.clip(
                previous + dt * derivative,
                0.0,
                1.0,
            )           

        return X_muscles           # shape (n_muscles, N)


def compare_muscle_state_trajectories(model_name: str, force_index: int = 0, initial_fatigue=None):
    from .msk_data import OCPResult, FatigueValidationData, JointStates
    ocp_result = OCPResult.from_npz(MODEL_CONFIG[model_name]["ocp_result_path"])
    validation_data = FatigueValidationData.from_npz(MODEL_CONFIG[model_name]["ocp_validation_path"])

    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])
    joint_states = JointStates.from_model(model)
    muscle_states = MuscleStates.from_model(model, muscles=ocp_result.muscle_names)
    fatigue_model = FatigueModel.from_npz(
        MODEL_CONFIG[model_name]["dataset_path"], muscle_states, force_index=force_index
    )

    if validation_data.muscle_states is None:
        validation_data.muscle_states = fatigue_model.simulate_muscle_states(
            X_joints=ocp_result.states[ocp_result.pos_indices, :].T,
            X_muscles=ocp_result.states[joint_states.size:, :].copy(),
            dt=ocp_result.time[1] - ocp_result.time[0],
            N=ocp_result.time.size,
            rates=validation_data.fatigue_rates[:, muscle_states.indexes],
            forces=validation_data.muscle_forces[:, muscle_states.indexes],
        )
        validation_data.save_npz(MODEL_CONFIG[model_name]["ocp_validation_path"])

    from .visuals.visualizer import Visualizer
    visualizer = Visualizer(model=model, data=fatigue_model.data)
    return visualizer.compare_muscle_state_trajectories(
        ocp_result,
        validation_data,
    )


if __name__ == "__main__":
    model_name = "tsm_3DOF"

    from .msk_data import OCPResult, FatigueValidationData, JointStates
    ocp_result = OCPResult.from_npz(MODEL_CONFIG[model_name]["ocp_result_path"])

    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])
    joint_states = JointStates.from_model(model)
    muscle_states = MuscleStates.from_model(model, muscles=ocp_result.muscle_names)
    fatigue_model = FatigueModel.from_npz(
        MODEL_CONFIG[model_name]["dataset_path"], muscle_states, force_index=0
    )
