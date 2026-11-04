
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
import tqdm
import numpy as np
import casadi as ca
from dataclasses import dataclass

from .msk_data import MSKdataset, ModelInfo, OCPResult, JointStates, \
        MODEL_CONFIG, Interpolants, MuscleInfo, ThresholdedParams, \
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
        self.valid_indices = np.flatnonzero(self.valid_mask)
        self._init_interpolant_grids(filter_success=True)

    @property
    def n_muscles(self) -> int:
        return self.muscles.size

    @property
    def valid_mask(self):
        samples, results = self.dataset.samples, self.dataset.results
        mask = (
            results.sample_success[:, self.force_index]
            & np.isfinite(samples.joint_coordinates).all(axis=1)
            & np.isfinite(results.sample_muscle_forces[:, self.force_index, :]).all(axis=1)
        ) 
        return mask

    @cached_property
    def rate_interpolant(self):
        return self.interpolants.interpolant("fatigue_rate")

    @cached_property
    def force_interpolant(self):
        return self.interpolants.interpolant("muscle_force")

    def _init_interpolant_grids(self, *, filter_success=False):
        indices = self.valid_indices if filter_success else slice(None)
        joint_coordinates = self.dataset.samples.joint_coordinates[indices]
        n_coordinates = joint_coordinates.shape[1]

        # extracts unique joint coordinate values for interpolation grid
        self.axes = [
            np.unique(joint_coordinates[:, i])
            for i in range(n_coordinates)
        ]
        self.axes_list = [axis.tolist() for axis in self.axes]
        self.interpolants.axes_list = self.axes_list
        self.interpolants.joint_coordinate_names = tuple(self.dataset.info.joint_coordinate_names)
        self.interpolants.muscle_names = self.muscles.names
        self.interpolants.force = self.dataset.samples.forces[self.force_index].copy()

        # the amount of grid points per joint coordinate axis
        grid_shape = tuple(len(axis) for axis in self.axes)          # shape = (n_q1, ...,  n_qN)
        if np.prod(grid_shape) != len(joint_coordinates):
            raise ValueError("Valid joint samples do not form a complete interpolation grid")

        muscle_forces = self.dataset.results.sample_muscle_forces[
            indices, self.force_index, :][:, self.muscles.indexes]

        def _build_reversed_grid(sample_values: np.ndarray):
            grid = sample_values.reshape(*grid_shape, self.n_muscles, order="C")       # shape (n_q1, ...,  n_qN, n_muscles)
            n_coordinates = len(grid_shape)
            grid_order = tuple(reversed(range(n_coordinates))) + (n_coordinates,)                       # shape (n_qN, ...,  n_q1, n_muscles)
            return grid.transpose(grid_order).ravel(order="C").tolist()
        
        self.interpolants.fatigue_rate_grid = _build_reversed_grid(self.rates_from_forces(muscle_forces))
        self.interpolants.muscle_force_grid = _build_reversed_grid(muscle_forces)

    @abstractmethod
    def rates_from_forces(self, forces):
        """Compute model-specific production coefficients from selected muscle forces."""

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

    @abstractmethod
    def fatigue(self, state):
        """Fatigued fraction of each muscle for the OCP objective."""

    @abstractmethod
    def _numerical_derivative(self, state, rates, forces):
        """Derivative at sampled rates and forces for trajectory validation."""

    def derivative_from_forces(self, state, forces):
        """Evaluate fatigue dynamics directly from selected muscle forces."""
        forces = np.asarray(forces)
        return self._numerical_derivative(state, self.rates_from_forces(forces), forces)

    def _nearest_neighbor_batch(self, q: np.ndarray):                                           # shape (N, n_coordinates)
        qs = self.dataset.samples.joint_coordinates[self.valid_indices]
        nn_indices = self.valid_indices[np.array([
            np.argmin(np.sum((qs - point) ** 2, axis=1)) for point in q
        ])]

        muscle_indexes = self.muscles.indexes

        forces = self.dataset.results.sample_muscle_forces[
            nn_indices, self.force_index, :][:, muscle_indexes]        # shape (N, n_muscles)
        
        return self.rates_from_forces(forces), forces

    def simulate_muscle_states(self, q: np.ndarray, m: np.ndarray, dt: float, N: int,
                               method="nearest", *, rates=None, forces=None):
        """Numerical trajectory simulation."""
        if (rates is None) != (forces is None):
            raise ValueError("Provide both rates and forces, or neither")
        if rates is not None:
            rates, forces = np.asarray(rates), np.asarray(forces)
        elif method == "nearest":
            rates, forces = self._nearest_neighbor_batch(q)         # shape (N, n_muscles)
        elif method == "interpolant":
            rates = np.vstack([np.asarray(self.rate_interpolant(point)).ravel() for point in q])
            forces = np.vstack([np.asarray(self.force_interpolant(point)).ravel() for point in q])
        else:
            raise ValueError(f"Unknown simulation method: {method}")

        for k in range(N - 1):
            previous = m[:, k]
            derivative = self._numerical_derivative(previous, rates[k], forces[k])

            m[:, k + 1] = np.clip(
                previous + dt * derivative,
                0.0,
                1.0,
            )           

        return m           # shape (n_states, N)


class Thresholded(FatigueModel):
    def __init__(self, dataset: MSKdataset, muscles: MuscleInfo, *, force_index: int = 0):
        self.params = ThresholdedParams.from_max_forces(muscles.max_forces)
        super().__init__(dataset, muscles, force_index=force_index)

    def rates_from_forces(self, forces):
        return np.asarray(forces) / self.params.capacity

    @property
    def n_states(self) -> int:
        return self.n_muscles

    def fatigue(self, state):
        return state
    
    def initial_state(self):
        return np.zeros((self.n_muscles,))

    def derivative(self, q: ca.MX, fatigue: ca.MX) -> ca.MX:
        rates = self.rate_interpolant(q)
        forces = self.force_interpolant(q)
        active = forces >= self.params.threshold

        return ca.if_else(
            active,
            (1 - fatigue) * rates,
            -fatigue * (self.params.recovery / self.params.capacity),
        )

    def _numerical_derivative(self, state, rates, forces):
        return np.where(
            forces >= self.params.threshold,
            (1 - state) * rates,
            -state * (self.params.recovery / self.params.capacity),
        )


class ThreeCompartment(FatigueModel):
    def __init__(self, dataset: MSKdataset, muscles: MuscleInfo, *, force_index: int = 0):
        self.params = ThreeCompartmentParams()
        super().__init__(dataset, muscles, force_index=force_index)

    def rates_from_forces(self, forces):
        return self.params.F * np.asarray(forces) / self.muscles.max_forces
    
    @property
    def n_states(self) -> int:
        return 2 * self.n_muscles

    def fatigue(self, state):
        return state[:self.n_muscles]

    def initial_state(self) -> np.ndarray:
        return np.zeros(self.n_states)  # [Mf, Ma]; Mr = 1 - Mf - Ma

    def split_state(self, state):
        """Return (active, resting, fatigued) from stored [fatigued, active]."""
        m = self.n_muscles
        mf = state[:m]
        ma = state[m:2 * m]
        mr = 1 - mf - ma
        return ma, mr, mf

    def derivative(self, q: ca.MX, m: ca.MX) -> ca.MX:
        forces = self.force_interpolant(q)
        ma, mr, mf = self.split_state(m)
        p = self.params

        # Normalized muscle-demand approximation.
        target_load = forces / self.muscles.max_forces

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
        dmf = fatigue_flow - recovery_flow

        return ca.vertcat(dmf, dma)

    def _numerical_derivative(self, state, rates, forces):
        ma, mr, mf = self.split_state(state)
        p = self.params
        target_load = forces / self.muscles.max_forces
        control = np.where(
            ma < target_load,
            p.LD * np.minimum(target_load - ma, mr),
            p.LR * (target_load - ma),
        )
        fatigue_flow = p.F * ma
        recovery_flow = p.R * mf
        return np.concatenate((fatigue_flow - recovery_flow,
                               control - fatigue_flow))


def compare_muscle_state_trajectories(model_name: str, force_index: int = 0, initial_fatigue=None):
    from .msk_data import OCPResult, FatigueValidationData, JointStates
    ocp_result = OCPResult.from_npz(MODEL_CONFIG[model_name]["ocp_result_path"])
    validation_data = FatigueValidationData.from_npz(MODEL_CONFIG[model_name]["ocp_validation_path"])

    import opensim as osim
    model = osim.Model(MODEL_CONFIG[model_name]["model_path"])
    joint_states = JointStates.from_model(model)
    muscles = MuscleInfo.from_model(model, muscles=ocp_result.muscle_names)
    muscle_state_count = ocp_result.states.shape[0] - joint_states.size
    if muscle_state_count == muscles.size:
        fatigue_model_type = Thresholded
    elif muscle_state_count in (2 * muscles.size, 3 * muscles.size):
        fatigue_model_type = ThreeCompartment
    else:
        raise ValueError("OCP muscle state count does not match saved muscle names")
    fatigue_model = fatigue_model_type(
        MSKdataset.from_npz(MODEL_CONFIG[model_name]["dataset_path"]),
        muscles,
        force_index=force_index,
    )

    if (validation_data.muscle_states is None
            or validation_data.muscle_states.shape[0] != fatigue_model.n_states):
        initial_states = ocp_result.states[joint_states.size:, :].copy()
        if isinstance(fatigue_model, ThreeCompartment) and muscle_state_count == 3 * muscles.size:
            initial_states = np.vstack((
                initial_states[2 * muscles.size:], initial_states[:muscles.size]))
        if initial_fatigue is not None:
            if isinstance(fatigue_model, ThreeCompartment):
                initial_states[:muscles.size, 0] = initial_fatigue
            else:
                initial_states[:, 0] = initial_fatigue
        validation_data.muscle_states = fatigue_model.simulate_muscle_states(
            q=ocp_result.states[ocp_result.pos_indices, :].T,
            m=initial_states,
            dt=ocp_result.time[1] - ocp_result.time[0],
            N=ocp_result.time.size,
            rates=fatigue_model.rates_from_forces(validation_data.muscle_forces[:, muscles.indexes]),
            forces=validation_data.muscle_forces[:, muscles.indexes],
        )
        validation_data.save_npz(MODEL_CONFIG[model_name]["ocp_validation_path"])

    from .visuals.visualizer import Visualizer
    visualizer = Visualizer(model=model, data=fatigue_model.dataset)
    return visualizer.compare_muscle_state_trajectories(
        ocp_result,
        validation_data,
    )
