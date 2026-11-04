"""Sequential RMR force estimation and fatigue integration for executed motion."""

import numpy as np

from rmr_solver import RMRsolver, ActSquared
from .msk_data import ModelInfo, RMRconfig


class RMRFatigueObserver:
    def __init__(self, solver, fatigue_model, initial_state, external_force):
        self.solver = solver
        self.fatigue_model = fatigue_model
        self.state = np.array(initial_state, dtype=float, copy=True)
        self.time = 0.0
        self.wrench = np.r_[external_force, np.zeros(3)]
        self.muscle_indices = np.array([
            tuple(solver.muscleNames).index(name) for name in fatigue_model.muscles.names
        ])

    @classmethod
    def from_model(cls, model, fatigue_model, initial_state):
        info = ModelInfo.from_model(model)
        solver = RMRsolver(model, visualize=False, constrainActDyn=True,
                           prescribedForceIndex=info.prescribed_force_indices)
        solver.setObjective(ActSquared(RMRconfig.from_model_info(info).weights))
        if tuple(solver.coordinateFreeNames) != info.joint_coordinate_names:
            raise ValueError("RMR and model coordinate orders differ")
        return cls(solver, fatigue_model, initial_state, fatigue_model.interpolants.force)

    def advance(self, positions, velocities, acceleration, dt):
        """Use start-of-interval kinematics; hold estimated forces over dt.

        Reuse this observer throughout the run so RMR retains activation history.
        Call once per plant substep, including when a control is held longer.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("RMR feedback timestep must be positive and finite")
        _, _, forces, info = self.solver.solve(
            time=self.time, position=positions, speed=velocities,
            acceleration=acceleration, values_prescribed_forces=self.wrench,
        )
        if not info["success"]:
            raise RuntimeError(f"RMR feedback failed at t={self.time:g}: {info.get('message', '')}")
        forces = np.asarray(forces)[self.muscle_indices]
        if not np.isfinite(forces).all():
            raise RuntimeError(f"Nonfinite RMR muscle forces at t={self.time:g}")
        rates = self.fatigue_model.rates_from_forces(forces)
        derivative = lambda state: self.fatigue_model._numerical_derivative(state, rates, forces)
        state = self.state
        k1 = derivative(state)
        k2 = derivative(state + dt * k1 / 2)
        k3 = derivative(state + dt * k2 / 2)
        k4 = derivative(state + dt * k3)
        updated = state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        if not np.isfinite(updated).all():
            raise RuntimeError(f"Nonfinite fatigue state at t={self.time:g}")
        self.state = updated
        self.time += dt
        return self.state.copy()
