"""Matched acceleration-controlled joint plant for MPC experiments."""

import numpy as np


class JointPlant:
    def __init__(self, positions, velocities):
        self.positions = np.asarray(positions, dtype=float).copy()
        self.velocities = np.asarray(velocities, dtype=float).copy()

    def step(self, acceleration, dt):
        acceleration = np.asarray(acceleration)
        self.positions += dt * self.velocities + 0.5 * dt**2 * acceleration
        self.velocities += dt * acceleration
        return self.positions.copy()
