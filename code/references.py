from abc import ABC, abstractmethod

import casadi as ca
import numpy as np

def _y_rotation(theta):
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), 0.0, ca.sin(theta)),
        ca.horzcat(0.0, 1.0, 0.0),
        ca.horzcat(-ca.sin(theta), 0.0, ca.cos(theta)),
    )


class ReferencePath(ABC):
    @property
    @abstractmethod
    def parameter_bounds(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def rate_bounds(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_parameters(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def initial_rates(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, parameters):
        """ return the cartesian position on path for given input parameters """
        raise NotImplementedError

    def add_constraints(self, opti, parameters, rates, timestep):
        """Add standard bounds and first-order path dynamics."""
        lower, upper = np.asarray(self.parameter_bounds).T
        rate_lower, rate_upper = np.asarray(self.rate_bounds).T
        opti.subject_to(opti.bounded(lower, parameters, upper))
        opti.subject_to(opti.bounded(rate_lower, rates, rate_upper))
        opti.subject_to(parameters[:, 0] == self.initial_parameters)
        for index in range(parameters.shape[1] - 1):
            opti.subject_to(
                parameters[:, index + 1]
                == parameters[:, index] + timestep * rates[:, index]
            )


class Circular2DOF(ReferencePath):
    def __init__(self, r1, tz, h, r2):
        self.r1 = r1
        self.tz = tz
        self.h = h
        self.r2 = r2

    @property
    def parameter_bounds(self):
        return np.array([[0.0, 0.5 * np.pi], [-np.inf, np.inf]])

    @property
    def rate_bounds(self):
        return np.array([[-0.3, 0.3], [0.01, 5.0]])

    @property
    def initial_parameters(self):
        return np.array([0.0, 0.0])

    @property
    def initial_rates(self):
        return np.array([0.0, 0.01])

    def evaluate(self, parameters: ca.MX):
        p1_ground = ca.vertcat(
            self.r1 * ca.cos(parameters[0]),
            self.h,
            self.tz + self.r1 * ca.sin(parameters[0]),
        )

        p2_plane = ca.vertcat(
            0.0,
            -self.r2 * ca.sin(parameters[1]),
            self.r2 * ca.cos(parameters[1]),
        )


        p2_in_ground = _y_rotation(parameters[0]).T @ p2_plane + p1_ground
        return p2_in_ground


class Circular1DOF(ReferencePath):
    def __init__(self, r1, tz, theta, h, r2):
        self.r = r1
        self.tz = tz
        self.theta = theta
        self.h = h
        self.r2 = r2

    @property
    def parameter_bounds(self):
        return np.array([[-np.inf, np.inf]])

    @property
    def rate_bounds(self):
        return np.array([[0.01, 5.0]])

    @property
    def initial_parameters(self):
        return np.array([0.0])

    @property
    def initial_rates(self):
        return np.array([0.01])

    def evaluate(self, parameters: ca.SX):
        p1_ground = ca.vertcat(
            self.r * ca.cos(self.theta),
            self.h,
            self.tz + self.r * ca.sin(self.theta),
        )

        p2_plane = ca.vertcat(
            0.0,
            -self.r2 * ca.sin(parameters[0]),
            self.r2 * ca.cos(parameters[0]),
        )

        p2_in_ground = _y_rotation(self.theta).T @ p2_plane + p1_ground
        return p2_in_ground

