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

    def add_endpoint_constraints(self, opti, endpoint, reference_point):
        pass

    def tracking_cost(self, endpoint, reference_point):
        return ca.sumsqr(endpoint - reference_point)


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


class SphereReference(ReferencePath):
    def __init__(self, radius, initial_center, center_bounds=None, center_rate_bounds=None):
        if radius <= 0:
            raise ValueError("radius must be positive.")

        initial_center = np.asarray(initial_center, dtype=float).reshape(-1)
        if initial_center.size != 3:
            raise ValueError("initial_center must contain exactly three values.")

        if center_bounds is None:
            center_bounds = np.tile([-np.inf, np.inf], (3, 1))
        center_bounds = np.asarray(center_bounds, dtype=float)
        if center_bounds.shape != (3, 2):
            raise ValueError("center_bounds must have shape (3, 2).")
        if np.any(center_bounds[:, 0] > center_bounds[:, 1]):
            raise ValueError("Each center bound must be increasing.")
        if np.any(initial_center < center_bounds[:, 0]) or np.any(initial_center > center_bounds[:, 1]):
            raise ValueError("initial_center must lie within center_bounds.")

        if center_rate_bounds is None:
            center_rate_bounds = np.tile([-np.inf, np.inf], (3, 1))
        center_rate_bounds = np.asarray(center_rate_bounds, dtype=float)
        if center_rate_bounds.shape != (3, 2):
            raise ValueError("center_rate_bounds must have shape (3, 2).")
        if np.any(center_rate_bounds[:, 0] > center_rate_bounds[:, 1]):
            raise ValueError("Each center rate bound must be increasing.")

        self.radius = float(radius)
        self.center_bounds = center_bounds
        self.center_rate_bounds = center_rate_bounds
        self._initial_center = initial_center

    @property
    def parameter_bounds(self):
        return self.center_bounds

    @property
    def rate_bounds(self):
        return self.center_rate_bounds

    @property
    def initial_parameters(self):
        return self._initial_center

    @property
    def initial_rates(self):
        return np.zeros(3)

    def evaluate(self, parameters):
        return parameters

    def add_endpoint_constraints(self, opti, endpoint, reference_point):
        opti.subject_to(ca.sumsqr(endpoint - reference_point) <= self.radius ** 2)

    def tracking_cost(self, endpoint, reference_point):
        return 0.0

