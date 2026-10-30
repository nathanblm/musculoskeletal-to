from abc import ABC, abstractmethod

import casadi as ca
import numpy as np

def _y_rotation(theta):
    return ca.vertcat(
        ca.horzcat(ca.cos(theta), 0.0, ca.sin(theta)),
        ca.horzcat(0.0, 1.0, 0.0),
        ca.horzcat(-ca.sin(theta), 0.0, ca.cos(theta)),
    )


class Task(ABC):
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

    def add_boundary_constraints(self, opti, parameters, rates, **kwargs):

        # position limits
        lower, upper = np.asarray(self.parameter_bounds).T
        opti.subject_to(opti.bounded(lower, parameters, upper))
        opti.subject_to(parameters[:, 0] == self.initial_parameters)

        # velocity limits
        lower, upper = np.asarray(self.rate_bounds).T
        opti.subject_to(opti.bounded(lower, rates, upper))
        opti.subject_to(rates[:, 0] == self.initial_rates)

        self._additional_boundary_constraints(opti, parameters, rates, **kwargs)

    def _additional_boundary_constraints(self, opti, parameters, rates, **kwargs):
        pass

    def add_path_constraints(self, opti, parameters, rates, k, dt, **kwargs):
        # this function will be iterated over the horizon inside the ocp

        opti.subject_to(parameters[:, k + 1] == parameters[:, k] + rates[:, k] * dt)

        # additional constraints
        self._additional_path_constraints(opti, parameters, rates, k, dt, **kwargs)

    def _additional_path_constraints(self, opti, parameters, rates, k, dt, **kwargs):
        pass


class SphericalRegion(Task):
    """Optimize a translating sphere containing the endpoint.

    Task parameters are the center's Cartesian coordinates in meters; their
    rates are Cartesian velocities, limited to +/-0.3 m/s on each axis.
    ``center`` specifies the initial center and ``radius`` stays fixed.
    """
    def __init__(self, center, radius):
        self.center = np.asarray(center, dtype=float).reshape(3)
        self.radius = radius

    @property
    def parameter_bounds(self):
        return np.array([[-np.inf, np.inf]] * 3)

    @property
    def rate_bounds(self):
        return np.array([[-0.3, 0.3]] * 3)

    @property
    def initial_parameters(self):
        return self.center.copy()

    @property
    def initial_rates(self):
        return np.zeros(3)

    def evaluate(self, parameters: ca.MX):
        return ca.vertcat(parameters[0], parameters[1], parameters[2])

    def _additional_boundary_constraints(self, opti, parameters, rates, **kwargs):
        # terminal constraint
        endpoint = kwargs.get("endpoint")
        task_point = self.evaluate(parameters[:, -1])

        opti.subject_to(ca.sumsqr(endpoint - task_point) <= self.radius**2)


    def _additional_path_constraints(self, opti, parameters, rates, k, dt, **kwargs):
        endpoint = kwargs.get("endpoint")
        task_point = self.evaluate(parameters[:, k])

        # ensure that the endpoint remains within the sphere
        opti.subject_to(
            ca.sumsqr(endpoint - task_point) <= self.radius**2
        )



class CircularPath2(Task):
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
        return _y_rotation(parameters[0]).T @ p2_plane + p1_ground


class CircularPath1(Task):
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
        return _y_rotation(self.theta).T @ p2_plane + p1_ground
