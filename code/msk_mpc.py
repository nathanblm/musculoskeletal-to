"""Position-feedback receding-horizon control using MusculoskeletalOCP."""

from time import perf_counter

import numpy as np


class MusculoskeletalMPC:
    def __init__(self, ocp, initial_state=None, *, steps_per_update=1):
        if not 1 <= steps_per_update < ocp.conf.N - 1:
            raise ValueError("steps_per_update must leave a control interval in the horizon")
        self.ocp = ocp
        self.steps_per_update = steps_per_update
        self.sample_time = steps_per_update * ocp.conf.dt
        # The plant holds the returned acceleration for the full update interval.
        for k in range(1, steps_per_update):
            ocp.opti.subject_to(ocp.U[:, k] == ocp.U[:, 0])
        self.state = np.array(ocp.initial_state if initial_state is None else initial_state,
                              dtype=float, copy=True)
        self.n_joint = ocp.joint_states.size
        self.previous_positions = self.state[:self.n_joint:2].copy()
        self.previous_control = None
        self.task_parameters = ocp.task.initial_parameters.copy()
        self.task_rates = ocp.task.initial_rates.copy()
        self.result = None
        self.solve_time = 0.0
        self.prediction_revision = float("nan")

    def update_state(self, positions):
        positions = np.asarray(positions)
        if self.previous_control is not None:
            dt = self.sample_time
            # Endpoint velocity for the matched constant-acceleration plant.
            velocity = ((positions - self.previous_positions) / dt
                        + 0.5 * dt * self.previous_control)
            derivative = self.ocp.fatigue_model.derivative
            muscle_state = self.state[self.n_joint:]
            midpoint = (self.previous_positions + positions) / 2
            k1 = np.asarray(derivative(self.previous_positions, muscle_state)).ravel()
            k2 = np.asarray(derivative(midpoint, muscle_state + dt * k1 / 2)).ravel()
            k3 = np.asarray(derivative(midpoint, muscle_state + dt * k2 / 2)).ravel()
            k4 = np.asarray(derivative(positions, muscle_state + dt * k3)).ravel()
            self.state[self.n_joint:] = muscle_state + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            self.state[1:self.n_joint:2] = velocity
        self.state[:self.n_joint:2] = positions
        self.previous_positions = positions.copy()

    def warm_start(self):
        shift = self.steps_per_update

        def shifted(values):
            return np.column_stack((values[:, shift:],
                                    np.repeat(values[:, -1:], shift, axis=1)))

        state_guess = shifted(self.result.states)
        state_guess[:, 0] = self.state
        self.ocp.opti.set_initial(self.ocp.X, state_guess)
        self.ocp.opti.set_initial(self.ocp.U, shifted(self.result.controls))
        self.ocp.opti.set_initial(self.ocp.task_params,
                                 shifted(self.ocp.task_parameters_solution))
        self.ocp.opti.set_initial(self.ocp.task_param_rates,
                                 shifted(self.ocp.task_param_rates_solution))

    def step(self, positions):
        self.update_state(positions)
        previous_prediction = None
        if self.result is not None:
            previous_prediction = self.result.endpoint_positions[:, self.steps_per_update:].copy()
            self.warm_start()
        start = perf_counter()
        self.result = self.ocp.solve(
            self.state, self.task_parameters, self.task_rates,
            diagnostics=False, validate=False,
        )
        self.solve_time = perf_counter() - start
        shift = self.steps_per_update
        if previous_prediction is not None:
            difference = self.result.endpoint_positions[:, :-shift] - previous_prediction
            self.prediction_revision = float(np.sqrt(np.mean(np.sum(difference**2, axis=0))))
        self.task_parameters = self.ocp.task_parameters_solution[:, shift].copy()
        self.task_rates = self.ocp.task_param_rates_solution[:, shift].copy()
        self.previous_control = self.result.controls[:, 0].copy()
        return self.previous_control.copy()
