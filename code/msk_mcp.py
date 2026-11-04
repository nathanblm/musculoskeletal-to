
import numpy as np
from time import perf_counter

class MusculoskeletalMPC():
    def __init__(self, ocp, plant, *, steps_per_update=1):
        if not 1 <= steps_per_update < ocp.conf.N - 1:
            raise ValueError("steps_per_update must satisfy 1 <= steps_per_update < N - 1")
        
        self.ocp = ocp
        self.plant = plant
        self.steps_per_update = steps_per_update

        self.task_parameters = ocp.task.initial_parameters.copy()
        self.task_rates = ocp.task.initial_rates.copy()

        self.result = None
        self.solve_time = 0.0
        self._executed_steps = 0

    @property
    def replanning_interval(self):
        return self.steps_per_update * self.ocp.conf.dt

    def _initial_quess(self):
        """Shift the previous prediction by the executed intervals."""
        if self.result is None:
            return None

        def shift(values):
            return np.concatenate(
                (
                    values[:, self._executed_steps:],
                    np.repeat(
                        values[:, -1:], 
                        self._executed_steps, 
                        axis=1
                    ),
                ),
                axis=1,
            )

        # shift previous solution by executed steps and repeat the last value for steps remaining in the horizon
        states = shift(self.result.states)
        # append "measured" state from the plant to the beginning of the state trajectory
        states[:, 0] = self.plant.state

        return {
            "states": states,
            "controls": shift(self.result.controls),
            "task_parameters": shift(self.ocp.task_parameters_solution),
            "task_rates": shift(self.ocp.task_param_rates_solution),
        }

    def step(self):
        """Solve the OCP and then apply the steps_per_update controls to the plant."""
        initial_guess = self._initial_quess()

        start = perf_counter()
        self.result = self.ocp.solve(
            initial_state=self.plant.state,
            task_parameters=self.task_parameters,
            task_rates=self.task_rates,
            initial_guess=initial_guess,
            diagnostics=False,
            validate=False,
        )
        self.solve_time = perf_counter() - start

        self._executed_steps = 0
        for k in range(self.steps_per_update):
            self.plant.step(self.result.controls[:, k], self.ocp.conf.dt)
            self._executed_steps += 1

            self.task_parameters = self.ocp.task_parameters_solution[:, k + 1].copy()
            self.task_rates = self.ocp.task_param_rates_solution[:, k + 1].copy()

        return self.plant.state
