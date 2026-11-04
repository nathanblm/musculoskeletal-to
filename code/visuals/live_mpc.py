"""Live Simbody motion and Matplotlib fatigue history with MPC predictions."""

from collections import deque

import numpy as np

from .simbody_renderer import SimbodyRenderer
from .live_fatigue import LiveFatiguePlot


class LiveMPCVisualizer(SimbodyRenderer):
    def __init__(self, model, controller, *, history_seconds=20.0, show_fatigue=True):
        super().__init__(model)
        self.controller = controller
        self.coordinates = [
            self.model.getCoordinateSet().get(name)
            for name in controller.plant.joint_states.names
        ]
        self.endpoint_trail = deque(
            maxlen=max(2, round(history_seconds / controller.replanning_interval) + 1)
        )
        self.prediction_line_ids = []
        self.prediction_marker_ids = []
        self.fatigue_plot = LiveFatiguePlot(controller.plant.fatigue_model) if show_fatigue else None
        self.fatigue_history = deque(maxlen=self.endpoint_trail.maxlen)

    def update(self):
        controller = self.controller
        plant = controller.plant
        self.state.setTime(float(plant.time))
        for i, coordinate in enumerate(self.coordinates):
            coordinate.setValue(self.state, float(plant.state[2 * i]), False)
            coordinate.setSpeedValue(self.state, float(plant.state[2 * i + 1]))
        self.model.assemble(self.state)
        self.model.realizePosition(self.state)
        self.endpoint_trail.append(self.marker_position(self.endpoint_marker))

        result = controller.result
        prediction = np.empty((0, 3)) if result is None else result.endpoint_positions.T
        reference = None if result is None else result.task.T
        self._set_path(prediction, self.prediction_line_ids, (0.95, 0.65, 0.1), 3)
        endpoints = () if len(prediction) == 0 else (prediction[0], prediction[-1])
        self._set_spheres(
            endpoints, ((0.95, 0.65, 0.1),) * 2, self.prediction_marker_ids, 0.008,
        )
        caption = (
            f"MPC | t={plant.time:.2f} s | solve={controller.solve_time:.3f} s"
            " | red: executed | orange: prediction | grey: task"
        )
        fatigue_model = plant.fatigue_model
        self.show(
            np.empty((0, 3)), np.empty((0, 3)), caption,
            force=fatigue_model.dataset.samples.forces[fatigue_model.force_index],
            trajectory=np.asarray(self.endpoint_trail), reference=reference,
        )

        if self.fatigue_plot is not None:
            n = plant.joint_states.size
            self.fatigue_history.append((plant.time, plant.state[n:].copy()))
            time, states = zip(*self.fatigue_history)
            if result is None:
                prediction_time = np.array([plant.time])
                predicted_states = plant.state[n:, None]
            else:
                # The prediction starts before the controls just applied to the plant.
                prediction_start = plant.time - controller._executed_steps * controller.ocp.conf.dt
                prediction_time = prediction_start + result.time
                predicted_states = result.states[n:]
            self.fatigue_plot.update(
                np.asarray(time), np.column_stack(states), prediction_time, predicted_states,
            )

    def close(self):
        try:
            self.visualizer.shutdown()
        finally:
            if self.fatigue_plot is not None:
                self.fatigue_plot.close()
