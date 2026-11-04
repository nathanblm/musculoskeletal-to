"""Live muscle-state history and the latest MPC prediction."""

import matplotlib.pyplot as plt

from ..msk_fatiguemodel import ThreeCompartment


class LiveFatiguePlot:
    def __init__(self, fatigue_model):
        self.fatigue_model = fatigue_model
        self.compartments = isinstance(fatigue_model, ThreeCompartment)
        labels = ("Fatigued fraction", "Active fraction", "Resting fraction") if self.compartments else ("Fatigue",)
        self.figure, axes = plt.subplots(len(labels), 1, sharex=True, squeeze=False,
                                         figsize=(9, 3 * len(labels)))
        self.axes = axes[:, 0]
        self.history_lines, self.prediction_lines = [], []
        for axis, label in zip(self.axes, labels):
            history, prediction = [], []
            for name in fatigue_model.muscles.names:
                line, = axis.plot([], [], label=name)
                future, = axis.plot([], [], "--", color=line.get_color(), alpha=0.7)
                history.append(line)
                prediction.append(future)
            self.history_lines.append(history)
            self.prediction_lines.append(prediction)
            axis.set_ylabel(label)
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.25)
        self.axes[0].legend()
        self.axes[-1].set_xlabel("Simulation time [s]")
        self.figure.suptitle("Muscle states | solid: plant | dashed: MPC prediction")
        self.figure.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)

    def state_groups(self, states):
        if self.compartments:
            active, resting, fatigued = self.fatigue_model.split_state(states)
            return fatigued, active, resting
        return (self.fatigue_model.fatigue(states),)

    def update(self, time, states, prediction_time, predicted_states):
        if not plt.fignum_exists(self.figure.number):
            return
        for axis, history, prediction, values, future in zip(
                self.axes, self.history_lines, self.prediction_lines,
                self.state_groups(states), self.state_groups(predicted_states)):
            for line, predicted, row, future_row in zip(history, prediction, values, future):
                line.set_data(time, row)
                predicted.set_data(prediction_time, future_row)
            axis.set_xlim(float(time[0]), max(float(time[-1]), float(prediction_time[-1]), float(time[0]) + 0.1))
        # Fatigue can be much smaller than the active/resting fractions.
        self.axes[0].relim()
        self.axes[0].autoscale(enable=True, axis="y")
        self.axes[0].set_ylim(0, max(0.01, self.axes[0].get_ylim()[1]))
        self.figure.canvas.draw_idle()
        plt.pause(0.001)

    def close(self):
        plt.close(self.figure)
