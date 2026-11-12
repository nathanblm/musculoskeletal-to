"""Live joint velocities on a shared Matplotlib axis, independent of ROS."""

from collections import deque
import math

import matplotlib.pyplot as plt


class LiveJointVelocityPlot:
    """Keep a bounded, rolling history of named joint velocities."""

    def __init__(self, names, *, history_seconds=20.0, max_samples=2000):
        self.history_seconds = history_seconds
        self.history = deque(maxlen=max_samples)
        self.origin = None
        self.figure, self.axis = plt.subplots(figsize=(10, 5))
        self.lines = {
            name: self.axis.plot([], [], label=name)[0] for name in names
        }
        self.axis.set_title('Joint velocities')
        self.axis.set_xlabel('Elapsed time [s]')
        self.axis.set_ylabel('Velocity [rad/s or m/s]')
        self.axis.grid(alpha=0.25)
        if self.lines:
            self.axis.legend()
        self.figure.tight_layout()
        plt.show(block=False)

    def append(self, timestamp, velocities):
        """Record a sample; missing joints appear as gaps, not zero velocity."""
        if not math.isfinite(timestamp):
            return
        if self.history and timestamp < self.history[-1][0]:
            self.history.clear()
            self.origin = None
        if self.origin is None:
            self.origin = timestamp
        self.history.append((timestamp, dict(velocities)))
        while self.history and self.history[0][0] < timestamp - self.history_seconds:
            self.history.popleft()

    def refresh(self):
        """Redraw and service GUI events, including while no samples arrive."""
        if not plt.fignum_exists(self.figure.number):
            return
        if self.history:
            times = [timestamp - self.origin for timestamp, _ in self.history]
            for name, line in self.lines.items():
                line.set_data(times, [
                    values.get(name, math.nan) for _, values in self.history
                ])
            self.axis.set_xlim(max(0.0, times[-1] - self.history_seconds),
                               max(times[-1], 0.1))
            self.axis.relim()
            self.axis.autoscale_view(scalex=False)
            self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self):
        plt.close(self.figure)
