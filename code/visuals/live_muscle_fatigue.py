"""Live muscle fatigue histories on a shared Matplotlib axis."""

from collections import deque
import math

import matplotlib.pyplot as plt


class LiveMuscleFatiguePlot:
    """Plot named muscle fatigue fractions as percentages over time."""

    def __init__(self, *, history_seconds=20.0, max_samples=2000):
        self.history_seconds = history_seconds
        self.history = deque(maxlen=max_samples)
        self.origin = None
        self.lines = {}
        self.figure, self.axis = plt.subplots(figsize=(10, 5))
        self.axis.set_title('Muscle fatigue')
        self.axis.set_xlabel('Elapsed time [s]')
        self.axis.set_ylabel('Fatigue [%]')
        self.axis.set_ylim(0.0, 100.0)
        self.axis.grid(alpha=0.25)
        self.figure.tight_layout()
        plt.show(block=False)

    def append(self, timestamp, fatigue):
        """Record a sample, discovering muscle names from incoming messages."""
        if not math.isfinite(timestamp):
            return
        if self.history and timestamp < self.history[-1][0]:
            self.history.clear()
            self.origin = None
        if self.origin is None:
            self.origin = timestamp
        for name in fatigue:
            if name not in self.lines:
                self.lines[name] = self.axis.plot([], [], label=name)[0]
                self.axis.legend()
        self.history.append((timestamp, dict(fatigue)))
        while self.history and self.history[0][0] < timestamp - self.history_seconds:
            self.history.popleft()

    def refresh(self):
        """Redraw and service GUI events even while no messages arrive."""
        if not plt.fignum_exists(self.figure.number):
            return
        if self.history:
            times = [timestamp - self.origin for timestamp, _ in self.history]
            for name, line in self.lines.items():
                line.set_data(times, [
                    100.0 * values.get(name, math.nan)
                    for _, values in self.history
                ])
            self.axis.set_xlim(max(0.0, times[-1] - self.history_seconds),
                               max(times[-1], 0.1))
            self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

    def close(self):
        plt.close(self.figure)
