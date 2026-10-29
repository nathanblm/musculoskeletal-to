"""Separate joint-space surface plots for saved muscle interpolants."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np

from ..msk_data import Interpolants


class _InterpolantSlice:
    """Sampling and independent slice controls for one muscle surface."""

    def __init__(self, data, muscle, field, coordinate_axes, configuration, interpolant):
        import casadi as ca

        self.data = data
        self.names = self.data.joint_coordinate_names
        self.muscle_index = (self.data.muscle_names.index(muscle)
                             if isinstance(muscle, str) else muscle)
        self.x_index, self.y_index = [
            self.names.index(item) if isinstance(item, str) else item
            for item in coordinate_axes
        ]
        self.grids = [np.asarray(axis) for axis in self.data.axes_list]
        self.position = np.array(
            [(axis[0] + axis[-1]) / 2 for axis in self.grids]
            if configuration is None else configuration, dtype=float,
        )
        self.fixed_indices = [i for i in range(len(self.grids))
                              if i not in (self.x_index, self.y_index)]
        self.label, self.unit = {"fatigue": ("Fatigue rate", "s⁻¹"),
                                "muscle_force": ("Muscle force", "N")}[field]
        q = ca.MX.sym("q", len(self.grids))
        value = interpolant(q)[self.muscle_index]
        self.evaluate = ca.Function("visualize_slice", [q], [value])
        self._mapped = {}

    def sample(self, x, y):
        points = np.repeat(self.position[:, None], x.size, axis=1)
        points[self.x_index], points[self.y_index] = x.ravel(), y.ravel()
        if x.size not in self._mapped:
            self._mapped[x.size] = self.evaluate.map(x.size)
        return np.asarray(self._mapped[x.size](points)).reshape(x.shape)

    def title(self):
        fixed = ", ".join(f"{self.names[i]}={self.position[i]:.4g}"
                          for i in self.fixed_indices)
        force = np.array2string(self.data.force, precision=2)
        return f"{self.data.muscle_names[self.muscle_index]} | F={force} N\n{fixed}"

    def add_sliders(self, fig, update):
        sliders = []
        for row, index in enumerate(self.fixed_indices):
            slider_axis = fig.add_axes([0.25, 0.035 + 0.045 * row, 0.55, 0.025])
            slider = Slider(slider_axis, self.names[index],
                            self.grids[index][0], self.grids[index][-1],
                            valinit=self.position[index])

            def change(value, coordinate=index):
                self.position[coordinate] = value
                update()

            slider.on_changed(change)
            sliders.append(slider)
        # Widgets need a live reference after a nonblocking call returns.
        fig._interpolant_sliders = sliders


def _strict_grid_minima(values):
    """Find nodes lower than all eight neighbours, including domain boundaries.

    Flat minima/plateaus are deliberately not labelled as isolated minima.
    """
    padded = np.pad(values, 1, constant_values=np.inf)
    rows, columns = values.shape
    minima = np.ones(values.shape, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                neighbour = padded[1 + dy:1 + dy + rows, 1 + dx:1 + dx + columns]
                minima &= values < neighbour
    return minima


def show_interpolant_surface(interpolants, *, muscle=None, field="fatigue",
                             coordinate_axes=(0, 1), configuration=None,
                             resolution=60, show_minima=True, block=True):
    """Open a separate 3D surface window for every saved muscle, in saved order.

    Accept an Interpolants object or archive path. Optionally select one muscle
    by name/index. Each window has independent sliders for the other coordinates.
    Height and colour show the interpolant value, with limits fixed across slices
    for that muscle. Minima markers compare the eight neighbouring saved knots;
    these are slice map minima, not optima of the full trajectory objective.
    Returns a list of (figure, 3D axis) pairs. All windows open before blocking.
    """
    data = (Interpolants.from_npz(interpolants)
            if isinstance(interpolants, (str, Path)) else interpolants)
    selected = range(len(data.muscle_names)) if muscle is None else (muscle,)
    interpolant = data.interpolant(field)
    plots = []
    for selected_muscle in selected:
        slice_data = _InterpolantSlice(
            data, selected_muscle, field, coordinate_axes, configuration, interpolant,
        )
        plots.append(_plot_surface(slice_data, field, resolution, show_minima))
    plt.show(block=block)
    return plots


def _plot_surface(slice_data, field, resolution, show_minima):
    """Build one surface and its controls without entering the GUI event loop."""
    names, grids = slice_data.names, slice_data.grids
    x_index, y_index = slice_data.x_index, slice_data.y_index
    x_grid, y_grid = grids[x_index], grids[y_index]
    # Include knots so sharper peaks/valleys are not lost between display samples.
    x, y = np.meshgrid(
        np.unique(np.concatenate((x_grid, np.linspace(x_grid[0], x_grid[-1], resolution)))),
        np.unique(np.concatenate((y_grid, np.linspace(y_grid[0], y_grid[-1], resolution)))),
    )
    knot_x, knot_y = np.meshgrid(x_grid, y_grid)
    saved_values = {"fatigue": slice_data.data.fatigue_rate_grid,
                    "muscle_force": slice_data.data.muscle_force_grid}[field]
    muscle_values = np.asarray(saved_values).reshape(-1, len(slice_data.data.muscle_names))[
        :, slice_data.muscle_index
    ]
    low, high = float(muscle_values.min()), float(muscle_values.max())
    padding = max(high - low, abs(high), 1e-12) * 0.05
    norm = Normalize(low, high if high > low else low + padding)

    fig = plt.figure(figsize=(11, 8 + 0.5 * len(slice_data.fixed_indices)))
    fig.canvas.manager.set_window_title(
        f"{slice_data.data.muscle_names[slice_data.muscle_index]} — {slice_data.label}"
    )
    axis = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.12 + 0.045 * len(slice_data.fixed_indices), top=0.88)
    axis.set(xlabel=names[x_index], ylabel=names[y_index],
             zlabel=f"{slice_data.label} [{slice_data.unit}]",
             xlim=(x_grid[0], x_grid[-1]), ylim=(y_grid[0], y_grid[-1]),
             zlim=(low - padding, high + padding),
             title=f"{slice_data.label} landscape — fixed-coordinate slice")
    axis.set_box_aspect((1.2, 1, 0.7))
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axis,
                 shrink=0.65, pad=0.12, label=f"{slice_data.label} [{slice_data.unit}]")
    surface = None
    minima_markers = axis.scatter([], [], [], color="crimson", edgecolors="white",
                                  s=55, depthshade=False, label="Strict grid minima in slice")
    if show_minima:
        axis.legend(loc="upper left", fontsize="small")

    def update():
        nonlocal surface
        values = slice_data.sample(x, y)
        if surface is not None:
            surface.remove()
        surface = axis.plot_surface(x, y, values, cmap="viridis", norm=norm,
                                    rcount=x.shape[0], ccount=x.shape[1],
                                    linewidth=0, antialiased=True, alpha=0.9)
        if show_minima:
            knot_values = slice_data.sample(knot_x, knot_y)
            minima = _strict_grid_minima(knot_values)
            minima_markers._offsets3d = (knot_x[minima], knot_y[minima], knot_values[minima])
        fig.suptitle(slice_data.title())
        fig.canvas.draw_idle()

    slice_data.add_sliders(fig, update)
    update()
    return fig, axis
