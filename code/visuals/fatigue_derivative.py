"""Interactive joint-space plot of thresholded muscle-state derivatives.

Run with ``python -m code.visuals.fatigue_derivative DATASET.npz
--interpolants INTERPOLANTS.npz``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.widgets import Slider
import numpy as np

from ..msk_data import Interpolants, MSKdataset, MODEL_CONFIG


def show_fatigue_derivative_surface(dataset, interpolants, *, muscle=None,
                                    coordinate_axes=(0, 1), configuration=None,
                                    resolution=60, block=True):
    """Plot ``dm/dt`` with sliders for current fatigue state and force threshold.

    ``dataset`` and ``interpolants`` may be their data objects or NPZ paths.
    The selected muscle must be present in both. If no muscle is selected, open
    one window for every interpolant muscle. The threshold slider is in N; the
    state slider ranges from 0 (fresh) to 1 (fully fatigued).
    """
    dataset = MSKdataset.from_npz(dataset) if isinstance(dataset, (str, Path)) else dataset
    interpolants = (Interpolants.from_npz(interpolants)
                    if isinstance(interpolants, (str, Path)) else interpolants)

    if muscle is None:
        plots = [
            show_fatigue_derivative_surface(
                dataset, interpolants, muscle=index,
                coordinate_axes=coordinate_axes, configuration=configuration,
                resolution=resolution, block=False,
            )
            for index in range(len(interpolants.muscle_names))
        ]
        plt.show(block=block)
        return plots
    if isinstance(muscle, str):
        muscle_index = interpolants.muscle_names.index(muscle)
    else:
        muscle_index = int(muscle)
    muscle_name = interpolants.muscle_names[muscle_index]
    parameter_index = dataset.info.muscle_names.index(muscle_name)

    names = interpolants.joint_coordinate_names
    x_index, y_index = [names.index(item) if isinstance(item, str) else int(item)
                        for item in coordinate_axes]
    grids = [np.asarray(axis, dtype=float) for axis in interpolants.axes_list]
    fixed_indices = [i for i in range(len(grids)) if i not in (x_index, y_index)]
    position = np.array(
        [(axis[0] + axis[-1]) / 2 for axis in grids]
        if configuration is None else configuration, dtype=float,
    )

    import casadi as ca
    q = ca.MX.sym("q", len(grids))
    muscle_rates = interpolants.interpolant("fatigue")(q)[muscle_index]
    muscle_force = interpolants.interpolant("muscle_force")(q)[muscle_index]
    evaluate = ca.Function("fatigue_derivative_slice", [q], [muscle_rates, muscle_force])
    mapped = {}

    def sample(x, y):
        points = np.repeat(position[:, None], x.size, axis=1)
        points[x_index], points[y_index] = x.ravel(), y.ravel()
        if x.size not in mapped:
            mapped[x.size] = evaluate.map(x.size)
        rates, forces = mapped[x.size](points)
        return (np.asarray(rates).reshape(x.shape),
                np.asarray(forces).reshape(x.shape))

    x_axis, y_axis = grids[x_index], grids[y_index]
    x_values = np.unique(np.concatenate((x_axis, np.linspace(x_axis[0], x_axis[-1], resolution))))
    y_values = np.unique(np.concatenate((y_axis, np.linspace(y_axis[0], y_axis[-1], resolution))))
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    rates, forces = sample(x_grid, y_grid)

    params = dataset.fatigue_params
    capacity = float(params.capacity[parameter_index])
    recovery_rate = float(params.recovery / capacity)
    initial_threshold = float(params.threshold[parameter_index])
    all_forces = np.asarray(interpolants.muscle_force_grid).reshape(
        -1, len(interpolants.muscle_names)
    )[:, muscle_index]
    all_rates = np.asarray(interpolants.fatigue_rate_grid).reshape(
        -1, len(interpolants.muscle_names)
    )[:, muscle_index]
    max_force = float(np.nanmax(all_forces))
    threshold_max = max(max_force, initial_threshold, 1e-12)
    derivative_limit = max(float(np.nanmax(np.abs(all_rates))), recovery_rate, 1e-12)
    norm = Normalize(-derivative_limit, derivative_limit)

    fig = plt.figure(figsize=(11, 8))
    fig.canvas.manager.set_window_title(f"{muscle_name} — fatigue state derivative")
    axis = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(bottom=0.19 + 0.045 * len(fixed_indices), top=0.87)
    z_min, z_max = -derivative_limit, derivative_limit
    axis.set(xlabel=names[x_index], ylabel=names[y_index],
             zlabel="Muscle-state derivative, dm/dt [s⁻¹]",
             xlim=(x_axis[0], x_axis[-1]), ylim=(y_axis[0], y_axis[-1]),
             zlim=(z_min, z_max),
             title="Thresholded fatigue-state derivative")
    axis.set_box_aspect((1.2, 1, 0.7))
    fig.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ax=axis,
                 shrink=0.65, pad=0.12, label="dm/dt [s⁻¹]")

    controls_bottom = 0.035
    slider_axis_state = fig.add_axes([0.25, controls_bottom + 0.045 * len(fixed_indices), 0.55, 0.025])
    state_slider = Slider(slider_axis_state, "Initial fatigue state m", 0.0, 1.0, valinit=0.0)
    slider_axis_threshold = fig.add_axes([0.25, controls_bottom + 0.045 * (len(fixed_indices) + 1), 0.55, 0.025])
    threshold_slider = Slider(slider_axis_threshold, "Force threshold [N]", 0.0, threshold_max,
                              valinit=min(initial_threshold, threshold_max))

    fixed_sliders = []
    for row, index in enumerate(fixed_indices):
        slider_axis = fig.add_axes([0.25, controls_bottom + 0.045 * row, 0.55, 0.025])
        slider = Slider(slider_axis, names[index], grids[index][0], grids[index][-1],
                        valinit=position[index])

        def set_coordinate(value, coordinate=index):
            position[coordinate] = value
            update()

        slider.on_changed(set_coordinate)
        fixed_sliders.append(slider)

    surface = None
    threshold_contour = None

    def update(_value=None):
        nonlocal surface, threshold_contour, rates, forces
        rates, forces = sample(x_grid, y_grid)
        state = state_slider.val
        threshold = threshold_slider.val
        derivative = np.where(forces >= threshold,
                              (1.0 - state) * rates,
                              -state * recovery_rate)
        if surface is not None:
            surface.remove()
        if threshold_contour is not None:
            threshold_contour.remove()
        surface = axis.plot_surface(x_grid, y_grid, derivative, cmap="coolwarm", norm=norm,
                                    rcount=x_grid.shape[0], ccount=x_grid.shape[1],
                                    linewidth=0, antialiased=True, alpha=0.92)
        if np.nanmin(forces) <= threshold <= np.nanmax(forces):
            threshold_contour = axis.contour(
                x_grid, y_grid, forces, levels=[threshold], zdir="z", offset=z_min,
                colors="black", linewidths=2,
            )
        else:
            threshold_contour = None
        fixed = ", ".join(f"{names[i]}={position[i]:.4g}" for i in fixed_indices)
        fig.suptitle(
            f"{muscle_name} | m={state:.2f} | threshold={threshold:.3g} N\n"
            f"black contour: F(q)=threshold | {fixed}"
        )
        fig.canvas.draw_idle()

    state_slider.on_changed(update)
    threshold_slider.on_changed(update)
    fig._fatigue_derivative_sliders = [state_slider, threshold_slider, *fixed_sliders]
    update()
    plt.show(block=block)
    return fig, axis


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muscle", help="Muscle name or interpolant column index")
    parser.add_argument("--coordinate-axes", type=int, nargs=2, default=(0, 1), metavar=("X", "Y"))
    parser.add_argument("--resolution", type=int, default=60)
    args = parser.parse_args()
    muscle = args.muscle
    if muscle is not None:
        try:
            muscle = int(muscle)
        except ValueError:
            pass
    model_config = MODEL_CONFIG["tsm_3DOF"]
    show_fatigue_derivative_surface(
        model_config["dataset_path"], model_config["interpolants_path"], muscle=muscle,
        coordinate_axes=args.coordinate_axes, resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
