"""Compare constant and posture-dependent recovery in the fatigue derivative.

Run with ``python -m code.visuals.fatigue_recovery``. This is a visualization
only; it does not alter the fatigue model used by the optimizer.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
import numpy as np

from ..msk_data import Interpolants, MSKdataset, MODEL_CONFIG


def show_fatigue_recovery_comparison(dataset, interpolants, *, muscle=None,
                                     coordinate_axes=(0, 1), configuration=None,
                                     resolution=60, k=3.0, exponent=8.0,
                                     reference_scale=1.0, block=True):
    """Show baseline and candidate fatigue derivative surfaces side by side.

    The candidate surface changes only the recovery branch: its recovery rate is
    multiplied by the paper's superelliptic function of force/max isometric force.
    The fatigue branch, activation threshold, and all optimizer code are untouched.
    """
    dataset = MSKdataset.from_npz(dataset) if isinstance(dataset, (str, Path)) else dataset
    interpolants = (Interpolants.from_npz(interpolants)
                    if isinstance(interpolants, (str, Path)) else interpolants)
    if muscle is None:
        plots = [show_fatigue_recovery_comparison(
            dataset, interpolants, muscle=i, coordinate_axes=coordinate_axes,
            configuration=configuration, resolution=resolution, k=k,
            exponent=exponent, reference_scale=reference_scale, block=False)
            for i in range(len(interpolants.muscle_names))]
        plt.show(block=block)
        return plots

    muscle_index = (interpolants.muscle_names.index(muscle)
                    if isinstance(muscle, str) else int(muscle))
    muscle_name = interpolants.muscle_names[muscle_index]
    model_muscle_index = dataset.info.muscle_names.index(muscle_name)
    max_force = float(dataset.info.muscle_max_forces[model_muscle_index])
    if max_force <= 0:
        raise ValueError(f"Maximum isometric force for {muscle_name} must be positive.")
    params = dataset.fatigue_params
    recovery_rate = float(params.recovery / params.capacity[model_muscle_index])
    initial_threshold = float(params.threshold[model_muscle_index])

    names = interpolants.joint_coordinate_names
    x_index, y_index = [names.index(item) if isinstance(item, str) else int(item)
                        for item in coordinate_axes]
    grids = [np.asarray(axis, dtype=float) for axis in interpolants.axes_list]
    fixed_indices = [i for i in range(len(grids)) if i not in (x_index, y_index)]
    position = np.array([(axis[0] + axis[-1]) / 2 for axis in grids]
                        if configuration is None else configuration, dtype=float)

    import casadi as ca
    q = ca.MX.sym("q", len(grids))
    rates_expr = interpolants.interpolant("fatigue")(q)[muscle_index]
    force_expr = interpolants.interpolant("muscle_force")(q)[muscle_index]
    evaluate = ca.Function("fatigue_recovery_comparison", [q], [rates_expr, force_expr])
    mapped = {}

    def sample(x, y):
        points = np.repeat(position[:, None], x.size, axis=1)
        points[x_index], points[y_index] = x.ravel(), y.ravel()
        if x.size not in mapped:
            mapped[x.size] = evaluate.map(x.size)
        rates, forces = mapped[x.size](points)
        return np.asarray(rates).reshape(x.shape), np.asarray(forces).reshape(x.shape)

    x_axis, y_axis = grids[x_index], grids[y_index]
    x_values = np.unique(np.r_[x_axis, np.linspace(x_axis[0], x_axis[-1], resolution)])
    y_values = np.unique(np.r_[y_axis, np.linspace(y_axis[0], y_axis[-1], resolution)])
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    rates, forces = sample(x_grid, y_grid)
    derivative_limit = max(float(np.nanmax(np.abs(rates))), recovery_rate, 1e-12)
    norm = Normalize(-derivative_limit, derivative_limit)

    fig = plt.figure(figsize=(15, 8))
    fig.canvas.manager.set_window_title(f"{muscle_name} — recovery comparison")
    ax_constant = fig.add_subplot(121, projection="3d")
    ax_posture = fig.add_subplot(122, projection="3d")
    fig.subplots_adjust(bottom=0.28 + 0.035 * len(fixed_indices), top=0.86, wspace=0.05)
    for axis, title in ((ax_constant, "Current model: constant recovery"),
                        (ax_posture, "Candidate: posture-dependent recovery")):
        axis.set(xlabel=names[x_index], ylabel=names[y_index],
                 zlabel="Fatigue-state derivative dm/dt [s⁻¹]",
                 xlim=(x_axis[0], x_axis[-1]), ylim=(y_axis[0], y_axis[-1]),
                 zlim=(-derivative_limit, derivative_limit), title=title)
        axis.set_box_aspect((1.1, 1, 0.7))
    fig.colorbar(ScalarMappable(norm=norm, cmap="coolwarm"), ax=[ax_constant, ax_posture],
                 shrink=0.65, pad=0.08, label="dm/dt [s⁻¹]")

    controls_bottom = 0.035
    n_fixed = len(fixed_indices)
    sliders = []

    def add_slider(label, row, vmin, vmax, valinit):
        slider_ax = fig.add_axes([0.25, controls_bottom + 0.04 * row, 0.50, 0.022])
        slider = Slider(slider_ax, label, vmin, vmax, valinit=np.clip(valinit, vmin, vmax))
        sliders.append(slider)
        return slider

    state_slider = add_slider("Fatigue state m", n_fixed, 0.0, 1.0, 0.5)
    threshold_slider = add_slider("Active-force threshold [N]", n_fixed + 1,
                                  0.0, max_force, min(initial_threshold, max_force))
    k_slider = add_slider("Recovery multiplier k", n_fixed + 2, 0.0, 20.0, k)
    exponent_slider = add_slider("Curve exponent", n_fixed + 3, 1.0, 16.0, exponent)
    reference_slider = add_slider("Reference force / max isometric", n_fixed + 4,
                                  0.1, 2.0, reference_scale)
    for row, index in enumerate(fixed_indices):
        slider_ax = fig.add_axes([0.25, controls_bottom + 0.04 * row, 0.50, 0.022])
        slider = Slider(slider_ax, names[index], grids[index][0], grids[index][-1],
                        valinit=position[index])

        def set_coordinate(value, coordinate=index):
            position[coordinate] = value
            update()

        slider.on_changed(set_coordinate)
        sliders.append(slider)

    surfaces = [None, None]

    def update(_value=None):
        nonlocal surfaces
        current_rates, current_forces = sample(x_grid, y_grid)
        fatigue_state = state_slider.val
        threshold = threshold_slider.val
        active = current_forces >= threshold
        baseline = np.where(active, (1.0 - fatigue_state) * current_rates,
                            -fatigue_state * recovery_rate)

        intensity = np.clip(current_forces / (max_force * reference_slider.val), 0.0, 1.0)
        curve_n = exponent_slider.val
        multiplier = 1.0 + k_slider.val * np.maximum(
            1.0 - intensity**curve_n, 0.0
        ) ** (1.0 / curve_n)
        candidate = np.where(active, (1.0 - fatigue_state) * current_rates,
                             -fatigue_state * recovery_rate * multiplier)

        for i, (axis, derivative) in enumerate(((ax_constant, baseline), (ax_posture, candidate))):
            if surfaces[i] is not None:
                surfaces[i].remove()
            surfaces[i] = axis.plot_surface(x_grid, y_grid, derivative, cmap="coolwarm", norm=norm,
                                            rcount=x_grid.shape[0], ccount=x_grid.shape[1],
                                            linewidth=0, antialiased=True, alpha=0.92)
        fixed = ", ".join(f"{names[i]}={position[i]:.4g}" for i in fixed_indices)
        fig.suptitle(f"{muscle_name} | m={fatigue_state:.2f} | threshold={threshold:.3g} N | {fixed}\n"
                     f"candidate parameters: k={k_slider.val:.2f}, n={curve_n:.2f}, "
                     f"reference={reference_slider.val:.2f}×max isometric force")
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)
    fig._fatigue_recovery_sliders = sliders
    update()
    plt.show(block=block)
    return fig, (ax_constant, ax_posture)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--muscle", help="Muscle name or interpolant column index")
    parser.add_argument("--coordinate-axes", type=int, nargs=2, default=(0, 1), metavar=("X", "Y"))
    parser.add_argument("--resolution", type=int, default=60)
    parser.add_argument("--k", type=float, default=3.0)
    parser.add_argument("--exponent", type=float, default=8.0)
    parser.add_argument("--reference-scale", type=float, default=1.0)
    args = parser.parse_args()
    muscle = args.muscle
    if muscle is not None:
        try:
            muscle = int(muscle)
        except ValueError:
            pass
    config = MODEL_CONFIG["tsm_3DOF"]
    show_fatigue_recovery_comparison(
        config["dataset_path"], config["interpolants_path"], muscle=muscle,
        coordinate_axes=args.coordinate_axes, resolution=args.resolution,
        k=args.k, exponent=args.exponent, reference_scale=args.reference_scale)


if __name__ == "__main__":
    main()
