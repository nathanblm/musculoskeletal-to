"""Plot a current fatigue model's joint-space derivative and local gradient.

Examples:
    python -m code.visuals.fatigue_derivative --model tsm_3DOF --muscle DeltoideusScapula_M
    python -m code.visuals.fatigue_derivative --model tsm_4DOF --fatigue-model 3cc

At fixed compartments, the 3CC fatigued-state derivative has zero direct
joint-space gradient. The default 3CC field is therefore the active-state
derivative, where posture enters through muscle force.
"""
from pathlib import Path

import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider
import numpy as np

from ..msk_data import MODEL_CONFIG, MSKdataset, ModelInfo, MuscleInfo
from ..msk_fatiguemodel import Thresholded, ThreeCompartment


def make_fatigue_model(model, dataset, muscle, kind="thresholded", force_index=0):
    """Build one selected muscle from the model and the current dataset maps."""
    dataset = MSKdataset.from_npz(dataset) if isinstance(dataset, (str, Path)) else dataset
    info = ModelInfo.from_model(model)
    if tuple(dataset.info.joint_coordinate_names) != tuple(info.joint_coordinate_names):
        raise ValueError("Dataset and model joint-coordinate order differ.")
    if tuple(dataset.info.muscle_names) != tuple(info.muscle_names):
        raise ValueError("Dataset and model muscle order differ.")
    if not 0 <= force_index < len(dataset.samples.forces):
        raise ValueError("force_index is outside the dataset force list.")
    if isinstance(muscle, (int, np.integer)):
        if not 0 <= muscle < len(info.muscle_names):
            raise ValueError("muscle index is outside the model muscle list.")
        muscle = info.muscle_names[int(muscle)]
    if muscle not in info.muscle_names:
        raise ValueError(f"Unknown muscle: {muscle!r}.")
    index = info.muscle_names.index(muscle)
    selected = MuscleInfo(
        names=(muscle,), indexes=np.array([index]),
        max_forces=np.array([info.muscle_max_forces[index]], dtype=float),
    )
    if kind == "thresholded":
        return Thresholded(dataset, selected, force_index=force_index)
    if kind == "3cc":
        return ThreeCompartment(dataset, selected, force_index=force_index)
    raise ValueError("fatigue_model must be 'thresholded' or '3cc'.")


def make_field(fatigue_model, field):
    """Return a CasADi function for field, spatial gradient, and muscle force."""
    q = ca.MX.sym("q", len(fatigue_model.interpolants.axes_list))
    state = ca.MX.sym("state", fatigue_model.n_states)
    force = fatigue_model.force_interpolant(q)[0]
    if field == "load":
        value = force / fatigue_model.muscles.max_forces[0]
        label = "Target load [fraction]"
    elif field == "fatigue":
        value = fatigue_model.derivative(q, state)[0]
        label = "Fatigue derivative [s⁻¹]"
    elif field == "active" and isinstance(fatigue_model, ThreeCompartment):
        value = fatigue_model.derivative(q, state)[1]
        label = "Active derivative [s⁻¹]"
    else:
        raise ValueError("field must be fatigue/load, or active for 3CC.")
    return ca.Function("fatigue_field", [q, state],
                       [value, ca.gradient(value, q), force]), label


def switch_neighborhood(mask):
    """Samples adjacent to a piecewise-model switch have no unique gradient."""
    near = np.zeros_like(mask, dtype=bool)
    change = mask[1:] != mask[:-1]
    near[1:] |= change
    near[:-1] |= change
    change = mask[:, 1:] != mask[:, :-1]
    near[:, 1:] |= change
    near[:, :-1] |= change
    return near


def show_fatigue_derivative_surface(
    model, dataset, *, muscle=0, fatigue_model="thresholded", field=None,
    force_index=0, coordinate_axes=(0, 1), configuration=None,
    resolution=60, block=True, show=True, save=None,
):
    """Show a 2D gradient map beside a 3D derivative surface."""
    fatigue = make_fatigue_model(model, dataset, muscle, fatigue_model, force_index)
    field = field or ("active" if fatigue_model == "3cc" else "fatigue")
    evaluate, label = make_field(fatigue, field)
    names = tuple(fatigue.interpolants.joint_coordinate_names)
    grids = [np.asarray(axis, float) for axis in fatigue.interpolants.axes_list]
    if len(grids) < 2 or resolution < 3:
        raise ValueError("At least two coordinates and resolution >= 3 are required.")
    axes = tuple(names.index(a) if isinstance(a, str) else int(a) for a in coordinate_axes)
    if len(axes) != 2 or axes[0] == axes[1] or any(a < 0 or a >= len(grids) for a in axes):
        raise ValueError("Select two distinct joint coordinates.")
    a, b = axes
    fixed = [i for i in range(len(grids)) if i not in axes]
    position = (np.asarray(configuration, float).copy() if configuration is not None
                else np.array([(g[0] + g[-1]) / 2 for g in grids]))
    if position.shape != (len(grids),):
        raise ValueError("configuration needs one value per joint coordinate.")
    x = np.linspace(grids[a][0], grids[a][-1], resolution)
    y = np.linspace(grids[b][0], grids[b][-1], resolution)
    xx, yy = np.meshgrid(x, y)
    mapped = evaluate.map(xx.size)

    fig = plt.figure(figsize=(13, 7))
    fig.subplots_adjust(bottom=.20 + .045 * len(fixed), top=.85, wspace=.28)
    plane = fig.add_subplot(121)
    surface_axis = fig.add_subplot(122, projection="3d")
    image = plane.pcolormesh(xx, yy, np.zeros_like(xx), shading="auto",
                             cmap="coolwarm", norm=Normalize(-1, 1))
    fig.colorbar(image, ax=plane, label=label, shrink=.8)
    plane.set(xlabel=names[a], ylabel=names[b], title="Derivative and downhill gradient")
    surface_axis.set(xlabel=names[a], ylabel=names[b],
                     zlabel='Load' if field == 'load' else 'Derivative [s⁻¹]',
                     title="Derivative surface")
    sliders = []

    def add_slider(name, row, minimum, maximum, initial):
        slider = Slider(fig.add_axes([.25, .04 + .045 * row, .55, .025]),
                        name, minimum, maximum, valinit=initial)
        sliders.append(slider)
        return slider

    for row, i in enumerate(fixed):
        slider = add_slider(names[i], row, grids[i][0], grids[i][-1], position[i])
        slider.on_changed(lambda value, i=i: set_position(i, value))
    mf_slider = add_slider("Fatigued fraction", len(fixed), 0, 1, 0)
    active_slider = (add_slider("Active share of nonfatigued", len(fixed) + 1, 0, 1, .2)
                     if fatigue_model == "3cc" else None)
    surface = None
    arrows = None

    def set_position(index, value):
        position[index] = value
        update()

    def update(_value=None):
        nonlocal surface, arrows
        mf = mf_slider.val
        if active_slider is None:
            state = np.array([mf])
        else:
            ma = active_slider.val * (1 - mf)
            state = np.array([mf, ma])
        points = np.repeat(position[:, None], xx.size, axis=1)
        points[a], points[b] = xx.ravel(), yy.ravel()
        values, gradient, forces = mapped(points, np.repeat(state[:, None], xx.size, axis=1))
        values = np.asarray(values).reshape(xx.shape)
        gradient = np.asarray(gradient)
        forces = np.asarray(forces).reshape(xx.shape)
        limit = max(float(np.max(np.abs(values))), 1e-12)
        image.set_array(values.ravel())
        image.set_clim(-limit, limit)
        if surface is not None:
            surface.remove()
        if arrows is not None:
            arrows.remove()
        surface = surface_axis.plot_surface(xx, yy, values, cmap="coolwarm",
                                            norm=image.norm, linewidth=0, alpha=.85)
        surface_axis.set_zlim(-limit, limit)
        gx, gy = -gradient[a].reshape(xx.shape), -gradient[b].reshape(xx.shape)
        length = np.hypot(gx, gy)
        if isinstance(fatigue, Thresholded):
            boundary = switch_neighborhood(forces >= fatigue.params.threshold[0])
        else:
            target = forces / fatigue.muscles.max_forces[0]
            mr = 1 - state[0] - state[1]
            boundary = (switch_neighborhood(target > state[1])
                        | switch_neighborhood(target - state[1] > mr))
        valid = (length > 1e-12) & ~boundary
        step = max(1, resolution // 14)
        sl = np.s_[::step, ::step]
        arrows = plane.quiver(
            xx[sl], yy[sl],
            np.divide(gx, length, out=np.zeros_like(gx), where=valid)[sl],
            np.divide(gy, length, out=np.zeros_like(gy), where=valid)[sl],
            angles="xy", scale_units="inches", scale=8, color="black", alpha=.7,
        )
        fig.suptitle(
            f"{fatigue.muscles.names[0]} | {fatigue_model} | {field} | Mf={mf:.2f}"
            + (f", Ma={state[1]:.2f}, Mr={1 - state[0] - state[1]:.2f}" if active_slider else "")
            + "\nArrows are local negative joint-space gradients; hidden joints are fixed."
        )
        fig.canvas.draw_idle()

    mf_slider.on_changed(update)
    if active_slider is not None:
        active_slider.on_changed(update)
    fig._fatigue_sliders = sliders
    update()
    if save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=160, bbox_inches="tight")
    if show:
        plt.show(block=block)
    return fig, (plane, surface_axis)


def main():
    import argparse
    import opensim as osim

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="tsm_3DOF", choices=MODEL_CONFIG)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--muscle", default="0")
    parser.add_argument("--fatigue-model", choices=("thresholded", "3cc"), default="thresholded")
    parser.add_argument("--field", choices=("fatigue", "active", "load"))
    parser.add_argument("--force-index", type=int, default=0)
    parser.add_argument("--coordinate-axes", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--resolution", type=int, default=60)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    config = MODEL_CONFIG[args.model]
    muscle = int(args.muscle) if args.muscle.isdecimal() else args.muscle
    show_fatigue_derivative_surface(
        osim.Model(config["model_path"]), args.dataset or config["dataset_path"],
        muscle=muscle, fatigue_model=args.fatigue_model, field=args.field,
        force_index=args.force_index, coordinate_axes=args.coordinate_axes,
        resolution=args.resolution, show=not args.no_show, save=args.save,
    )


if __name__ == "__main__":
    main()
