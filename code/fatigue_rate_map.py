"""Plot sampled fatigue rates after selecting endpoint positions in Cartesian space."""

from pathlib import Path
from numbers import Real
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .msk_data import MSKdataset, MODEL_CONFIG


def plot_fatigue_rate_maps(
    dataset: Union[MSKdataset, str, Path],
    *,
    endpoint_bounds=None,
    muscles=None,
    force_index=0,
    space="joint",
    plot_axes=None,
    filter_success=True,
    point_size=8,
    opacity=0.8,
    threshold_fraction=0.3,
    block=True,
):
    """Plot one fatigue-rate scatter map per muscle and return ``(fig, axes)``.

    ``endpoint_bounds`` is a mapping from ``'x'``, ``'y'``, or ``'z'`` to an
    inclusive ``(minimum, maximum)`` interval in ground-frame metres. Omitted
    axes, or ``None`` for either end of an interval, are unbounded. The bounds
    always select samples by their saved Cartesian endpoint positions, even
    when ``space='joint'``. For example::

        fig, axes = plot_fatigue_rate_maps(
            "results/my_dataset.npz",
            endpoint_bounds={"x": (0.1, 0.4), "y": (-0.2, None)},
            muscles=("BIC_long", "TRIlong"),
            force_index=0,
            space="joint",
        )

    ``plot_axes`` selects two or three coordinates to display. Joint axes may
    be indices or names from ``dataset.info.joint_coordinate_names``; Cartesian
    axes may be indices or ``'x'``, ``'y'``, ``'z'``. By default, the first three
    coordinates are shown in 3D. Each subplot has its own fatigue-rate colour scale.
    ``axes`` is a flat array of Matplotlib axes, one per selected muscle.
    Only load NPZ files from trusted sources because MSKdataset uses pickle.
    """
    if isinstance(dataset, (str, Path)):
        dataset = MSKdataset.from_npz(dataset)
    if not isinstance(dataset, MSKdataset):
        raise TypeError("dataset must be an MSKdataset or a path to one.")

    samples, results, info = dataset.samples, dataset.results, dataset.info
    endpoints = np.asarray(samples.cartesian_coordinates)
    joints = np.asarray(samples.joint_coordinates)
    rates = np.asarray(results.sample_fatigue_rates)
    names = tuple(info.muscle_names)
    n_samples = len(endpoints)
    n_forces = len(samples.forces)
    if endpoints.shape != (n_samples, 3) or joints.ndim != 2 or len(joints) != n_samples:
        raise ValueError("Saved endpoint and joint coordinates must have matching sample rows.")
    if rates.shape != (n_samples, n_forces, len(names)):
        raise ValueError("sample_fatigue_rates must have shape (samples, forces, muscles).")
    if not isinstance(force_index, (int, np.integer)) or not 0 <= force_index < n_forces:
        raise IndexError("force_index is outside the saved force samples.")
    if filter_success and np.asarray(results.sample_success).shape != (n_samples, n_forces):
        raise ValueError("sample_success must have shape (samples, forces).")
    if space not in ("joint", "cartesian"):
        raise ValueError("space must be 'joint' or 'cartesian'.")
    if not np.isfinite(point_size) or point_size <= 0:
        raise ValueError("point_size must be finite and positive.")
    if not np.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("opacity must be finite and between 0 and 1.")
    if not np.isfinite(threshold_fraction) or not 0 <= threshold_fraction <= 1:
        raise ValueError("threshold_fraction must be between 0 and 1.")

    if muscles is None:
        selected_names = names
    elif isinstance(muscles, str):
        selected_names = (muscles,)
    else:
        selected_names = tuple(muscles)
    if not selected_names or len(set(selected_names)) != len(selected_names):
        raise ValueError("muscles must contain distinct muscle names.")
    unknown = set(selected_names) - set(names)
    if unknown:
        raise ValueError(f"Unknown muscles: {sorted(unknown)}")

    if endpoint_bounds is None:
        endpoint_bounds = {}
    if not hasattr(endpoint_bounds, "items"):
        raise TypeError("endpoint_bounds must map x, y, or z to (minimum, maximum).")
    mask = np.isfinite(endpoints).all(axis=1)
    for axis_name, interval in endpoint_bounds.items():
        if axis_name not in ("x", "y", "z"):
            raise ValueError("endpoint_bounds keys must be 'x', 'y', or 'z'.")
        try:
            lower, upper = interval
        except (TypeError, ValueError) as exc:
            raise ValueError(f"endpoint_bounds[{axis_name!r}] must be a pair.") from exc
        for value in (lower, upper):
            if value is not None and (not isinstance(value, Real) or not np.isfinite(value)):
                raise ValueError("Endpoint bounds must be finite numbers or None.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Lower {axis_name} bound exceeds upper bound.")
        coordinate = endpoints[:, "xyz".index(axis_name)]
        if lower is not None:
            mask &= coordinate >= lower
        if upper is not None:
            mask &= coordinate <= upper

    points = joints if space == "joint" else endpoints
    coordinate_names = tuple(info.joint_coordinate_names) if space == "joint" else ("x", "y", "z")
    if len(coordinate_names) != points.shape[1]:
        raise ValueError("Coordinate names must match the plotted coordinate columns.")
    if plot_axes is None:
        plot_axes = (0, 1, 2)
    if len(plot_axes) not in (2, 3):
        raise ValueError("plot_axes must select two or three distinct coordinates.")
    axis_indices = []
    for axis in plot_axes:
        if isinstance(axis, str):
            if axis not in coordinate_names:
                raise ValueError(f"Unknown plot coordinate {axis!r}.")
            axis = coordinate_names.index(axis)
        if not isinstance(axis, (int, np.integer)) or not 0 <= axis < points.shape[1]:
            raise ValueError("plot_axes contains an invalid coordinate index.")
        axis_indices.append(int(axis))
    if len(set(axis_indices)) != len(axis_indices):
        raise ValueError("plot_axes must select distinct coordinates.")

    mask &= np.isfinite(points[:, axis_indices]).all(axis=1)
    if filter_success:
        mask &= results.sample_success[:, force_index]
    if not np.any(mask):
        raise ValueError("No samples match the endpoint bounds and success filter.")

    muscle_indices = [names.index(name) for name in selected_names]
    selected_rates = rates[:, force_index, muscle_indices]
    if not np.isfinite(selected_rates[mask]).any():
        raise ValueError("No finite fatigue rates match the selected samples and muscles.")

    n_columns = min(3, len(muscle_indices))
    n_rows = (len(muscle_indices) + n_columns - 1) // n_columns
    fig = plt.figure(figsize=(5 * n_columns, 4 * n_rows + 0.7))
    axes = []
    for position, name in enumerate(selected_names):
        ax = fig.add_subplot(n_rows, n_columns, position + 1,
                             projection="3d" if len(axis_indices) == 3 else None)
        axes.append(ax)
        values = rates[:, force_index, muscle_indices[position]]
        forces = results.sample_muscle_forces[:, force_index, muscle_indices[position]]
        valid = mask & np.isfinite(values) & np.isfinite(forces)
        threshold = threshold_fraction * info.muscle_max_forces[muscle_indices[position]]
        active = valid & (forces >= threshold)
        inactive = valid & ~active
        if np.any(inactive):
            ax.scatter(*(points[inactive, i] for i in axis_indices),
                       color="lightgray", s=point_size, alpha=opacity)
        if np.any(active):
            scatter = ax.scatter(*(points[active, i] for i in axis_indices),
                                 c=values[active], cmap="viridis", s=point_size,
                                 alpha=opacity)
            fig.colorbar(scatter, ax=ax, label="Fatigue rate above threshold [s$^{-1}$]", shrink=0.8)
        if np.any(active) or np.any(inactive):
            ax.legend(handles=[
                Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
                       markersize=7, label="below threshold"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue",
                       markersize=7, label="above threshold"),
            ], loc="best")
        else:
            if len(axis_indices) == 3:
                ax.text2D(0.5, 0.5, "No finite rates", ha="center", transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "No finite rates", ha="center", transform=ax.transAxes)
        ax.set_title(name)
        labels = [f"{coordinate_names[i]} [m]" if space == "cartesian" else coordinate_names[i]
                  for i in axis_indices]
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if len(axis_indices) == 3:
            ax.set_zlabel(labels[2])
        elif space == "cartesian":
            ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
    fig.suptitle(
        f"Fatigue rates | threshold {threshold_fraction:g} | "
        f"force {force_index} | {int(mask.sum())} endpoint samples"
    )
    fig.tight_layout()
    plt.show(block=block)
    return fig, np.asarray(axes, dtype=object)

if __name__ == "__main__":
    model_name = "tsm_3DOF"

    fig, axes = plot_fatigue_rate_maps(
    MODEL_CONFIG[model_name]["dataset_path"],
    endpoint_bounds={"x": (0.0, None), "y": (0.0, None), "z": (-0.1, None)},
    muscles=("DeltoideusScapula_M", "Infraspinatus_S"),
    force_index=0,
    threshold_fraction=0.1,
)
