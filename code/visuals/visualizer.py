"""Plot dataset maps and OCP trajectories using the schemas in msk_data.

Inputs are assumed to satisfy those schemas. Failed/non-finite solver samples
are filtered for display; array shapes and argument types are not revalidated.
OpenSim is loaded only when a model operation is requested.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..msk_data import MSKdataset
from .simbody_renderer import SimbodyRenderer


def _sample_path(path, count=181, *, endpoint=True):
    """Sample a one-parameter reference curve in Cartesian space."""
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=endpoint)
    return np.asarray([np.asarray(path.evaluate(angle), dtype=float).reshape(3)
                       for angle in angles])


class Visualizer:
    _FIELDS = {
        "activation": ("sample_activations", "Activation", "RdYlGn_r", (0, 1)),
        "muscle_force": ("sample_muscle_forces", "Muscle force [N]", "viridis", None),
        "fatigue": ("sample_fatigue_rates", "Fatigue rate [s^-1]", "viridis", None),
        "dominance": ("sample_fatigue_dominance", "Fatigue dominance", "coolwarm", (-1, 1)),
    }

    def __init__(self, model=None, *, data: Optional["MSKdataset"] = None,
                 model_path=None, muscle_names=None,
                 endpoint_marker="endpoint"):
        """Use an OpenSim model for motion export and an optional MSKdataset for plots."""
        if isinstance(model, (str, Path)):
            model_path, model = model, None
        self.model = model
        self.data = data
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else None
        )
        self.endpoint_marker = endpoint_marker
        if muscle_names is None:
            muscle_names = data.info.muscle_names if data is not None else ()
        self.muscle_names = tuple(muscle_names)
        self._renderer = None
        self._reference_model = None
        self._reference_state = None

    @classmethod
    def from_npz(cls, npz_path: Union[Path, str], **kwargs):
        return cls(data=MSKdataset.from_npz(npz_path), **kwargs)

    def _get_model(self):
        if self.model is None:
            if self.model_path is None:
                raise ValueError("Provide a model or model_path for this operation.")
            import opensim as osim
            self.model = osim.Model(str(self.model_path))
        return self.model

    def save_motion_sto(self, path, time, states):
        """Export unlocked coordinate values and speeds to an OpenSim .sto file.

        states has shape (2 * n_unlocked_coordinates, n_times), with rows
        [q1, speed1, q2, speed2, ...] in the model's CoordinateSet order.
        Rotations are in radians. Labels come from each coordinate's full model
        path; locked coordinates remain at the defaults in the supplied model.
        Export does not initialize a simulation or open a visualizer window.
        """
        model = self._get_model()
        model.finalizeConnections()
        state_labels = [
            f"{coordinate.getAbsolutePathString()}/{kind}"
            for coordinate in model.getCoordinateSet()
            if not coordinate.get_locked()
            for kind in ("value", "speed")
        ]
        time = np.asarray(time, dtype=float)
        states = np.asarray(states, dtype=float)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.write("motion\n")
            file.write("version=1\n")
            file.write(f"nRows={len(time)}\n")
            file.write(f"nColumns={len(state_labels) + 1}\n")
            file.write("inDegrees=no\n")
            file.write("endheader\n")
            file.write("time\t" + "\t".join(state_labels) + "\n")
            for k, sample_time in enumerate(time):
                row = np.concatenate(([sample_time], states[:, k]))
                file.write("\t".join(f"{value:.12g}" for value in row) + "\n")
        return path

    def _simbody(self):
        if self._renderer is None:
            self._renderer = SimbodyRenderer(self._get_model(), self.endpoint_marker)
        return self._renderer

    def _marker_position(self, name):
        if self._renderer is not None:
            return self._renderer.marker_position(name)
        if self._reference_model is None:
            import opensim as osim
            self._reference_model = osim.Model(self._get_model())
            self._reference_model.setUseVisualizer(False)
            self._reference_state = self._reference_model.initSystem()
            self._reference_model.realizePosition(self._reference_state)
        marker = self._reference_model.getMarkerSet().get(name)
        position = marker.getLocationInGround(self._reference_state)
        return np.array([position.get(i) for i in range(3)])

    def _muscle_index(self, muscle):
        return self.muscle_names.index(muscle) if isinstance(muscle, str) else int(muscle)

    def _plot_coordinates(self, space, coordinate_axes, slice_axis, slice_coordinate):
        """Resolve the sample coordinates and axes used for projection/slicing."""
        if space == "cartesian":
            points = self.data.samples.cartesian_coordinates
            axis_names = ("X", "Y", "Z")
            plot_axes = [0, 1, 2]
            slice_index = None if slice_axis is None else "xyz".index(slice_axis)
            title = "Workspace"
        else:
            points = self.data.samples.joint_coordinates
            names = self.data.info.joint_coordinate_names
            coordinates = (0, 1) if coordinate_axes is None else coordinate_axes
            plot_axes = [names.index(item) if isinstance(item, str) else int(item)
                         for item in coordinates]
            axis_names = tuple(names)
            slice_index = names.index(slice_coordinate) if isinstance(slice_coordinate, str) else slice_coordinate
            title = "Joint space"
        return points, axis_names, plot_axes, slice_index, title

    def _field_colors(self, field, muscle, force_index, mask, value_range, cmap, title):
        """Filter unavailable values and scale colours before display subsampling."""
        values, norm, colormap, label = None, None, None, None
        if field != "workspace":
            attribute, label, default_cmap, default_range = self._FIELDS[field]
            array = getattr(self.data.results, attribute)
            title = label
            if field == "dominance":
                values = array[:, force_index]
            else:
                index = self._muscle_index(muscle)
                values = array[:, force_index, index]
                name = self.muscle_names[index]
                title += f" | {name}"
            mask &= np.isfinite(values)
            if not np.any(mask):
                raise ValueError("No finite samples satisfy the validity filter.")
            limits = value_range if value_range is not None else default_range
            if limits is None:
                # Full valid sample range, before slicing/subsampling, in physical units.
                limits = (0.0, max(float(values[mask].max()), 1e-12))
            limits = np.asarray(limits, dtype=float)
            norm = Normalize(*limits, clip=True)
            colormap = plt.get_cmap(cmap or default_cmap)

        return values, norm, colormap, label, title

    def show(self, field="activation", *, muscle=None, force_index=0, space="cartesian",
             backend="matplotlib", slice_axis=None, slice_value=None,
             coordinate_axes=None, slice_coordinate=None,
             tolerance=0.01, reference_marker=None, filter_success=True,
             value_range=None, cmap=None, max_points=None, radius=0.003,
             opacity=0.8, point_size=8, show_force=True, force_scale=0.01,
             force_origin=None, trajectory=None, reference_trajectory=None,
             circular_path=None, block=True):
        """Display sampled values in Cartesian or joint space; return (fig, ax) or point count.

        slice_axis: 'x', 'y' or 'z'. slice_value: absolute ground coordinate,
        or an offset from reference_marker in the model's default pose (metres).
        In joint space, coordinate_axes selects two or three coordinate indices
        (default: the first two), and slice_coordinate selects an optional
        coordinate by index or name. Matplotlib projects slices onto the
        remaining plotted axes; Simbody retains 3D Cartesian positions.
        value_range fixes the colour scale, useful for comparing sample evaluations.
        max_points optionally subsamples only the displayed points.
        force_scale is metres/Newton; force_origin is a ground-frame point.
        trajectory and reference_trajectory optionally add paths to Simbody.
        circular_path optionally adds its complete reference curve to Simbody.
        """

        points, axis_names, plot_axes, slice_index, title = self._plot_coordinates(
            space, coordinate_axes, slice_axis, slice_coordinate,
        )
        mask = np.isfinite(points).all(axis=1)
        if filter_success:
            mask &= self.data.results.sample_success[:, force_index]
        values, norm, colormap, label, title = self._field_colors(
            field, muscle, force_index, mask, value_range, cmap, title,
        )

        if space == "cartesian":
            origin = np.zeros(3) if reference_marker is None else self._marker_position(reference_marker)
        else:
            origin = np.zeros(points.shape[1])
        axis = slice_index
        if axis is not None:
            plane = (origin[axis] + slice_value) if space == "cartesian" else slice_value
            mask &= np.abs(points[:, axis] - plane) <= tolerance
            axis_label = slice_axis if space == "cartesian" else axis_names[axis]
            unit = " m" if space == "cartesian" else ""
            title += f" | {axis_label}={plane:.3f} +/- {tolerance:g}{unit}"
        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            raise ValueError("No samples match the selection; check the slice and success filter.")
        if max_points is not None and len(indices) > max_points:
            indices = np.random.default_rng(0).choice(indices, size=max_points, replace=False)
        selected_points = points[indices]
        force = self.data.samples.forces[force_index]
        title += f" | F[{force_index}]={np.array2string(force, precision=2)} N"
        if force_origin is not None:
            force_origin = np.asarray(force_origin, dtype=float)

        if backend == "simbody":
            colors = np.tile([0.1, 0.4, 1.0], (len(indices), 1)) if values is None else colormap(norm(values[indices]))[:, :3]
            caption = title if norm is None else f"{title} | scale [{norm.vmin:g}, {norm.vmax:g}]"
            reference_path = None if circular_path is None else _sample_path(circular_path)
            return self._simbody().show(
                selected_points, colors, caption, force=force if show_force else None,
                radius=radius, opacity=opacity, force_scale=force_scale, force_origin=force_origin,
                trajectory=trajectory, reference=reference_trajectory,
                circular_path=reference_path,
            )

        displayed = selected_points - origin
        fig = plt.figure()
        axes = plot_axes if space == "joint" or axis is None else [i for i in plot_axes if i != axis]
        is_3d = len(axes) == 3
        ax = fig.add_subplot(111, projection="3d") if is_3d else fig.add_subplot(111)
        style = {"s": point_size, "alpha": opacity}
        if values is None:
            style["color"] = "tab:blue"
        else:
            style.update(c=values[indices], cmap=colormap, norm=norm)
        scatter = ax.scatter(*(displayed[:, i] for i in axes), **style)
        if values is not None:
            fig.colorbar(scatter, ax=ax, label=label, shrink=0.8)
        if space == "cartesian":
            reference = "ground" if reference_marker is None else reference_marker
            labels = [f"{'XYZ'[i]} relative to {reference} [m]" for i in axes]
        else:
            labels = [axis_names[i] for i in axes]
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        if is_3d:
            ax.set_zlabel(labels[2])
            span = np.ptp(displayed[:, axes], axis=0)
            ax.set_box_aspect(np.maximum(span, max(float(span.max()), 1e-3) * 0.01))
        elif space == "cartesian":
            ax.set_aspect("equal", adjustable="box")
        if show_force and space == "cartesian" and np.linalg.norm(force) > 0:
            # Without a model this is a reference arrow at the selected cloud's centre.
            start = (selected_points.mean(axis=0) if force_origin is None else force_origin) - origin
            vector = force_scale * force
            if is_3d:
                ax.quiver(*start, *vector, color="tab:blue", normalize=False)
            elif np.linalg.norm(vector[axes]) > 0:
                ax.quiver(*start[axes], *vector[axes], angles="xy", scale_units="xy", scale=1, color="tab:blue")
        ax.set_title(title, fontsize=10)
        ax.grid(True)
        fig.tight_layout()
        plt.show(block=block)
        return fig, ax

    @staticmethod
    def _muscle_states(ocp_result, states):
        """Muscle rows follow the interleaved joint position/speed rows."""
        return np.asarray(states, dtype=float)[2 * len(ocp_result.pos_indices):]

    @staticmethod
    def _style_state_axis(axis, title, ylabel):
        axis.set(title=title, xlabel="Time [s]", ylabel=ylabel, ylim=(0, 1))
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")

    def _ocp_muscle_selection(self, ocp_result):
        """Return dataset indices and state rows in saved OCP muscle order."""
        saved_names = tuple(ocp_result.muscle_names)

        dataset_indices = [self._muscle_index(name) for name in saved_names]
        return dataset_indices, saved_names, self._muscle_states(ocp_result, ocp_result.states)

    def visualize_ocp_result(
        self,
        ocp_result,
        *,
        force_index=0,
        fatigue_states=None,
        show_agonist_antagonist=False,
        slice_axis=None,
        slice_value=None,
        reference_marker=None,
        tolerance=0.01,
        filter_success=True,
        value_range=None,
        cmap=None,
        max_points=None,
        radius=0.003,
        opacity=0.5,
        show_force=True,
        force_scale=0.01,
        force_origin=None,
        circular_path=None,
        block=True,
    ):
        """Show dominance samples and trajectories in Simbody, with fatigue in Matplotlib.

        ``ocp_result`` may be an :class:`OCPResult` or a path to its ``.npz``
        archive. Fatigue traces follow the muscle names saved in the result.
        Set ``show_agonist_antagonist`` to overlay configured group means.
        Returns ``(number_of_map_points, fatigue_figure, fatigue_axis)``.
        """
        if isinstance(ocp_result, (str, Path)):
            from ..msk_data import OCPResult
            ocp_result = OCPResult.from_npz(ocp_result)
        time = np.asarray(ocp_result.time, dtype=float)

        endpoint = ocp_result.endpoint_positions.T
        reference = ocp_result.reference.T

        _, saved_names, ocp_fatigue = self._ocp_muscle_selection(ocp_result)
        if fatigue_states is None:
            fatigue_states = ocp_fatigue
        fatigue_states = np.asarray(fatigue_states, dtype=float)

        groups = (
            (self.data.info.agonist_muscles, "Agonists", "tab:red"),
            (self.data.info.antagonist_muscles, "Antagonists", "tab:blue"),
        ) if show_agonist_antagonist else ()
        group_colors = {name: color for names, _, color in groups for name in names}
        fatigue_fig, fatigue_ax = plt.subplots(figsize=(7, 5))
        for name, states in zip(saved_names, fatigue_states):
            fatigue_ax.plot(time, states, color=group_colors.get(name), alpha=0.45, label=name)
        for names, label, color in groups:
            rows = [saved_names.index(name) for name in names if name in saved_names]
            if rows:
                fatigue_ax.plot(time, fatigue_states[rows].mean(axis=0), color=color,
                                linewidth=2.5, label=f"{label} mean")
        title = "Agonist–antagonist fatigue progression" if groups else "Muscle fatigue progression"
        self._style_state_axis(fatigue_ax, title, "Fatigue state")

        fatigue_fig.tight_layout()
        plt.show(block=False)
        point_count = self.show(
            "dominance",
            force_index=force_index,
            backend="simbody",
            slice_axis=slice_axis,
            slice_value=slice_value,
            reference_marker=reference_marker,
            tolerance=tolerance,
            filter_success=filter_success,
            value_range=value_range,
            cmap=cmap,
            max_points=max_points,
            radius=radius,
            opacity=opacity,
            show_force=show_force,
            force_scale=force_scale,
            force_origin=force_origin,
            trajectory=endpoint,
            reference_trajectory=reference,
            circular_path=circular_path,
            block=False,
        )
        plt.show(block=block)
        return point_count, fatigue_fig, fatigue_ax

    def compare_muscle_state_trajectories(
        self,
        ocp_result,
        validation_data,
        *,
        block=True,
    ):
        """Compare continuous-map, discrete-map, and RMR muscle-state trajectories.

        Each simulated muscle gets its own subplot. OCP state arrays follow
        the muscle order saved with the OCP result; validation states are
        expected in that same order.
        """

        ocp_time = np.asarray(ocp_result.time, dtype=float)
        validation_time = np.asarray(validation_data.time, dtype=float)
        validation_states = validation_data.muscle_states
        validation_states = np.asarray(validation_states, dtype=float)

        selected, saved_names, ocp_muscle_states = self._ocp_muscle_selection(ocp_result)
        discrete_muscle_states = self._muscle_states(ocp_result, ocp_result.val_states)

        columns = min(3, len(selected))
        rows = int(np.ceil(len(selected) / columns))
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(5 * columns, 3.2 * rows),
            squeeze=False,
        )
        flat_axes = axes.ravel()
        for row, (axis, index) in enumerate(zip(flat_axes, selected)):
            color = None
            for time, states, label, style in (
                (ocp_time, ocp_muscle_states, "Continuous map (OCP)", "-"),
                (ocp_time, discrete_muscle_states, "Discrete map", ":"),
                (validation_time, validation_states, "RMR validation", "--"),
            ):
                line, = axis.plot(time, states[row], label=label, linestyle=style,
                                  color=color, linewidth=2)
                color = line.get_color()
            self._style_state_axis(axis, self.muscle_names[index], "Muscle state")

        for axis in flat_axes[len(selected):]:
            axis.set_visible(False)

        fig.suptitle("Muscle-state trajectories: continuous map, discrete map, and RMR")
        fig.tight_layout()
        plt.show(block=block)
        return fig, axes

    def show_activation(self, muscle, force_index=0, **kwargs):
        return self.show("activation", muscle=muscle, force_index=force_index, **kwargs)

    def show_muscle_force(self, muscle, force_index=0, **kwargs):
        return self.show("muscle_force", muscle=muscle, force_index=force_index, **kwargs)

    def show_fatigue(self, muscle, force_index=0, **kwargs):
        return self.show("fatigue", muscle=muscle, force_index=force_index, **kwargs)

    def show_dominance(self, force_index=0, **kwargs):
        return self.show("dominance", force_index=force_index, **kwargs)

    def show_workspace(self, force_index=0, **kwargs):
        kwargs.setdefault("filter_success", False)
        return self.show("workspace", force_index=force_index, **kwargs)

    @staticmethod
    def show_interpolant_surface(interpolants, **kwargs):
        """Open one joint-space surface per saved muscle; return figure/axis pairs."""
        from .interpolants import show_interpolant_surface
        return show_interpolant_surface(interpolants, **kwargs)

    def show_circular_path(self, circular_path, *, block=True):
        """Display only a circular reference path in the Simbody model view."""
        path_points = _sample_path(circular_path)
        renderer = self._simbody()
        renderer.show(
            np.empty((0, 3)), np.empty((0, 3)), "Circular reference path",
            circular_path=path_points,
        )
        if block:
            input("Press Enter to close the viewer...")
        return len(path_points)


if __name__ == "__main__":
    from .visualizer_cli import main

    main()
