"""Reusable OpenSim decorations for sample clouds, forces, and paths."""

import os
from pathlib import Path
import sys

import numpy as np


class SimbodyRenderer:
    def __init__(self, model, endpoint_marker="endpoint"):
        os.environ.setdefault("SIMBODY_HOME", str(Path(sys.prefix) / "simbody"))
        import opensim as osim
        self.osim = osim
        self.model = osim.Model(str(model)) if isinstance(model, (str, Path)) else osim.Model(model)
        self.model.setUseVisualizer(True)
        self.state = self.model.initSystem()
        self.model.realizePosition(self.state)
        self.endpoint_marker = endpoint_marker
        self.visualizer = self.model.getVisualizer().updSimbodyVisualizer()
        self.ground = self.model.getGround().getMobilizedBodyIndex()
        self.point_ids, self.line_ids = [], []
        self.trajectory_line_ids, self.reference_line_ids = [], []
        self.circular_path_line_ids = []
        self.trajectory_marker_ids = []
        self.caption_id = None

        # Kept here so importing this module does not require OpenSim.
        class GeometryUpdater(osim.DecorativeGeometryImplementation):
            def __init__(self):
                super().__init__()
                self.points = (osim.Vec3(0), osim.Vec3(0))
                self.thickness = 1
                self.text = ""

            def implementLineGeometry(self, line):
                line.setPoint1(self.points[0])
                line.setPoint2(self.points[1])
                line.setLineThickness(self.thickness)

            def implementTextGeometry(self, text):
                text.setText(self.text)

        self.updater = GeometryUpdater()

    def _vec3(self, values):
        return self.osim.Vec3(*map(float, values))

    def marker_position(self, name):
        position = self.model.getMarkerSet().get(name).getLocationInGround(self.state)
        return np.array([position.get(i) for i in range(3)])

    def _add(self, geometry):
        return self.visualizer.addDecoration(self.ground, self.osim.Transform(), geometry)

    def _hide(self, identifiers):
        for identifier in identifiers:
            self.visualizer.updDecoration(identifier).setRepresentation(
                self.osim.DecorativeGeometry.Hide
            )

    def _set_spheres(self, points, colors, identifiers, radius, opacity=1.0):
        """Reuse decorations, hiding surplus spheres when a selection shrinks."""
        osim = self.osim
        for i, (point, color) in enumerate(zip(points, colors)):
            if i == len(identifiers):
                sphere = osim.DecorativeSphere(1.0)
                sphere.setResolution(0.25)
                identifiers.append(self._add(sphere))
            item = self.visualizer.updDecoration(identifiers[i])
            item.setTransform(osim.Transform(self._vec3(point)))
            item.setScaleFactors(osim.Vec3(radius, radius, radius))
            item.setColor(self._vec3(color))
            item.setOpacity(opacity)
            item.setRepresentation(osim.DecorativeGeometry.DrawSurface)
        self._hide(identifiers[len(points):])

    def _set_force(self, force, scale, origin):
        if force is None or np.linalg.norm(force) == 0:
            self._hide(self.line_ids)
            return
        start = self.marker_position(self.endpoint_marker) if origin is None else origin
        magnitude = np.linalg.norm(force)
        direction = force / magnitude
        end = start + scale * force
        side = np.cross(direction, np.eye(3)[np.argmin(np.abs(direction))])
        side /= np.linalg.norm(side)
        up = np.cross(direction, side)
        head_length = 0.25 * scale * magnitude
        base = end - head_length * direction
        segments = [(start, end)]
        segments += [(end, base + 0.5 * head_length * offset) for offset in (side, -side, up, -up)]
        self._set_lines(segments, self.line_ids, (0.1, 0.4, 1.0), 4)

    def _set_lines(self, segments, identifiers, color, thickness):
        """Update existing line geometry instead of accumulating decorations."""
        self.updater.thickness = thickness
        for i, (start, end) in enumerate(segments):
            self.updater.points = (self._vec3(start), self._vec3(end))
            if i == len(identifiers):
                identifiers.append(self._add(self.osim.DecorativeLine(*self.updater.points)))
            item = self.visualizer.updDecoration(identifiers[i])
            item.implementGeometry(self.updater)
            item.setColor(self._vec3(color))
            item.setRepresentation(self.osim.DecorativeGeometry.DrawSurface)
        self._hide(identifiers[len(segments):])

    def _set_path(self, points, identifiers, color, thickness):
        points = np.empty((0, 3)) if points is None else np.asarray(points, dtype=float)
        # Leave gaps where solver trajectories contain unavailable samples.
        segments = [(start, end) for start, end in zip(points[:-1], points[1:])
                    if np.isfinite(start).all() and np.isfinite(end).all()]
        self._set_lines(segments, identifiers, color, thickness)

    def _set_trajectory_markers(self, points, radius):
        points = np.asarray(points, dtype=float)
        finite = points[np.isfinite(points).all(axis=1)]
        endpoints = (finite[0], finite[-1]) if len(finite) else ()
        self._set_spheres(
            endpoints, ((0.1, 0.8, 0.2), (1.0, 0.2, 0.1)),
            self.trajectory_marker_ids, max(2.5 * radius, 0.008),
        )

    def _set_caption(self, caption):
        self.updater.text = caption
        if self.caption_id is None:
            label = self.osim.DecorativeText(caption)
            label.setIsScreenText(True)
            label.setColor(self.osim.Vec3(1.0, 0.4, 0.1))
            self.caption_id = self._add(label)
        else:
            self.visualizer.updDecoration(self.caption_id).implementGeometry(self.updater)

    def show(self, points, colors, caption, *, force=None, radius=0.006,
             opacity=1.0, force_scale=0.01, force_origin=None,
             trajectory=None, reference=None, circular_path=None):
        # Update every decoration first, then display once.
        self._set_spheres(points, colors, self.point_ids, radius, opacity)
        self._set_force(force, force_scale, force_origin)
        trajectory = np.empty((0, 3)) if trajectory is None else trajectory
        reference = np.empty((0, 3)) if reference is None else reference
        self._set_path(trajectory, self.trajectory_line_ids, (1.0, 0.15, 0.05), 5)
        self._set_path(reference, self.reference_line_ids, (0.55, 0.55, 0.55), 2)
        self._set_path(circular_path, self.circular_path_line_ids, (0.95, 0.65, 0.1), 3)
        self._set_trajectory_markers(trajectory, radius)
        self._set_caption(caption)
        self.model.getVisualizer().show(self.state)
        return len(points)


