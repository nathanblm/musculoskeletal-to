"""Estimate OpenSim arm coordinates from RTMW3D arm and hand landmarks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import opensim as osim


# Camera: X right, Y down, Z away. OpenSim: X forward, Y up, Z right.
CAMERA_TO_OPENSIM = np.array([
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
])


@dataclass(frozen=True)
class ArmMeasurement:
    """Shoulder-relative arm and MCP landmarks in camera coordinates, in metres."""

    elbow: np.ndarray
    wrist: np.ndarray
    index_mcp: np.ndarray
    pinky_mcp: np.ndarray


class OpenSimArmJointState:
    """Fit five landmarks with OpenSim's marker inverse kinematics solver."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        alignment: np.ndarray = CAMERA_TO_OPENSIM,
        depth_sign: float = 1.0,
    ):
        self.model = osim.Model(str(model_path))
        self.state = self.model.initSystem()
        self.markers = tuple(
            self.model.getMarkerSet().get(name)
            for name in ('shoulder', 'elbow', 'wrist', 'index_mcp', 'pinky_mcp')
        )
        self.marker_names = [marker.getName() for marker in self.markers]
        coordinates = self.model.getCoordinateSet()
        self.coordinates = tuple(
            coordinates.get(i) for i in range(coordinates.getSize())
            if not coordinates.get(i).get_locked()
        )
        self.coordinate_names = tuple(c.getName() for c in self.coordinates)
        for coordinate in self.coordinates:
            coordinate.setClamped(self.state, True)

        self.previous = np.array([
            coordinate.getValue(self.state) for coordinate in self.coordinates
        ])
        self.alignment = np.asarray(alignment, dtype=float)
        self.depth_sign = depth_sign

    @staticmethod
    def _measurement_from_person(person) -> ArmMeasurement:
        points = np.asarray(person['keypoints_3d'], dtype=float)
        shoulder = points[6]
        return ArmMeasurement(
            elbow=points[8] - shoulder,
            wrist=points[10] - shoulder,
            index_mcp=points[117] - shoulder,
            pinky_mcp=points[129] - shoulder,
        )

    def estimate(self, measurement: ArmMeasurement) -> np.ndarray | None:
        self.model.realizePosition(self.state)
        shoulder = self.markers[0].getLocationInGround(self.state)
        origin = np.array([shoulder[i] for i in range(3)])
        points = np.array([measurement.elbow, measurement.wrist,
                           measurement.index_mcp, measurement.pinky_mcp])
        targets = origin + (points * [1, 1, self.depth_sign]) @ self.alignment.T

        row = osim.RowVectorVec3(5)
        row[0] = osim.Vec3(*origin)
        for i, point in enumerate(targets, 1):
            row[i] = osim.Vec3(*point)
        table = osim.TimeSeriesTableVec3()
        table.setColumnLabels(self.marker_names)
        table.appendRow(0.0, row)
        reference = osim.MarkersReference(table, osim.SetMarkerWeights())
        solver = osim.InverseKinematicsSolver(
            self.model, reference, osim.SimTKArrayCoordinateReference())
        solver.setAdvanceTimeFromReference(False)
        solver.setAccuracy(1e-5)
        try:
            solver.assemble(self.state)
        except RuntimeError:
            for coordinate, value in zip(self.coordinates, self.previous):
                coordinate.setValue(self.state, float(value), False)
            return None

        self.previous = np.array([
            coordinate.getValue(self.state) for coordinate in self.coordinates
        ])
        return self.previous.copy()

    def estimate_from_people(self, people) -> np.ndarray | None:
        if not people:
            return None
        return self.estimate(self._measurement_from_person(people[0]))
