"""Estimate OpenSim arm coordinates from RTMW3D arm and MCP landmarks."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


# RTMW3D camera axes: X right, Y down, Z away from the camera.
# OpenSim ground axes are:
# +X -> forward from the subject
# +Y -> up from the subject
# +Z -> right from the subject

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
    """Fit elbow, wrist and both MCP markers relative to the shoulder."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        coordinate_names: tuple[str, ...] | None = None,
        alignment: np.ndarray | None = None,
        depth_sign: float = 1.0,
        initial_coordinates: np.ndarray | None = None,
    ):
        import opensim as osim

        self.model = osim.Model(str(model_path))
        self.state = self.model.initSystem()
        self.markers = tuple(
            self.model.getMarkerSet().get(name)
            for name in ('shoulder', 'elbow', 'wrist', 'index_mcp', 'pinky_mcp'))

        coordinates = self.model.getCoordinateSet()
        if coordinate_names is None:
            coordinate_names = tuple(
                coordinates.get(i).getName()
                for i in range(coordinates.getSize())
                if not coordinates.get(i).get_locked()
            )
        self.coordinate_names = tuple(coordinate_names)
        if not self.coordinate_names:
            raise ValueError("The model has no free coordinates to estimate")

        self.coordinates = tuple(
            coordinates.get(name) for name in self.coordinate_names)
        if any(coordinate is None for coordinate in self.coordinates):
            raise ValueError(
                f"Unknown coordinate in {self.coordinate_names}")
        self.lower_bounds = np.array([
            coordinate.getRangeMin() for coordinate in self.coordinates
        ])
        self.upper_bounds = np.array([
            coordinate.getRangeMax() for coordinate in self.coordinates
        ])
        self.previous = (
            np.asarray(initial_coordinates, dtype=float).copy()
            if initial_coordinates is not None
            else np.array([
                coordinate.getDefaultValue() for coordinate in self.coordinates
            ])
        )
        if self.previous.shape != (len(self.coordinates),):
            raise ValueError("initial_coordinates has the wrong shape")

        self.alignment = (
            CAMERA_TO_OPENSIM.copy() if alignment is None
            else np.asarray(alignment, dtype=float))
        if self.alignment.shape != (3, 3):
            raise ValueError("alignment must have shape (3, 3)")
        if (not np.all(np.isfinite(self.alignment))
                or not np.allclose(
                    self.alignment.T @ self.alignment, np.eye(3))
                or not np.isclose(np.linalg.det(self.alignment), 1.0)):
            raise ValueError(
                "alignment must be a finite proper rotation matrix")
        if depth_sign not in (-1.0, 1.0):
            raise ValueError("depth_sign must be -1 or 1")
        self.depth_sign = float(depth_sign)

    @staticmethod
    def _measurement_from_person(person) -> ArmMeasurement | None:
        """Extract a right-arm measurement from one tracked person."""
        points = np.asarray(person["keypoints_3d"], dtype=float)

        shoulder = points[6]
        return ArmMeasurement(
            elbow=points[8] - shoulder,
            wrist=points[10] - shoulder,
            index_mcp=points[117] - shoulder,
            pinky_mcp=points[129] - shoulder,
        )

    def measurement_from_people(self, people) -> ArmMeasurement | None:
        """Use the first person for single-person experiments."""
        if not people:
            return None
        return self._measurement_from_person(people[0])

    def _model_measurement(self, coordinates: np.ndarray) -> np.ndarray:
        for coordinate, value in zip(self.coordinates, coordinates):
            coordinate.setValue(self.state, float(value), False)
        self.model.realizePosition(self.state)
        points = np.array([
            [marker.getLocationInGround(self.state)[axis] for axis in range(3)]
            for marker in self.markers
        ])
        return (points[1:] - points[0]).ravel()

    def estimate(self, measurement: ArmMeasurement) -> np.ndarray | None:
        """Return generalized coordinates in the model's coordinate order."""
        points = np.array([measurement.elbow, measurement.wrist,
                           measurement.index_mcp, measurement.pinky_mcp])
        target = ((points * [1, 1, self.depth_sign]) @ self.alignment.T).ravel()

        result = least_squares(
            lambda coordinates: self._model_measurement(coordinates) - target,
            x0=self.previous,
            bounds=(self.lower_bounds, self.upper_bounds),
        )
        if not result.success:
            self._model_measurement(self.previous)
            return None

        self.previous = result.x.copy()
        return self.previous.copy()

    def estimate_from_people(self, people) -> np.ndarray | None:
        measurement = self.measurement_from_people(people)
        return None if measurement is None else self.estimate(measurement)
