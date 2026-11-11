"""Estimate OpenSim arm coordinates from H36M joint centres."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


H36M_JOINT_INDEX = {
    "right_shoulder": 14,
    "right_elbow": 15,
    "right_wrist": 16,
}

# motionBERT output axis are:
# +X -> right in input image
# +Y -> down in input image
# +Z -> away from camera in input image

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
    """Shoulder-relative arm joint centres in camera coordinates, in metres."""

    elbow: np.ndarray
    wrist: np.ndarray


class OpenSimArmJointState:
    """Fit model coordinates to shoulder-relative joint-centre positions.

    The model must expose a GlenoHumeral, elbow, and rc joint. The parent
    frame of each joint is used as its joint centre, so no additional OpenSim
    markers are required.

    Measurements use MotionBERT camera axes. The default alignment assumes
    an upright camera facing the subject; pass a camera-to-model rotation
    matrix for other camera placements (identity for model-frame inputs).
    depth_sign optionally reverses inferred Z before applying that rotation.

    For a two-coordinate shoulder and one-coordinate elbow, shoulder fitting
    uses only upper-arm direction, and elbow fitting uses the inter-segment
    angle. Wrist mismatch therefore cannot move the fitted shoulder.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        coordinate_names: tuple[str, ...] | None = None,
        joint_names: tuple[str, str, str] = (
            "GlenoHumeral",
            "elbow",
            "rc",
        ),
        alignment: np.ndarray | None = None,
        depth_sign: float = 1.0,
        initial_coordinates: np.ndarray | None = None,
    ):
        import opensim as osim

        self.model = osim.Model(str(model_path))
        self.state = self.model.initSystem()
        self.joints = tuple(
            self.model.getJointSet().get(name) for name in joint_names)
        if any(joint is None for joint in self.joints):
            raise ValueError(
                f"Model is missing one of the joints {joint_names}")

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
        self.shoulder_indices = [
            i for i, coordinate in enumerate(self.coordinates)
            if coordinate.getJoint().getName() == self.joints[0].getName()
        ]
        self.elbow_indices = [
            i for i, coordinate in enumerate(self.coordinates)
            if coordinate.getJoint().getName() == self.joints[1].getName()
        ]
        self.separate_elbow = (
            len(self.coordinates) == 3
            and len(self.shoulder_indices) == 2
            and len(self.elbow_indices) == 1
        )

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

        if not np.all(np.isfinite(points)):
            return None

        shoulder = points[H36M_JOINT_INDEX["right_shoulder"]]
        elbow = points[H36M_JOINT_INDEX["right_elbow"]] - shoulder
        wrist = points[H36M_JOINT_INDEX["right_wrist"]] - shoulder
        if min(np.linalg.norm(elbow), np.linalg.norm(wrist - elbow)) < 1e-8:
            return None
        return ArmMeasurement(
            elbow=np.asarray(elbow, dtype=float),
            wrist=np.asarray(wrist, dtype=float),
        )

    def measurement_from_people(self, people) -> ArmMeasurement | None:
        """Use the first person for single-person experiments."""
        if not people:
            return None
        return self._measurement_from_person(people[0])

    def _joint_centres(
        self, coordinates: np.ndarray,
    ) -> tuple[np.ndarray, ...]:
        for coordinate, value in zip(self.coordinates, coordinates):
            coordinate.setValue(self.state, float(value), False)
        self.model.realizePosition(self.state)
        return tuple(
            np.array([
                joint.getParentFrame().getPositionInGround(self.state)[axis]
                for axis in range(3)
            ])
            for joint in self.joints
        )

    def _model_measurement(self, coordinates: np.ndarray) -> np.ndarray:
        shoulder, elbow, wrist = self._joint_centres(coordinates)
        return np.concatenate((
            elbow - shoulder,
            wrist - shoulder,
        ))

    def estimate(self, measurement: ArmMeasurement) -> np.ndarray:
        """Return generalized coordinates in the model's coordinate order."""
        target = np.concatenate((
            self.alignment @ (measurement.elbow * [1, 1, self.depth_sign]),
            self.alignment @ (measurement.wrist * [1, 1, self.depth_sign]),
        ))

        if self.separate_elbow:
            return self._estimate_separate_elbow(target)

        result = least_squares(
            lambda coordinates: self._model_measurement(coordinates) - target,
            x0=self.previous,
            bounds=(self.lower_bounds, self.upper_bounds),
        )
        if not result.success:
            raise RuntimeError(f"OpenSim arm IK failed: {result.message}")

        self.previous = result.x.copy()
        return self.previous.copy()

    @staticmethod
    def _bend_angle(upper_arm, forearm):
        """Measure flexion between segments, independent of bending plane."""
        return np.arctan2(
            np.linalg.norm(np.cross(upper_arm, forearm)),
            np.dot(upper_arm, forearm),
        )

    def _estimate_separate_elbow(self, target):
        """Fit a 3DOF arm without letting the wrist pull on the shoulder."""
        elbow, wrist = target.reshape(2, 3)
        if min(np.linalg.norm(elbow), np.linalg.norm(wrist - elbow)) < 1e-8:
            raise ValueError('Arm segments must have nonzero length')
        direction = elbow / np.linalg.norm(elbow)
        coordinates = self.previous.copy()

        def shoulder_residual(values):
            coordinates[self.shoulder_indices] = values
            upper_arm = self._model_measurement(coordinates)[:3]
            return upper_arm / np.linalg.norm(upper_arm) - direction

        shoulder = least_squares(
            shoulder_residual,
            x0=self.previous[self.shoulder_indices],
            bounds=(self.lower_bounds[self.shoulder_indices],
                    self.upper_bounds[self.shoulder_indices]),
        )
        if not shoulder.success:
            raise RuntimeError(
                f'OpenSim shoulder IK failed: {shoulder.message}')
        coordinates[self.shoulder_indices] = shoulder.x
        target_angle = self._bend_angle(elbow, wrist - elbow)
        elbow_index = self.elbow_indices[0]

        def elbow_error(value):
            coordinates[elbow_index] = value
            upper_arm, endpoint = self._model_measurement(
                coordinates).reshape(2, 3)
            angle = self._bend_angle(upper_arm, endpoint - upper_arm)
            return float((angle - target_angle) ** 2)

        # A bounded scalar solve can leave a fully extended starting pose;
        # least_squares can stall when initialized at the lower elbow bound.
        lower = self.lower_bounds[elbow_index]
        upper = self.upper_bounds[elbow_index]
        elbow_fit = minimize_scalar(
            elbow_error, bounds=(lower, upper), method='bounded',
            options={'xatol': 1e-10},
        )
        if not elbow_fit.success:
            raise RuntimeError(f'OpenSim elbow IK failed: {elbow_fit.message}')
        candidates = (lower, elbow_fit.x, upper)
        coordinates[elbow_index] = min(candidates, key=elbow_error)
        self._model_measurement(coordinates)
        self.previous = coordinates.copy()
        return self.previous.copy()

    def estimate_from_people(self, people) -> np.ndarray | None:
        measurement = self.measurement_from_people(people)
        return None if measurement is None else self.estimate(measurement)
