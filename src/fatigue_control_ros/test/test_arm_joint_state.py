"""Regression checks for camera-frame arm fitting with native OpenSim."""

from pathlib import Path

import numpy as np
import pytest

from arm_joint_state import ArmMeasurement, OpenSimArmJointState


MODEL_PATH = (Path(__file__).resolve().parents[3]
              / 'musculoskeletal_models/tsm_3DOF/tsm_3DOF.osim')


@pytest.mark.parametrize('depth_sign', [-1.0, 1.0])
@pytest.mark.parametrize('plane,elevation,flexion', [
    (0.3, 0.1, 0.1),    # arm down
    (0.3, 2.5, 0.1),    # arm raised
    (1.2, 1.0, 0.8),    # forward, bent elbow
    (-0.8, 1.5, 1.2),   # another shoulder plane, bent elbow
])
def test_camera_pose_recovers_model_pose(
        plane, elevation, flexion, depth_sign):
    """Camera-frame landmarks must reproduce their known model posture."""
    arm = OpenSimArmJointState(MODEL_PATH, depth_sign=depth_sign)
    known = dict(plane_elv=plane, shoulder_elv=elevation,
                 elbow_flexion=flexion)
    q = np.array([known[name] for name in arm.coordinate_names])
    shoulder, elbow, wrist = arm._joint_centres(q)

    # Independent inverse axis mapping: a front-facing subject's +right
    # is camera -X, +up is camera -Y, and +forward is camera -Z.
    points = np.zeros((17, 3))
    for index, point in [(14, shoulder), (15, elbow), (16, wrist)]:
        camera_point = -point[[2, 1, 0]]
        camera_point[2] *= depth_sign
        points[index] = camera_point + [0.2, -0.1, 2.0]
    fitted = arm.estimate_from_people([{'keypoints_3d': points.tolist()}])
    actual = arm._model_measurement(fitted)
    expected = np.concatenate((elbow - shoulder, wrist - shoulder))
    np.testing.assert_allclose(actual, expected, atol=1e-5)
    np.testing.assert_allclose(fitted, q, atol=1e-4)
    camera_depth = points[16, 2] - points[14, 2]
    assert actual[3] == pytest.approx(-depth_sign * camera_depth, abs=1e-5)
    if elevation > 2:
        assert points[15, 1] < points[14, 1]  # up in camera coordinates
        assert actual[1] > 0                 # up in model coordinates
    elif elevation < 0.2:
        assert points[15, 1] > points[14, 1]
        assert actual[1] < 0


def test_explicit_model_frame_alignment():
    """Identity remains available for measurements already in model axes."""
    arm = OpenSimArmJointState(MODEL_PATH, alignment=np.eye(3))
    expected = arm.previous.copy()
    elbow, wrist = arm._model_measurement(expected).reshape(2, 3)
    fitted = arm.estimate(ArmMeasurement(elbow, wrist))
    np.testing.assert_allclose(fitted, expected, atol=1e-6)


@pytest.mark.parametrize('plane_rotation', [0.0, 0.5, 1.0, -1.0])
def test_forearm_motion_does_not_move_shoulder(plane_rotation):
    """Forearm-plane and length differences cannot pull the GH joint."""
    arm = OpenSimArmJointState(MODEL_PATH, alignment=np.eye(3))
    fixed_shoulder = np.array([0.4, 1.1])
    for flexion in (0.1, 0.8, 1.6, 0.4):
        q = np.array([*fixed_shoulder, flexion])
        elbow, wrist = arm._model_measurement(q).reshape(2, 3)
        axis = elbow / np.linalg.norm(elbow)
        forearm = wrist - elbow
        # Rotate only the forearm around the fixed upper arm. The 3DOF
        # model cannot reproduce this bending plane without moving the GH.
        forearm = (
            forearm * np.cos(plane_rotation)
            + np.cross(axis, forearm) * np.sin(plane_rotation)
            + axis * np.dot(axis, forearm) * (1 - np.cos(plane_rotation))
        )
        # Subject and model need not have matching segment lengths.
        elbow = 1.2 * elbow
        fitted = arm.estimate(ArmMeasurement(elbow, elbow + 0.9 * forearm))
        np.testing.assert_allclose(fitted[:2], fixed_shoulder, atol=1e-5)
        assert fitted[2] == pytest.approx(flexion, abs=1e-5)


def test_four_dof_still_fits_axial_rotation():
    """The 4DOF model retains its simultaneous fit including axial rotation."""
    model = MODEL_PATH.parents[1] / 'tsm_4DOF/tsm_4DOF.osim'
    arm = OpenSimArmJointState(model, alignment=np.eye(3))
    assert not arm.separate_elbow
    q = np.array([0.4, 1.1, -0.5, 0.8])
    elbow, wrist = arm._model_measurement(q).reshape(2, 3)
    fitted = arm.estimate(ArmMeasurement(elbow, wrist))
    np.testing.assert_allclose(fitted, q, atol=1e-4)


def test_collapsed_segments_are_skipped():
    """Degenerate landmarks have no usable segment direction or bend angle."""
    arm = OpenSimArmJointState(MODEL_PATH)
    assert arm.estimate_from_people([
        {'keypoints_3d': np.zeros((17, 3)).tolist()},
    ]) is None


@pytest.mark.parametrize('rotation', [
    np.diag([1., -1., 1.]), np.zeros((3, 3)), np.full((3, 3), np.nan),
])
def test_invalid_alignment_rejected(rotation):
    """Reject reflections and invalid matrices rather than distorting poses."""
    with pytest.raises(ValueError, match='proper rotation'):
        OpenSimArmJointState(MODEL_PATH, alignment=rotation)
