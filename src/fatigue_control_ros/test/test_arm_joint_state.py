"""Recover known model poses from the five whole-body arm landmarks."""

from pathlib import Path

import numpy as np
import pytest

from arm_joint_state import ArmMeasurement, OpenSimArmJointState


MODEL_PATH = (Path(__file__).resolve().parents[3]
              / 'musculoskeletal_models/tsm_hand_4DOF/tsm_4DOF_arm26_hand.osim')


@pytest.mark.parametrize('depth_sign', [-1.0, 1.0])
@pytest.mark.parametrize('pose', [
    [0.3, 0.1, -0.4, 0.1],
    [0.3, 2.3, 0.4, 0.1],
    [1.2, 1.0, -0.5, 0.8],
    [-0.8, 1.5, 0.7, 1.2],
    [0.4, 1.1, -0.6, 0.0],  # MCPs constrain rotation with a straight elbow.
])
def test_camera_landmarks_recover_model_pose(pose, depth_sign):
    arm = OpenSimArmJointState(MODEL_PATH, depth_sign=depth_sign)
    expected = np.array(pose)
    landmarks = arm._model_measurement(expected).reshape(4, 3)
    camera = (landmarks @ arm.alignment) * [1, 1, depth_sign]
    points = np.zeros((133, 3))
    points[6] = [0.2, -0.1, 2.0]
    points[[8, 10, 117, 129]] = camera + points[6]
    fitted = arm.estimate_from_people([{'keypoints_3d': points.tolist()}])
    np.testing.assert_allclose(fitted, expected, atol=3e-4)
    np.testing.assert_allclose(arm._model_measurement(fitted), landmarks.ravel(), atol=1e-5)


def test_both_mcps_are_in_the_ik_objective():
    arm = OpenSimArmJointState(MODEL_PATH, alignment=np.eye(3))
    pose = np.array([0.4, 1.1, -0.6, 0.0])
    targets = arm._model_measurement(pose).reshape(4, 3)
    fitted = arm.estimate(ArmMeasurement(*targets))
    twisted = pose.copy()
    twisted[2] += 0.5
    # Check the hand-marker residuals change with axial rotation.
    wrong_hand = arm._model_measurement(twisted).reshape(4, 3)[2:]
    assert np.linalg.norm(wrong_hand - targets[2:]) > 0.01
    np.testing.assert_allclose(fitted, pose, atol=3e-4)


def test_no_person_produces_no_joint_state():
    arm = OpenSimArmJointState(MODEL_PATH)
    assert arm.estimate_from_people([]) is None


@pytest.mark.parametrize('dof', [3, 4])
def test_default_elbow_bends_forward(dof):
    model = (MODEL_PATH.parents[1] / f'tsm_hand_{dof}DOF'
             / f'tsm_{dof}DOF_arm26_hand_scaled.osim')
    arm = OpenSimArmJointState(model)
    assert arm.model.getCoordinateSet().get('axial_rot').getDefaultValue() == 0
    straight = arm._model_measurement(arm.previous).reshape(4, 3)
    bent_pose = arm.previous.copy()
    bent_pose[arm.elbow_index] = 1.2
    bent = arm._model_measurement(bent_pose).reshape(4, 3)
    assert bent[1, 0] > straight[1, 0] + 0.15
    # Camera depth decreases when the wrist comes toward the camera.
    camera = bent @ arm.alignment
    assert camera[1, 2] < (straight @ arm.alignment)[1, 2]


@pytest.mark.parametrize('dof', [3, 4])
@pytest.mark.parametrize('elbow', [0.0, 1.2])
def test_lengths_and_hand_errors_do_not_change_elbow(dof, elbow):
    model = (MODEL_PATH.parents[1] / f'tsm_hand_{dof}DOF'
             / f'tsm_{dof}DOF_arm26_hand_scaled.osim')
    arm = OpenSimArmJointState(model, alignment=np.eye(3))
    # Change shoulder orientation between frames while holding elbow bending.
    for shoulder in [[0.4, 1.1], [-0.8, 1.5]]:
        pose = arm.previous.copy()
        pose[:2] = shoulder
        pose[arm.elbow_index] = elbow
        points = arm._model_measurement(pose).reshape(4, 3)
        distorted = points.copy()
        distorted[0] = 1.2 * points[0]
        distorted[1] = distorted[0] + 0.8 * (points[1] - points[0])
        distorted[2:] = distorted[1] + 1.3 * (points[2:] - points[1])
        fitted = arm.estimate(ArmMeasurement(*distorted))
        np.testing.assert_allclose(fitted, pose, atol=3e-4)
        # Inconsistent hand landmarks must not pull the elbow away from its bend.
        distorted[2:] += [0.04, -0.03, 0.02]
        fitted = arm.estimate(ArmMeasurement(*distorted))
        np.testing.assert_allclose(fitted[arm.elbow_index], elbow, atol=1e-8)


def test_unconverged_frame_keeps_previous_pose_and_next_frame_recovers(monkeypatch):
    from types import SimpleNamespace
    import arm_joint_state as core

    arm = OpenSimArmJointState(MODEL_PATH, alignment=np.eye(3))
    previous = arm.previous.copy()
    pose = np.array([0.4, 1.1, -0.6, 0.8])
    targets = arm._model_measurement(pose).reshape(4, 3)
    measurement = ArmMeasurement(*targets)
    solve = core.least_squares
    monkeypatch.setattr(core, 'least_squares', lambda *a, **k: SimpleNamespace(success=False))
    assert arm.estimate(measurement) is None
    np.testing.assert_array_equal(arm.previous, previous)
    np.testing.assert_allclose([c.getValue(arm.state) for c in arm.coordinates], previous)
    monkeypatch.setattr(core, 'least_squares', solve)
    np.testing.assert_allclose(arm.estimate(measurement), pose, atol=3e-4)
