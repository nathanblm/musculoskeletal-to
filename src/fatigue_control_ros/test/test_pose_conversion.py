"""Pipeline regression checks and ROS adapter contract checks."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

from cv_bridge import CvBridge
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample
import numpy as np
import pytest

import mmpose_pipeline as core
from fatigue_control_ros import pose_estimator as ros


@pytest.fixture
def pipeline(monkeypatch):
    """Use real MMEngine samples without loading weights or starting ROS."""
    detector = SimpleNamespace(cfg={})
    pose3d = SimpleNamespace(test_cfg={})
    monkeypatch.setattr(core, 'init_detector', lambda *args, **kwargs: detector)
    monkeypatch.setattr(core, 'adapt_mmdet_pipeline', lambda cfg: cfg)
    monkeypatch.setattr(core, 'init_model', lambda *args, **kwargs: pose3d)
    return core.MMPosePipeline({'device': 'cpu'})


def mock_detection(monkeypatch, empty=False):
    """Install repeatable detector and 2D estimator outputs."""
    points = (np.arange(266, dtype=np.float32).reshape(1, 133, 2) % 40)
    scores = np.linspace(0.5, 1, 133, dtype=np.float32)[None]
    boxes = np.array([[0., 0., 64., 48.]], dtype=np.float32)
    sample = PoseDataSample()
    sample.gt_instances = InstanceData()
    sample.pred_instances = InstanceData(
        keypoints=np.arange(399, dtype=np.float32).reshape(1, 133, 3) / 100,
        transformed_keypoints=points.copy(),
        keypoint_scores=scores.copy(), bboxes=boxes)
    detection = InstanceData(
        bboxes=boxes[:0] if empty else boxes,
        labels=np.array([], dtype=int) if empty else np.array([0]),
        scores=np.array([]) if empty else np.array([0.9]),
    )
    monkeypatch.setattr(core, 'inference_detector',
                        lambda *args: SimpleNamespace(pred_instances=detection))
    topdown = Mock(side_effect=lambda *args: [sample])
    monkeypatch.setattr(core, 'inference_topdown', topdown)
    return points, scores, topdown


def test_wholebody_coordinates_and_overlay(pipeline, monkeypatch):
    """Keep all fingers in image pixels and pelvis-relative metric 3D."""
    points, scores, _ = mock_detection(monkeypatch)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = pipeline.process(image, 1)
    person = frame.people[0]
    np.testing.assert_array_equal(person['keypoints_2d'], points[0])
    np.testing.assert_array_equal(person['scores_2d'], scores[0])
    np.testing.assert_array_equal(person['scores_3d'], scores[0])
    joints = np.array(person['keypoints_3d'])
    assert joints.shape == (133, 3)
    np.testing.assert_allclose(joints[[11, 12]].mean(axis=0), 0, atol=1e-7)
    original = pipeline.previous[0].pred_instances.keypoints[0]
    np.testing.assert_allclose(joints[132] - joints[112], original[132] - original[112])
    assert np.any(frame.overlay)
    assert not np.any(image)


def test_empty_detection_does_not_infer_whole_frame(pipeline, monkeypatch):
    """An empty person list must not trigger top-down fallback inference."""
    _, _, topdown = mock_detection(monkeypatch, empty=True)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = pipeline.process(image, 1)
    assert frame.people == []
    np.testing.assert_array_equal(frame.overlay, image)
    topdown.assert_not_called()


@pytest.mark.parametrize('timestamp,shape', [
    (2, (48, 64, 3)), (1, (48, 64, 3)),
    (2_000_000_000, (48, 64, 3)), (2, (96, 128, 3)),
])
def test_tracking_and_discontinuity_reset(pipeline, monkeypatch, timestamp, shape):
    """Track continuous frames and start a new identity after discontinuities."""
    mock_detection(monkeypatch)
    first = pipeline.process(np.zeros((48, 64, 3), dtype=np.uint8), 1)
    second = pipeline.process(np.zeros(shape, dtype=np.uint8), timestamp)
    continuous = timestamp == 2 and shape == (48, 64, 3)
    assert (first.people[0]['track_id'] == second.people[0]['track_id']) == continuous


def test_ros_adapter_preserves_headers_and_schema():
    """ROS transport consumes plain pipeline outputs without handling ML samples."""
    bridge = CvBridge()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    msg = bridge.cv2_to_imgmsg(image, 'bgr8')
    msg.header.stamp.sec, msg.header.stamp.nanosec = 7, 123
    msg.header.frame_id = 'camera'
    frame = core.PoseFrame(people=[{'track_id': 42}], overlay=image)
    node = SimpleNamespace(
        bridge=bridge, pipeline=Mock(), pose_pub=Mock(), image_pub=Mock(),
        frame_count=0,
    )
    node.pipeline.process.return_value = frame
    node.publish_poses = lambda header, people: ros.PoseEstimator.publish_poses(
        node, header, people)
    ros.PoseEstimator.image_callback(node, msg)
    assert node.pipeline.process.call_args.args[1] == 7_000_000_123
    payload = json.loads(node.pose_pub.publish.call_args.args[0].data)
    assert payload['people'] == frame.people
    assert payload['stamp'] == {'sec': 7, 'nanosec': 123}
    assert payload['image_frame_id'] == 'camera'
    assert payload['joints_2d'] == core.WHOLEBODY_JOINTS
    assert payload['joints_3d'] == core.WHOLEBODY_JOINTS
    assert payload['schema'] == 'COCO-WholeBody-133'
    assert node.image_pub.publish.call_args.args[0].header == msg.header


def test_finger_confidence_does_not_keep_an_invalid_body(pipeline, monkeypatch):
    mock_detection(monkeypatch)
    sample = core.inference_topdown()[0]
    sample.pred_instances.keypoint_scores[0, :17] = 0
    assert pipeline.process(np.zeros((48, 64, 3), dtype=np.uint8), 1).people == []


def test_unconverged_frame_skips_joint_state_and_keeps_publishing_images():
    bridge = CvBridge()
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    msg = bridge.cv2_to_imgmsg(image, 'bgr8')
    node = SimpleNamespace(
        bridge=bridge, pipeline=Mock(), pose_pub=Mock(), image_pub=Mock(),
        frame_count=0, arm_state=Mock(), joint_state_pub=Mock(),
        publish_poses=Mock(), get_logger=Mock(return_value=Mock()),
    )
    node.pipeline.process.return_value = core.PoseFrame(
        people=[{'track_id': 1}], overlay=image)
    node.arm_state.coordinate_names = ['axial_rot']
    node.arm_state.estimate_from_people.side_effect = [None, np.array([0.4])]
    ros.PoseEstimator.image_callback(node, msg)
    node.joint_state_pub.publish.assert_not_called()
    node.image_pub.publish.assert_called_once()
    ros.PoseEstimator.image_callback(node, msg)
    node.joint_state_pub.publish.assert_called_once()
    assert node.image_pub.publish.call_count == 2
