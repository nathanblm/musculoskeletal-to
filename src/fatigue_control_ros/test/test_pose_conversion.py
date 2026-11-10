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
    pose2d = SimpleNamespace(dataset_meta={
        'dataset_name': 'coco', 'skeleton_links': [(5, 7), (7, 9)],
    })
    pose3d = SimpleNamespace(
        dataset_meta={'dataset_name': 'h36m'},
        cfg=SimpleNamespace(test_dataloader=SimpleNamespace(dataset={'seq_len': 1})),
    )
    monkeypatch.setattr(core, 'init_detector', lambda *args, **kwargs: detector)
    monkeypatch.setattr(core, 'adapt_mmdet_pipeline', lambda cfg: cfg)
    models = iter([pose2d, pose3d])
    monkeypatch.setattr(core, 'init_model', lambda *args, **kwargs: next(models))
    return core.MMPosePipeline({'device': 'cpu'})


def mock_detection(monkeypatch, empty=False):
    """Install repeatable detector and 2D estimator outputs."""
    points = np.arange(34, dtype=np.float32).reshape(1, 17, 2)
    scores = np.linspace(0.5, 1, 17, dtype=np.float32)[None]
    boxes = np.array([[0., 0., 64., 48.]], dtype=np.float32)
    sample = PoseDataSample()
    sample.gt_instances = InstanceData()
    sample.pred_instances = InstanceData(
        keypoints=points.copy(), keypoint_scores=scores.copy(), bboxes=boxes)
    detection = InstanceData(
        bboxes=boxes[:0] if empty else boxes,
        labels=np.array([], dtype=int) if empty else np.array([0]),
        scores=np.array([]) if empty else np.array([0.9]),
    )
    monkeypatch.setattr(core, 'inference_detector',
                        lambda *args: SimpleNamespace(pred_instances=detection))
    topdown = Mock(side_effect=lambda *args: [sample])
    monkeypatch.setattr(core, 'inference_topdown', topdown)
    lifter = Mock(return_value=[SimpleNamespace(
        pred_instances=SimpleNamespace(keypoints=np.zeros((1, 1, 17, 3))))])
    monkeypatch.setattr(core, 'inference_pose_lifter_model', lifter)
    return points, scores, lifter, topdown


def test_lifting_preserves_coco_overlay_and_output(pipeline, monkeypatch):
    """H36M conversion must never reorder COCO overlay or JSON keypoints."""
    points, scores, lifter, _ = mock_detection(monkeypatch)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = pipeline.process(image, 1)
    converted = lifter.call_args.args[1][0][0].pred_instances.keypoints
    np.testing.assert_allclose(converted[0, 0], points[0, [11, 12]].mean(axis=0))
    assert not np.shares_memory(converted, pipeline.previous[0].pred_instances.keypoints)
    np.testing.assert_array_equal(pipeline.previous[0].pred_instances.keypoints, points)
    np.testing.assert_array_equal(frame.people[0]['keypoints_2d'], points[0])
    np.testing.assert_array_equal(frame.people[0]['scores_2d'], scores[0])
    assert np.any(frame.overlay)
    assert not np.any(image)


def test_empty_detection_does_not_infer_whole_frame(pipeline, monkeypatch):
    """An empty person list must not trigger top-down fallback inference."""
    _, _, lifter, topdown = mock_detection(monkeypatch, empty=True)
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = pipeline.process(image, 1)
    assert frame.people == []
    np.testing.assert_array_equal(frame.overlay, image)
    topdown.assert_not_called()
    lifter.assert_not_called()


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
    assert payload['joints_2d'] == 'COCO-17'
    assert payload['joints_3d'] == core.H36M_JOINTS
    assert node.image_pub.publish.call_args.args[0].header == msg.header
