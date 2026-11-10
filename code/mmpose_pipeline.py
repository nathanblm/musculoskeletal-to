from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import cv2
import mmpose
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (
    _track_by_iou, convert_keypoint_definition, extract_pose_sequence,
    inference_pose_lifter_model, inference_topdown, init_model,
)
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
import numpy as np
import torch


MMPOSE_ROOT = Path(mmpose.__file__).parent / '.mim'
WEIGHTS_ROOT = 'https://download.openmmlab.com/mmpose/v1/'
H36M_JOINTS = [
    'pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip',
    'left_knee', 'left_ankle', 'spine', 'thorax', 'neck', 'head',
    'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder',
    'right_elbow', 'right_wrist',
]


def default_model_settings():
    return {
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'detector_config': str(MMPOSE_ROOT / 'demo/mmdetection_cfg/'
                               'rtmdet_m_640-8xb32_coco-person.py'),
        'detector_checkpoint': WEIGHTS_ROOT + 'projects/rtmposev1/'
                               'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        'pose2d_config': str(MMPOSE_ROOT / 'configs/body_2d_keypoint/'
                             'rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py'),
        'pose2d_checkpoint': WEIGHTS_ROOT + 'projects/rtmposev1/'
                             'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth',
        'pose3d_config': str(MMPOSE_ROOT / 'configs/body_3d_keypoint/'
                             'motionbert/h36m/'
                             'motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py'),
        'pose3d_checkpoint': WEIGHTS_ROOT + 'body_3d_keypoint/pose_lift/'
                             'h36m/motionbert_ft_h36m-d80af323_20230531.pth',
        'bbox_threshold': 0.3,
        'nms_threshold': 0.3,
        'tracking_threshold': 0.3,
        'keypoint_threshold': 0.3,
        'normalize_pose_2d': True,
    }


@dataclass
class PoseFrame:

    people: list
    overlay: np.ndarray


class MMPosePipeline:

    def __init__(self, settings=None):
        self.settings = default_model_settings()
        if settings is not None:
            self.settings.update(settings)
        p = self.settings

        self.detector = init_detector(
            p['detector_config'], p['detector_checkpoint'], device=p['device'])
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # Stage 2: RTMPose-M, 17 COCO joints, cropped using detected boxes.
        self.pose2d = init_model(
            p['pose2d_config'], p['pose2d_checkpoint'], device=p['device'])

        # Stage 3: MotionBERT, 17 Human3.6M joints.
        self.pose3d = init_model(
            p['pose3d_config'], p['pose3d_checkpoint'], device=p['device'])

        # construct temporal window similar to what MotionBert was trained on
        dataset = self.pose3d.cfg.test_dataloader.dataset
        self.seq_len = dataset.get('seq_len', 1)
        self.seq_step = dataset.get('seq_step', 1)
        self.causal = dataset.get('causal', False)
        self.history = deque(maxlen=(self.seq_len - 1) * self.seq_step + 1)
        self.previous = []
        self.next_id = 0
        self.last_stamp = None
        self.last_shape = None

    def process(self, image, timestamp_ns):
        shape = image.shape[:2]

        # reset temporal history if the timestamp is not strictly increasing, went forward over 1 second, or the image shape changed
        if (self.last_stamp is not None and             
                (timestamp_ns <= self.last_stamp or timestamp_ns - self.last_stamp > 10**9
                 or shape != self.last_shape)):
            self.previous = []
            self.history.clear()
        self.last_stamp, self.last_shape = timestamp_ns, shape
        results = self._detect_2d(image)
        tracked = self._track(results)
        converted = self._convert_for_lifting(tracked)
        lifted = self._lift(converted, image.shape)

        return PoseFrame(
            people=self._format_people(tracked, lifted),
            overlay=self.draw_overlay(image, tracked),
        )

    def _detect_2d(self, image):
        # create bounding boxes for each person in the image
        detections = inference_detector(self.detector, image).pred_instances.cpu().numpy()

        # keep only the first class (person) and filter by score threshold
        keep = ((detections.labels == 0) &
                (detections.scores > self.settings['bbox_threshold']))
        boxes = detections.bboxes[keep]
        scores = detections.scores[keep]

        # apply non-maximum suppression to remove overlapping boxes
        boxes = boxes[nms(np.column_stack((boxes, scores)),
                          self.settings['nms_threshold'])] if len(boxes) else boxes
        
        # infer the 2D joint
        return inference_topdown(self.pose2d, image, boxes) if len(boxes) else []

    def _track(self, results):
        previous = list(self.previous)
        tracked = []
        for result in results:
            result = result.cpu().numpy()

            # skip any person with fewer than 3 keypoints above the threshold
            if np.count_nonzero(result.pred_instances.keypoint_scores[0] >
                                self.settings['keypoint_threshold']) < 3:
                continue

            # track persons across frames according to bounding box overlap
            track_id, previous, _ = _track_by_iou(
                result, previous, self.settings['tracking_threshold'])

            # if no match was found, assign a new track ID
            if track_id == -1:
                track_id = self.next_id
                self.next_id += 1

            result.set_field(track_id, 'track_id')
            tracked.append(result)
        self.previous = tracked

        return tracked

    def _convert_for_lifting(self, tracked):
        converted = []
        for result in tracked:
            sample = deepcopy(result)

            # convert COCO-17 keypoints to Human3.6M-17 keypoints for lifting
            sample.pred_instances.keypoints = convert_keypoint_definition(
                result.pred_instances.keypoints,
                self.pose2d.dataset_meta['dataset_name'],   # COCO-17
                self.pose3d.dataset_meta['dataset_name'])   # Human3.6M-17
            converted.append(sample)
        self.history.append(converted)

        return converted

    def _lift(self, converted, image_shape):
        lifted = []
        if converted:
            # extract the temporal window for lifting and pad it to seq_len
            # by repeating boundary frames when history is not yet long enough.
            sequence = extract_pose_sequence(
                list(self.history), 
                frame_idx=len(self.history) - 1,
                causal=self.causal,                 # we only use past frames for lifting
                seq_len=self.seq_len, 
                step=self.seq_step
            )

            lifted = inference_pose_lifter_model(
                self.pose3d, 
                sequence, 
                with_track_id=True,
                image_size=(image_shape[1], image_shape[0]),
                norm_pose_2d=self.settings['normalize_pose_2d']
            )

        return lifted

    @staticmethod
    def _format_people(tracked, lifted):
        people = []
        for result, prediction in zip(tracked, lifted):
            joints = np.asarray(prediction.pred_instances.keypoints).reshape(-1, 17, 3)[0]
            people.append({
                'track_id': int(result.track_id),
                'bbox_xyxy': result.pred_instances.bboxes[0].tolist(),
                'keypoints_2d': result.pred_instances.keypoints[0].tolist(),
                'scores_2d': result.pred_instances.keypoint_scores[0].tolist(),
                'keypoints_3d': joints.tolist(),
            })
        return people

    def draw_overlay(self, image, results):
        """Return a BGR image with COCO skeletons without changing the input."""
        canvas = image.copy()
        links = self.pose2d.dataset_meta.get('skeleton_links', [])
        for result in results:
            points = result.pred_instances.keypoints[0]
            scores = result.pred_instances.keypoint_scores[0]
            visible = scores > self.settings['keypoint_threshold']
            for a, b in links:
                if visible[a] and visible[b]:
                    cv2.line(canvas, tuple(points[a].astype(int)),
                             tuple(points[b].astype(int)), (0, 255, 0), 2)
            for point in points[visible]:
                cv2.circle(canvas, tuple(point.astype(int)), 3, (0, 0, 255), -1)
        return canvas
