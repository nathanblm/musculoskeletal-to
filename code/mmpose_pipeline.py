from pathlib import Path

import cv2
import mmpose
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import _track_by_iou, inference_topdown, init_model
from mmpose.datasets.datasets.utils import parse_pose_metainfo
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
import numpy as np
import torch


MMPOSE_ROOT = Path(mmpose.__file__).parent / '.mim'
WEIGHTS_ROOT = 'https://download.openmmlab.com/mmpose/v1/'
WHOLEBODY_META = parse_pose_metainfo(dict(
    from_file=str(MMPOSE_ROOT / 'configs/_base_/datasets/coco_wholebody.py')))
WHOLEBODY_JOINTS = [WHOLEBODY_META['keypoint_id2name'][i] for i in range(133)]


def default_model_settings():
    return {
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'detector_config': str(MMPOSE_ROOT / 'demo/mmdetection_cfg/'
                               'rtmdet_m_640-8xb32_coco-person.py'),
        'detector_checkpoint': WEIGHTS_ROOT + 'projects/rtmposev1/'
                               'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth',
        'pose3d_config': str(Path(__file__).parent / 'rtmpose3d/configs/'
                             'rtmw3d-l_8xb64_cocktail14-384x288.py'),
        'pose3d_checkpoint': WEIGHTS_ROOT + 'wholebody_3d_keypoint/rtmw3d/'
                             'rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth',
        'bbox_threshold': 0.3,
        'nms_threshold': 0.3,
        'tracking_threshold': 0.3,
        'keypoint_threshold': 0.3,
    }


class MMPosePipeline:

    def __init__(self, settings=None):
        self.settings = default_model_settings()
        if settings is not None:
            self.settings.update(settings)
        p = self.settings

        self.detector = init_detector(
            p['detector_config'], p['detector_checkpoint'], device=p['device'])
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # RTMW3D predicts all 133 body, face, foot and hand joints directly.
        self.pose3d = init_model(
            p['pose3d_config'], p['pose3d_checkpoint'], device=p['device'])
        self.pose3d.test_cfg['mode'] = '3d'
        self.previous = []
        self.next_id = 0
        self.last_stamp = None
        self.last_shape = None

    def process(self, image, timestamp_ns):
        shape = image.shape[:2]

        # Reset tracking after a timestamp discontinuity or resolution change.
        if (self.last_stamp is not None and
                (timestamp_ns <= self.last_stamp or timestamp_ns - self.last_stamp > 10**9
                 or shape != self.last_shape)):
            self.previous = []
        self.last_stamp, self.last_shape = timestamp_ns, shape
        results = self._detect_pose(image)
        tracked = self._track(results)

        return self._format_people(tracked)

    def _detect_pose(self, image):
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
        
        return inference_topdown(self.pose3d, image, boxes) if len(boxes) else []

    def _track(self, results):
        previous = list(self.previous)
        tracked = []
        for result in results:
            result = result.cpu().numpy()

            # Require body evidence; visible face/fingers alone are insufficient.
            if np.count_nonzero(result.pred_instances.keypoint_scores[0, :17] >
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

    @staticmethod
    def _format_people(tracked):
        people = []
        for result in tracked:
            pred = result.pred_instances

            # RTMW3D estimates landmark positions in camera coordinates.
            # for arm IK, we only need the arm geometry relative to the shoulder.
            # we use the shoulder as the origin so the landmarks match the OpenSim model
            # whose torso/shoulder is fixed during fitting.

            joints = np.asarray(pred.keypoints).reshape(-1, 133, 3)[0].copy()
            joints -= joints[6]
            people.append({
                'track_id': int(result.track_id),
                'keypoints_2d': pred.transformed_keypoints[0].tolist(),
                'scores_2d': pred.keypoint_scores[0].tolist(),
                'keypoints_3d': joints.tolist(),
            })
        return people

    def draw_overlay(self, image, people):
        canvas = image.copy()
        links = WHOLEBODY_META['skeleton_links']
        for person in people:
            points = np.asarray(person['keypoints_2d'])
            scores = np.asarray(person['scores_2d'])
            visible = scores > self.settings['keypoint_threshold']
            for a, b in links:
                if visible[a] and visible[b]:
                    cv2.line(canvas, tuple(points[a].astype(int)),
                             tuple(points[b].astype(int)), (0, 255, 0), 2)
            for point in points[visible]:
                cv2.circle(canvas, tuple(point.astype(int)), 3, (0, 0, 255), -1)
        return canvas
