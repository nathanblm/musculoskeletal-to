# Webcam streaming

From the workspace root, build and launch with ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select fatigue_control_ros --symlink-install
source install/setup.bash
ros2 launch fatigue_control_ros webcam.launch.py
```

The installed `usb_cam` driver publishes images on `/camera/image_raw`
(`sensor_msgs/msg/Image`) and calibration information on `/camera/camera_info`.
The default camera is `/dev/video0`, streaming 640 x 480 at 30 FPS using MJPEG.

Override settings to match your webcam's supported capture modes:

```bash
ros2 launch fatigue_control_ros webcam.launch.py video_device:=/dev/video2 image_width:=1280 image_height:=720 framerate:=30.0
```

For a webcam using YUYV instead of MJPEG, pass `pixel_format:=yuyv`.
Check supported capture modes with `v4l2-ctl --device=/dev/video0 --list-formats-ext`
if `v4l2-ctl` is installed. The selected device must exist and be accessible
to your user.

In another terminal with the same ROS environment sourced, check the stream:

```bash
ros2 topic hz /camera/image_raw
```

## RTMPose 2D and MotionBERT 3D

The ROS adapter in `fatigue_control_ros/pose_estimator.py` handles parameters,
image messages and publishing. The core in `../../code/mmpose_pipeline.py`
loads models, tracks people, lifts poses and renders overlays without ROS imports.
`MMPosePipeline.process(image, timestamp_ns)` accepts a BGR NumPy image and
returns a `PoseFrame` with pose dictionaries and an annotated BGR image.
Use one pipeline per camera stream and call it sequentially.

Packaging installs the core directly from `code/mmpose_pipeline.py`; there is
no second source file or source link under `src/`. The installed import is
`mmpose_pipeline`, because `code` is also a Python standard-library module.
Build from the workspace root
using the Python environment containing MMPose, MMDetection, MMCV and PyTorch:

```bash
source /opt/ros/humble/setup.bash
.venvs/mmpose/bin/python "$(command -v colcon)" build --packages-select fatigue_control_ros --symlink-install
source install/setup.bash
ros2 run fatigue_control_ros pose_estimator
```

Keep the webcam launch running in another terminal. After building, new
terminals only need to source ROS and `install/setup.bash` before using
`ros2 run`. Building with the explicit Python interpreter ensures the ROS
executable uses the environment containing the model dependencies.

This explicitly uses RTMDet-M person detection, RTMPose-M (`human`/`body`
inferencer default), IoU tracking, COCO-to-Human3.6M joint conversion, and
MotionBERT lifting. It does not call `MMPoseInferencer`. Model configs come
from the installed MMPose `.mim` directory. Checkpoints download on first use
unless already in the Torch cache. Each config/checkpoint pair can be overridden
with ROS parameters (`detector_config`, `detector_checkpoint`, `pose2d_config`,
`pose2d_checkpoint`, `pose3d_config`, `pose3d_checkpoint`).

CUDA is selected when available; otherwise CPU is used. For example:

```bash
ros2 run fatigue_control_ros pose_estimator --ros-args -p device:=cpu -p image_topic:=/camera/image_raw
```

Outputs:

- `/pose_estimator/image`: annotated BGR image, viewable in `rqt_image_view`.
- `/pose_estimator/poses`: `std_msgs/msg/String` containing JSON. The message
  includes the source timestamp and image frame, joint ordering, and a `people`
  list. Each person has `track_id`, `bbox_xyxy`, COCO-17 `keypoints_2d` in
  pixels, `scores_2d`, and H36M-17 `keypoints_3d` in metres. No detections
  produces an empty list. Track IDs are matched between consecutive frames;
  they are not persistent identities after occlusion.

3D coordinates are raw MotionBERT output relative to the pelvis, with no
display-axis transformation or floor rebasing. They are inferred poses, not
calibrated world positions or robot joint angles. `image_frame_id` identifies
the input image and does not assert a ROS TF frame for the 3D coordinates.

The bundled fine-tuned MotionBERT configuration uses `seq_len=1` at inference,
despite `243frm` in its filename. The node reads sequence settings from the
config and retains a bounded history. For temporal configurations, unavailable
future frames are padded for online operation. A gap over one second, a
backward/repeated timestamp, or a resolution change resets tracking history.

