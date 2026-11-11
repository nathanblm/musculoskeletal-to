# Webcam streaming

The combined Python 3.10 environment is `.venvs/ros` (activate with
`source .venvs/ros/bin/activate`). It contains MMPose, OpenSim 4.4 and CasADi.
The Python 3.10 OpenSim bindings and native libraries are bundled under
`.venvs/ros/opensim-runtime` and loaded through `opensim.pth`.

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

To start the webcam, pose estimator and Simbody model viewer together, with the
`tsm_3DOF` OpenSim model as the default:

```bash
ros2 launch fatigue_control_ros pose_estimator.launch.xml
```

The model path can be overridden for another model:

```bash
ros2 launch fatigue_control_ros pose_estimator.launch.xml \
  model_path:=musculoskeletal_models/tsm_4DOF/tsm_4DOF.osim
```

The `model_viewer` node uses the same `model_path` and listens to
`/pose_estimator/joint_states` with sensor-data QoS. It displays the default
pose until joint positions arrive, then updates unlocked OpenSim coordinates
by name. Missing coordinates retain their values; unknown names are ignored.
OpenSim's native Simbody visualizer and a graphical desktop are required.
The bundled runtime must include `simbody/libexec/simbody/simbody-visualizer`
(or `simbody/bin/simbody-visualizer`) and its shared libraries, including
`simbody/lib/liblodepng.so`. The node detects either executable layout;
an explicit `SIMBODY_HOME` overrides discovery.
Pass `use_model_viewer:=false` to launch without the viewer.

To run only the viewer:

```bash
ros2 run fatigue_control_ros model_viewer --ros-args \
  -p model_path:=/absolute/path/to/tsm_3DOF.osim
```

Override its input using `-p joint_states_topic:=/another/joint_states`.

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
.venvs/ros/bin/python "$(command -v colcon)" build --packages-select fatigue_control_ros --symlink-install
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

## OpenSim joint-state output

The pose estimator uses `../../code/arm_marker_ik.py` to fit the existing
`HumHead_Center`, `centelbow`, and `endpoint` virtual markers with OpenSim's
`InverseKinematicsSolver`. Shoulder-relative H36M elbow and wrist vectors are
rotated into model axes and translated to the model's fixed shoulder position.
No physical markers are required. This prototype creates a marker reference
and calls `assemble()` for each frame, starting from the previous fitted pose.
It fits elbow and wrist positions together, without segment-length rescaling.
The original fitter remains in `../../code/arm_joint_state.py`.

Before fitting, camera coordinates are rotated into the OpenSim ground frame.
The rotation assumes an upright camera facing the subject. The separate
`pose_depth_sign` parameter defaults to `-1.0` to correct the reversed
front/back motion observed with this setup. With both defaults, the combined
mapping is `(X_model, Y_model, Z_model) = (Z_pose, -Y_pose, -X_pose)`.
This preserves the Y-down to Y-up correction while reversing only model X
relative to the previous mapping. Set `pose_depth_sign` to `1.0` to use
the camera-depth-away convention without a depth reversal.
Raw `/pose_estimator/poses` JSON keeps the original MotionBERT coordinates.

For a different camera placement, set the `camera_to_model_rotation` ROS
parameter to a row-major list of nine floats describing the calibrated
camera-to-model rotation. This is a fixed camera alignment; it does not
track torso rotation. The model's coordinate limits still restrict the poses
the fit can reproduce.

Provide the model when starting the node:

```bash
ros2 run fatigue_control_ros pose_estimator --ros-args \
-p model_path:=/absolute/path/to/tsm_3DOF.osim
```

The result is published as `sensor_msgs/msg/JointState` on
`/pose_estimator/joint_states`. The `name` field is populated from the free
OpenSim coordinate names and the `position` field contains radians. The first
person detected in each current frame is used, without looking up or retaining
a `track_id`. Keep only one person visible during experiments. Frames without
a person or with invalid 3D landmarks produce no joint state; publishing resumes
as soon as a valid person is detected again.

The bundled fine-tuned MotionBERT configuration uses `seq_len=1` at inference,
despite `243frm` in its filename. The node reads sequence settings from the
config and retains a bounded history. For temporal configurations, unavailable
future frames are padded for online operation. A gap over one second, a
backward/repeated timestamp, or a resolution change resets tracking history.
