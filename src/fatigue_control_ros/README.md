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
scaled `tsm_hand_3DOF` OpenSim model as the default:

```bash
ros2 launch fatigue_control_ros pose_estimator.launch.xml
```

The model path can be overridden for another model:

```bash
ros2 launch fatigue_control_ros pose_estimator.launch.xml \
  model_path:=musculoskeletal_models/tsm_hand_3DOF/tsm_3DOF_arm26_hand_scaled.osim
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
  -p model_path:=/absolute/path/to/tsm_3DOF_arm26_hand_scaled.osim
```

Override its input using `-p joint_states_topic:=/another/joint_states`.

## RTMW3D whole-body pose

The ROS adapter in `fatigue_control_ros/pose_estimator.py` handles parameters,
image messages and publishing. The core in `../../code/mmpose_pipeline.py`
loads models, tracks people and renders whole-body overlays without ROS imports.
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

This uses RTMDet-M person detection, IoU tracking and
[RTMW3D-L](https://github.com/open-mmlab/mmpose/tree/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/projects/rtmpose3d)
for direct 2D/3D estimation of 133 landmarks: 17 body, 6 feet, 68 face and
21 per hand (wrist plus four joints per finger). MotionBERT is no longer used.
The RTMW3D project components and config are included in `../../code/rtmpose3d`
and installed with this package; no separate MMPose checkout is needed.
Checkpoints download on first use unless already in the Torch cache.
Config/checkpoint pairs can be overridden with ROS parameters
(`detector_config`, `detector_checkpoint`, `pose3d_config`, `pose3d_checkpoint`).
The pose model must provide RTMW3D-compatible 133-point camera-space output
and `transformed_keypoints` for the image overlay. The previous `pose2d_*`
and `normalize_pose_2d` parameters have been removed.

CUDA is selected when available; otherwise CPU is used. For example:

```bash
ros2 run fatigue_control_ros pose_estimator --ros-args -p device:=cpu -p image_topic:=/camera/image_raw
```

Outputs:

- `/pose_estimator/image`: annotated BGR image, viewable in `rqt_image_view`.
- `/pose_estimator/poses`: `std_msgs/msg/String` containing JSON. The message
  includes the source timestamp and image frame, joint ordering, and a `people`
  list. `schema` is `COCO-WholeBody-133`; `joints_2d` and `joints_3d` are
  explicit ordered landmark-name lists. Each person has `track_id`, `bbox_xyxy`,
  `keypoint_schema`, 133 `keypoints_2d` in pixels, 133 `keypoints_3d` in metres,
  and `scores_2d`/`scores_3d`. Both score arrays use the model's shared landmark
  confidence, not independent estimates of depth accuracy. No detections
  produces an empty list. Track IDs match consecutive frames and reset after
  occlusion; they are not persistent identities.

The 3D origin is the midpoint of the hips, with camera axes X right, Y down,
Z away. Coordinates have no display-axis transformation or floor rebasing.
RTMW3D uses an assumed focal length (about 1145 pixels) and mean root depth
(about 5.14 m) without camera calibration. Metric scale and depth are therefore
approximate; these are not calibrated world positions. `image_frame_id`
identifies the source image, not a ROS TF frame for the estimated 3D pose.
The existing `pose_depth_sign` remains configurable; verify its value against
the new model's depth estimates in your camera setup.

Hand landmarks occupy indices 91–111 (left) and 112–132 (right). For the right
hand, 112 is the hand wrist; finger bases are 113 (thumb), 117 (index),
121 (middle), 125 (ring) and 129 (pinky). Use the supplied landmark names to
resolve indices. The vectors from the hand wrist to the index and pinky bases
provide a palm plane for a future hand-orientation IK constraint. Reject low
confidence or collinear landmarks before constructing that plane.

## OpenSim joint-state output

The pose estimator uses `../../code/arm_joint_state.py` to fit `elbow`,
`wrist`, `index_mcp`, and `pinky_mcp` relative to the `shoulder` marker.
RTMW3D indices are 6 (shoulder), 8 (elbow), 10 (wrist), 117 (index MCP),
and 129 (pinky MCP). A bounded least-squares fit uses the measured upper-arm,
forearm, and wrist-to-MCP directions plus the upper-arm-to-forearm bend angle.
It also penalizes every joint angle's change from its previous value with the
same fixed weight at all elbow flexions, reducing jitter from noisy landmarks.
The default model
is `musculoskeletal_models/tsm_hand_3DOF/tsm_3DOF_arm26_hand_scaled.osim`, which
contains these five markers. The viewer follows the fitted joint states directly.

The default 3DOF model is a self-contained copy of the scaled 4DOF model,
with shoulder `axial_rot` locked at 0 rad so elbow flexion moves the forearm
forward in the model's default shoulder pose. Its free coordinates are `plane_elv`,
`shoulder_elv`, and `elbow_flexion`. The folder includes all referenced mesh
files and a copy of the scaling measurements. To run the 4DOF version again,
pass `model_path:=musculoskeletal_models/tsm_hand_4DOF/tsm_4DOF_arm26_hand_scaled.osim`.

The scaled model uses these right-arm tape measurements: shoulder tip to
outer elbow 35 cm, outer elbow to wrist 27 cm, wrist to index MCP 9 cm,
wrist to pinky MCP 8.5 cm, and index-to-pinky MCP width 6.5 cm. The surface
landmarks correspond to `acc` (clavicle) and `EpL` (humerus) in the original
TSM model; these are not the internal `shoulder` and `elbow` tracking markers.
The surface distances are matched in the model's default reference pose.

The humerus is scaled along its length by 1.31396 and both forearm bones by
1.03787 using OpenSim's model scaling, which also updates joint frames and
muscle length properties. The hand's in-plane geometry and MCP markers are
transformed together to fit all three hand measurements; palm-normal
thickness is retained. Scaled hand meshes have the `scaled_` filename prefix.
Original model and geometry files are retained. Segment masses are retained
because no subject mass data were supplied; inertias reflect the new geometry.
This is a geometric fit for IK, not a calibration of muscle strength or
RTMW3D's monocular depth scale.

Measurements and scaling transforms are recorded in
`musculoskeletal_models/tsm_hand_4DOF/scaling_measurements.json`.
Regenerate the scaled copy from the original model with:

```bash
.venvs/ros/bin/python musculoskeletal_models/tsm_hand_4DOF/scale_to_measurements.py
```

Before fitting, camera coordinates are rotated into the OpenSim ground frame.
The rotation assumes an upright camera facing the subject. The separate
`pose_depth_sign` defaults to `1.0`. With the default camera axes, the mapping
is `(X_model, Y_model, Z_model) = (-Z_pose, -Y_pose, -X_pose)`, so a wrist moving
toward the camera bends forward in the OpenSim model. The former `-1.0` default
reversed that motion. Set `pose_depth_sign` to `-1.0` only if your camera's
reported depth axis runs toward the camera.
Raw `/pose_estimator/poses` JSON keeps pelvis-relative RTMW3D camera coordinates.

For a different camera placement, set the `camera_to_model_rotation` ROS
parameter to a row-major list of nine floats describing the calibrated
camera-to-model rotation. This is a fixed camera alignment; it does not
track torso rotation. The model's coordinate limits still restrict the poses
the fit can reproduce.

Provide the model when starting the node:

```bash
ros2 run fatigue_control_ros pose_estimator --ros-args \
-p model_path:=/absolute/path/to/tsm_3DOF_arm26_hand_scaled.osim
```

The result is published as `sensor_msgs/msg/JointState` on
`/pose_estimator/joint_states`. The `name` field is populated from the free
OpenSim coordinate names and the `position` field contains radians. The first
person detected in each current frame is used, without looking up or retaining
a `track_id`. Keep only one person visible during experiments. Frames without
a person or with invalid 3D landmarks produce no joint state; publishing resumes
as soon as a valid person is detected again.

RTMW3D processes each frame independently. A gap over one second, a
backward/repeated timestamp, or a resolution change resets tracking.
