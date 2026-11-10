# Human pose and TSM visualization

`pose_publisher` publishes geometric estimates in radians as `sensor_msgs/JointState`
on `/human/pose_angles`. `msk_model_visualizer` subscribes to those estimates and
displays the `tsm_3DOF` model using OpenSim's Simbody visualizer. It also accepts
the same topic played back from a ROS bag.

## Separate environments

ROS Humble and pose estimation run in `.venvs/mmpose` with Python 3.10. The
OpenSim worker runs in `.venvs/opensim` with Python 3.9, communicating through
a pipe. OpenSim is never imported into the ROS process.

The existing Conda `opensim` installation was checked: Python 3.9.25, OpenSim
4.4 (`py39np120`), NumPy 1.24.4, SciPy 1.10.1, and Simbody 3.8 with the viewer
at `<conda-prefix>/simbody/libexec/simbody/simbody-visualizer`. SciPy's installed
runtime version differs from the 1.13.1 entry reported by Conda metadata.
This viewer only needs OpenSim, NumPy, SciPy, and the native Simbody viewer;
it does not need PyTorch, MMPose, CasADi, or ROS in its Python environment.

Create the separate venv from the working Conda installation:

```bash
uv venv --python ~/miniconda3/envs/opensim/bin/python \
  --system-site-packages .venvs/opensim
touch .venvs/COLCON_IGNORE
```

This venv deliberately inherits the Conda installation's packages and native
libraries. Keep that Conda environment installed; this is not a standalone
copy of OpenSim. `SIMBODY_HOME` defaults to that installation's `simbody` folder.

## Build and run

From the workspace root, build with the ROS/pose Python:

```bash
source /opt/ros/humble/setup.bash
source .venvs/mmpose/bin/activate
python -m colcon build --base-paths src --packages-select human_pose --symlink-install
```

In a new terminal, activate the ROS/pose environment, source ROS and the
workspace, then start the visualizer:

```bash
source /opt/ros/humble/setup.bash
source .venvs/mmpose/bin/activate
source install/local_setup.bash
ros2 run human_pose msk_model_visualizer
```

Run `ros2 run human_pose pose_publisher` in another terminal with the same ROS
setup. The visualizer starts `.venvs/opensim/bin/python` itself; do not activate
the OpenSim venv in the ROS terminal. Stop the node with Ctrl+C.

Parameters: `topic` (default `/human/pose_angles`), `side` (`right` or `left`),
`rate_hz` (default `30.0`), `model_path`, and `opensim_python`. When installed
outside this workspace, provide the OpenSim interpreter explicitly:

```bash
ros2 run human_pose msk_model_visualizer --ros-args \
  -p opensim_python:=/absolute/path/to/.venvs/opensim/bin/python
```

## Angle alignment and limitations

Input fields are `<side>_plane_elevation_estimate`,
`<side>_shoulder_elevation_estimate`, and `<side>_elbow_flexion_estimate`.
Their order in the message does not matter. Plane zero means lateral, positive
pi/2 means anterior; elevation zero means arm down; elbow zero means straight.

The worker fits `plane_elv`, `shoulder_elv`, and `elbow_flexion` with bounded
least squares against the model's actual shoulder-to-elbow direction and
elbow-to-wrist bend in the thorax frame. Direct coordinate assignment would
ignore TSM's fixed scapula orientation. Coordinates stay within model ranges.
Unreachable postures generate a warning and display the closest fitted pose.

TSM is a right-arm model. Selecting `side:=left` mirrors the left arm's
body-relative geometric angles onto that right arm. The thorax stays fixed;
global body translation/rotation and humeral axial rotation are not tracked.
This displays pose estimates, not biomechanically calibrated joint measurements.
On missing input, the viewer holds the last posture and warns after two seconds.
