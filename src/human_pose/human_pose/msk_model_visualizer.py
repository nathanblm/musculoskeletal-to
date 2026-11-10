"""Bridge ROS pose estimates to an isolated OpenSim/Simbody viewer process."""

import json
import math
import os
from pathlib import Path
import subprocess
import time

from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


ANGLE_SUFFIXES = (
    'plane_elevation_estimate',
    'shoulder_elevation_estimate',
    'elbow_flexion_estimate',
)


def pose_angles(message, side):
    """Read radians by name, independently of the incoming field order."""
    if len(message.name) != len(message.position):
        raise ValueError('JointState name and position lengths differ')
    if len(set(message.name)) != len(message.name):
        raise ValueError('JointState contains duplicate names')
    values = dict(zip(message.name, message.position))
    try:
        angles = [float(values[f'{side}_{suffix}']) for suffix in ANGLE_SUFFIXES]
    except KeyError as error:
        raise ValueError(f'Missing angle: {error.args[0]}') from error
    if not all(math.isfinite(value) for value in angles):
        raise ValueError('Pose angles must be finite')
    return angles


def default_opensim_python():
    # Support both source execution and the workspace's installed ROS package.
    for origin in (Path(__file__).resolve(), Path(__file__).absolute()):
        for parent in origin.parents:
            candidate = parent / '.venvs' / 'opensim' / 'bin' / 'python'
            if candidate.is_file():
                return str(candidate)
    return ''


class MSKModelVisualizer(Node):
    def __init__(self):
        super().__init__('msk_model_visualizer')
        share = Path(get_package_share_directory('human_pose'))
        self.declare_parameter('topic', '/human/pose_angles')
        self.declare_parameter('side', 'right')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('opensim_python', default_opensim_python())
        self.declare_parameter('model_path', str(share / 'models/tsm_3DOF/tsm_3DOF.osim'))
        self.side = self.get_parameter('side').value
        rate = float(self.get_parameter('rate_hz').value)
        python = Path(self.get_parameter('opensim_python').value).expanduser()
        model = Path(self.get_parameter('model_path').value).expanduser()
        if self.side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('rate_hz must be positive and finite')
        if not python.is_file():
            raise FileNotFoundError('Set opensim_python to the OpenSim environment Python')
        if not model.is_file():
            raise FileNotFoundError(f'OpenSim model not found: {model}')

        self.latest = None
        self.last_received = None
        self.stale = False
        self.worker = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.subscription = self.create_subscription(
            JointState, self.get_parameter('topic').value, self.on_angles, qos,
        )
        self.timer = self.create_timer(1.0 / rate, self.update_viewer)
        environment = os.environ.copy()
        # ROS and MMPose paths must not leak Python 3.10 extensions into 3.9.
        for name in ('PYTHONPATH', 'PYTHONHOME', 'QT_QPA_PLATFORM_PLUGIN_PATH',
                     'QT_QPA_FONTDIR', 'QT_PLUGIN_PATH'):
            environment.pop(name, None)
        environment['PYTHONNOUSERSITE'] = '1'
        worker_path = Path(__file__).with_name('opensim_viewer.py')
        self.worker = subprocess.Popen(
            [str(python), '-u', str(worker_path), '--model', str(model)],
            stdin=subprocess.PIPE, env=environment,
        )
        os.set_blocking(self.worker.stdin.fileno(), False)
        self.get_logger().info(
            f'Waiting for {self.side} arm angles on {self.get_parameter("topic").value}; '
            'displaying the TSM right-arm model. Ctrl+C stops the viewer.'
        )

    def on_angles(self, message):
        try:
            self.latest = pose_angles(message, self.side)
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=5.0)
            return
        self.last_received = time.monotonic()
        if self.stale:
            self.get_logger().info('Pose stream resumed')
            self.stale = False

    def update_viewer(self):
        if self.worker.poll() is not None:
            raise RuntimeError(f'OpenSim viewer exited with code {self.worker.returncode}')
        if (self.last_received is not None
                and time.monotonic() - self.last_received > 2.0 and not self.stale):
            self.get_logger().warning('No pose received for 2 seconds; holding last posture')
            self.stale = True
        if self.latest is None:
            return
        payload = (json.dumps(self.latest, allow_nan=False) + '\n').encode()
        try:
            # Small atomic nonblocking writes keep a slow viewer from blocking ROS.
            os.write(self.worker.stdin.fileno(), payload)
        except BlockingIOError:
            return
        self.latest = None

    def destroy_node(self):
        if self.worker is not None:
            self.worker.stdin.close()
            try:
                self.worker.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                try:
                    self.worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.worker.kill()
                    self.worker.wait()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = MSKModelVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
