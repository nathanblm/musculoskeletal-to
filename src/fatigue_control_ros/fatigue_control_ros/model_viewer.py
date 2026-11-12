import math
import os
from pathlib import Path
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from visuals.live_joint_velocity import LiveJointVelocityPlot


def configure_simbody(osim):
    if os.environ.get('SIMBODY_HOME'):
        return
    prefixes = (Path(sys.prefix), *Path(osim.__file__).resolve().parents)
    for prefix in prefixes:
        for root in (prefix / 'simbody', prefix):
            for directory in ('bin', 'libexec/simbody'):
                executable = root / directory / 'simbody-visualizer'
                if executable.is_file() and os.access(executable, os.X_OK):
                    os.environ['SIMBODY_HOME'] = str(root)
                    return


class ModelViewer(Node):

    def __init__(self):
        super().__init__('model_viewer')
        self.declare_parameter('model_path', '')
        self.declare_parameter(
            'joint_states_topic', '/pose_estimator/joint_states')
        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise ValueError('model_path must be provided for visualization')
        model_path = Path(model_path).expanduser().resolve(strict=True)

        import opensim as osim

        configure_simbody(osim)

        self.model = osim.Model(str(model_path))
        self.model.setUseVisualizer(True)
        self.state = self.model.initSystem()
        coordinates = self.model.getCoordinateSet()
        self.coordinates = {
            coordinates.get(i).getName(): coordinates.get(i)
            for i in range(coordinates.getSize())
            if not coordinates.get(i).get_locked()
        }
        self.visualizer = self.model.getVisualizer().updSimbodyVisualizer()
        self.model.realizePosition(self.state)
        self.model.getVisualizer().show(self.state)
        self.velocity_plot = LiveJointVelocityPlot(self.coordinates)
        self.plot_timer = self.create_timer(1.0 / 30.0, self.velocity_plot.refresh)
        self.subscription = self.create_subscription(
            JointState,
            self.get_parameter('joint_states_topic').value,
            self.joint_state_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f'Viewing {model_path}; '
            f'listening on {self.subscription.topic_name}')

    def joint_state_callback(self, msg):
        """Apply named positions in OpenSim units (radians or metres)."""
        if (not msg.name or len(msg.name) != len(msg.position)
                or len(set(msg.name)) != len(msg.name)
                or not all(math.isfinite(value) for value in msg.position)):
            self.get_logger().warning(
                'Ignoring JointState: need unique names and finite positions',
                throttle_duration_sec=5.0,
            )
            return
        updates = [(self.coordinates[name], value)
                   for name, value in zip(msg.name, msg.position)
                   if name in self.coordinates]
        if not updates:
            self.get_logger().warning(
                'JointState has no matching unlocked OpenSim coordinates',
                throttle_duration_sec=5.0,
            )
            return
        for coordinate, value in updates:
            coordinate.setValue(self.state, float(value), False)
        self.model.assemble(self.state)
        self.model.realizePosition(self.state)
        self.model.getVisualizer().show(self.state)
        if msg.velocity:
            if (len(msg.velocity) != len(msg.name)
                    or not all(math.isfinite(value) for value in msg.velocity)):
                self.get_logger().warning(
                    'Ignoring velocities: need one finite velocity per joint name',
                    throttle_duration_sec=5.0,
                )
                return
            timestamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if timestamp_ns == 0:
                timestamp_ns = self.get_clock().now().nanoseconds
            self.velocity_plot.append(timestamp_ns * 1e-9, {
                name: value for name, value in zip(msg.name, msg.velocity)
                if name in self.coordinates
            })

    def destroy_node(self):
        try:
            self.visualizer.shutdown()
        finally:
            try:
                self.velocity_plot.close()
            finally:
                result = super().destroy_node()
        return result


def main(args=None):
    """Run the model viewer until ROS shuts down."""
    rclpy.init(args=args)
    node = None
    try:
        node = ModelViewer()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
