import math
import os
from pathlib import Path
import sys
from collections import deque
from threading import Lock

import rospy
from sensor_msgs.msg import JointState
from fatigue_control_msgs.msg import MuscleState

from code.visuals.live_joint_velocity import LiveJointVelocityPlot
from code.visuals.live_muscle_fatigue import LiveMuscleFatiguePlot


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


class ModelViewer:

    def __init__(self):
        self._pending = deque(maxlen=100)
        self._pending_lock = Lock()
        model_path = rospy.get_param('~model_path', '')
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
        self.fatigue_plot = LiveMuscleFatiguePlot()
        self.subscription = rospy.Subscriber(
            rospy.get_param('~joint_states_topic', '/human_state_estimator/joint_states'),
            JointState, self.joint_state_callback, queue_size=1, tcp_nodelay=True,
        )
        self.muscle_subscription = rospy.Subscriber(
            rospy.get_param('~muscle_states_topic', '/human_state_estimator/muscle_states'),
            MuscleState, self.muscle_state_callback, queue_size=1, tcp_nodelay=True,
        )
        rospy.loginfo(
            f'Viewing {model_path}; listening on '
            f'{self.subscription.resolved_name} and '
            f'{self.muscle_subscription.resolved_name}')

    def refresh_plots(self):
        # rospy invokes subscribers on background threads. Apply model and GUI
        # updates here, on the main thread, including when images stop arriving.
        with self._pending_lock:
            pending = list(self._pending)
            self._pending.clear()
        for kind, msg in pending:
            if kind == 'joint':
                self._apply_joint_state(msg)
            else:
                self._apply_muscle_state(msg)
        self.velocity_plot.refresh()
        self.fatigue_plot.refresh()

    def joint_state_callback(self, msg):
        with self._pending_lock:
            self._pending.append(('joint', msg))

    def muscle_state_callback(self, msg):
        with self._pending_lock:
            self._pending.append(('muscle', msg))

    def _apply_muscle_state(self, msg):
        """Plot finite fatigue fractions with one value per unique muscle name."""
        if (not msg.name or len(msg.name) != len(msg.fatigue)
                or len(set(msg.name)) != len(msg.name)
                or not all(math.isfinite(value) and 0.0 <= value <= 1.0
                           for value in msg.fatigue)):
            rospy.logwarn_throttle(
                5.0, 'Ignoring MuscleState: need unique names and fatigue in [0, 1]')
            return
        timestamp_ns = msg.header.stamp.to_nsec()
        if timestamp_ns == 0:
            timestamp_ns = rospy.Time.now().to_nsec()
        self.fatigue_plot.append(
            timestamp_ns * 1e-9, dict(zip(msg.name, msg.fatigue)))

    def _apply_joint_state(self, msg):
        """Apply named positions in OpenSim units (radians or metres)."""
        if (not msg.name or len(msg.name) != len(msg.position)
                or len(set(msg.name)) != len(msg.name)
                or not all(math.isfinite(value) for value in msg.position)):
            rospy.logwarn_throttle(
                5.0, 'Ignoring JointState: need unique names and finite positions')
            return
        updates = [(self.coordinates[name], value)
                   for name, value in zip(msg.name, msg.position)
                   if name in self.coordinates]
        if not updates:
            rospy.logwarn_throttle(
                5.0, 'JointState has no matching unlocked OpenSim coordinates')
            return
        for coordinate, value in updates:
            coordinate.setValue(self.state, float(value), False)
        self.model.assemble(self.state)
        self.model.realizePosition(self.state)
        self.model.getVisualizer().show(self.state)
        if msg.velocity:
            if (len(msg.velocity) != len(msg.name)
                    or not all(math.isfinite(value) for value in msg.velocity)):
                rospy.logwarn_throttle(
                    5.0, 'Ignoring velocities: need one finite velocity per joint name')
                return
            timestamp_ns = msg.header.stamp.to_nsec()
            if timestamp_ns == 0:
                timestamp_ns = rospy.Time.now().to_nsec()
            self.velocity_plot.append(timestamp_ns * 1e-9, {
                name: value for name, value in zip(msg.name, msg.velocity)
                if name in self.coordinates
            })

    def close(self):
        self.subscription.unregister()
        self.muscle_subscription.unregister()
        try:
            self.visualizer.shutdown()
        finally:
            try:
                self.velocity_plot.close()
            finally:
                self.fatigue_plot.close()


def main():
    """Service the visualizer on the main thread until ROS shuts down."""
    rospy.init_node('model_viewer')
    node = None
    try:
        node = ModelViewer()
        while not rospy.is_shutdown():
            node.refresh_plots()
            rospy.rostime.wallsleep(1.0 / 30.0)
    except (KeyboardInterrupt, rospy.ROSInterruptException):
        pass
    finally:
        if node is not None:
            node.close()


if __name__ == '__main__':
    main()
