"""Apply estimated human joint positions to a Gazebo model for visualization."""

import rospy
from gazebo_msgs.srv import SetModelConfiguration
from sensor_msgs.msg import JointState


class GazeboJointStateBridge:
    def __init__(self):
        self.model_name = rospy.get_param('~model_name', 'tsm_hand_4DOF')
        self.urdf_param = rospy.get_param('~urdf_param', 'human_description')
        topic = rospy.get_param(
            '~joint_states_topic', '/human/joint_states')
        self.set_configuration = rospy.ServiceProxy(
            '/gazebo/set_model_configuration', SetModelConfiguration)
        self.subscriber = rospy.Subscriber(
            topic, JointState, self._on_joint_state, queue_size=1)

    def _on_joint_state(self, msg):
        if not msg.name or len(msg.name) != len(msg.position):
            rospy.logwarn_throttle(5, 'Skipping incomplete human joint state')
            return
        try:
            response = self.set_configuration(
                self.model_name, self.urdf_param,
                list(msg.name), list(msg.position))
        except rospy.ServiceException as error:
            rospy.logwarn_throttle(5, 'Gazebo joint update failed: %s', error)
            return
        if not response.success:
            rospy.logwarn_throttle(
                5, 'Gazebo rejected human joint update: %s', response.status_message)


def main():
    rospy.init_node('gazebo_joint_state_bridge')
    GazeboJointStateBridge()
    rospy.spin()
