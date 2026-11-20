#!/usr/bin/env python3
"""Map observed human joint angles to an iiwa joint position target."""

import numpy as np
import opensim as osim
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
# The human URDF converter maps OpenSim (x, y, z) to Gazebo (y, z, x).
OSIM_TO_GAZEBO = np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])
ROBOT_Z_ALONG_FORCE = Rotation.from_euler('y', np.pi / 2).as_matrix()


def transform(translation, quaternion):
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    matrix[:3, 3] = translation
    return matrix


class HumanIiwaPoseCoupling:
    def __init__(self):
        self.model = osim.Model(rospy.get_param('~model_path'))
        self.state = self.model.initSystem()
        self.coordinates = self.model.getCoordinateSet()
        self.hand_body = self.model.getBodySet().get(
            rospy.get_param('~hand_body', 'hand'))
        self.endpoint = self.model.getMarkerSet().get('endpoint')
        self.robot_base = rospy.get_param('~robot_base_frame', 'iiwa_link_0')
        self.command_robot = rospy.get_param('~command_robot', False)

        # Both transforms are configured as [x, y, z, qx, qy, qz, qw].
        base = rospy.get_param('~robot_base_in_human_root')
        grasp = rospy.get_param('~robot_ee_in_endpoint')
        self.root_to_base = transform(base[:3], base[3:])
        self.endpoint_to_ee = transform(grasp[:3], grasp[3:])
        self.endpoint_to_ee[:3, :3] = self.endpoint_to_ee[:3, :3] @ ROBOT_Z_ALONG_FORCE

        self.pose_pub = rospy.Publisher('~robot_pose_target', PoseStamped, queue_size=1)
        self.reference_pose_pub = rospy.Publisher(
            '/iiwa/CartesianImpedanceController/reference_pose', PoseStamped, queue_size=1)
        rospy.Subscriber('/human/joint_states', JointState, self.on_human_state, queue_size=1)

    def on_human_state(self, msg):
        positions = dict(zip(msg.name, msg.position))
        for i in range(self.coordinates.getSize()):
            coordinate = self.coordinates.get(i)
            coordinate.setValue(self.state, positions[coordinate.getName()], False)
        self.model.realizePosition(self.state)

        hand = self.hand_body.getTransformInGround(self.state)
        endpoint = self.endpoint.getLocationInGround(self.state)
        root_to_endpoint = np.eye(4)
        root_to_endpoint[:3, :3] = [
            [hand.R().get(i, j) for j in range(3)] for i in range(3)]
        root_to_endpoint[:3, 3] = [endpoint.get(i) for i in range(3)]
        root_to_endpoint = OSIM_TO_GAZEBO @ root_to_endpoint @ OSIM_TO_GAZEBO.T
        base_to_ee = np.linalg.inv(self.root_to_base) @ root_to_endpoint @ self.endpoint_to_ee
        x, y, z = base_to_ee[:3, 3]
        qx, qy, qz, qw = Rotation.from_matrix(base_to_ee[:3, :3]).as_quat()

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = self.robot_base
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = x, y, z
        pose.pose.orientation.x, pose.pose.orientation.y = qx, qy
        pose.pose.orientation.z, pose.pose.orientation.w = qz, qw
        self.pose_pub.publish(pose)

        if self.command_robot:
            self.reference_pose_pub.publish(pose)


if __name__ == '__main__':
    rospy.init_node('human_iiwa_pose_coupling')
    HumanIiwaPoseCoupling()
    rospy.spin()
