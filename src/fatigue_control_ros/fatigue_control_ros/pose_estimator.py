import json

from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from arm_joint_state import CAMERA_TO_OPENSIM, OpenSimArmJointState, OpenSimMarkerIK
from mmpose_pipeline import (
    WHOLEBODY_JOINTS, MMPosePipeline, default_model_settings,
)

class PoseEstimator(Node):

    def __init__(self):
        super().__init__('pose_estimator')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', '')
        self.declare_parameter('ik_solver', 'custom')
        self.declare_parameter(
            'camera_to_model_rotation', CAMERA_TO_OPENSIM.ravel().tolist())
        self.declare_parameter('pose_depth_sign', 1.0)
        defaults = default_model_settings()
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        settings = {name: self.get_parameter(name).value for name in defaults}
        self.get_logger().info(
            f"Loading RTMDet and RTMW3D whole-body pose on {settings['device']}")
        
        self.pipeline = MMPosePipeline(settings)
        self.bridge = CvBridge()
        self.frame_count = 0
        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise ValueError('model_path must be provided for joint-state output')
        rotation = self.get_parameter('camera_to_model_rotation').value
        if len(rotation) != 9:
            raise ValueError('camera_to_model_rotation must contain 9 values')
        solver = {
            'custom': OpenSimArmJointState,
            'opensim': OpenSimMarkerIK,
        }[self.get_parameter('ik_solver').value]
        self.arm_state = solver(
            model_path, alignment=[rotation[i:i + 3] for i in range(0, 9, 3)],
            depth_sign=self.get_parameter('pose_depth_sign').value)

        self.pose_pub = self.create_publisher(
            String, 
            '~/poses', 
            10
        
        )
        self.joint_state_pub = self.create_publisher(
            JointState,
            '~/joint_states',
            qos_profile_sensor_data
        )

        self.image_pub = self.create_publisher(
            Image, 
            '~/image', 
            qos_profile_sensor_data
        )

        self.subscription = self.create_subscription(
            Image, 
            self.get_parameter('image_topic').value, 
            self.image_callback,
            QoSProfile(
                depth=1, 
                reliability=ReliabilityPolicy.BEST_EFFORT
            ),
        )

        self.get_logger().info(
            'Ready: publishing ~/poses, ~/joint_states and ~/image')

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Cannot decode image: {error}')
            return
        timestamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        frame = self.pipeline.process(image, timestamp_ns)
        self.frame_count += 1
        self.publish_poses(msg.header, frame.people)
        if hasattr(self, 'arm_state'):
            coordinates = self.arm_state.estimate_from_people(frame.people)
            if coordinates is not None:
                joint_state = JointState()
                joint_state.header = msg.header
                joint_state.name = list(self.arm_state.coordinate_names)
                joint_state.position = coordinates.tolist()
                self.joint_state_pub.publish(joint_state)
            elif self.frame_count == 1 or self.frame_count % 30 == 0:
                if not frame.people:
                    reason = 'no people detected'
                else:
                    reason = 'IK did not converge for the current landmarks'
                self.get_logger().warning(
                    f'No joint state published: {reason}')
        output = self.bridge.cv2_to_imgmsg(frame.overlay, encoding='bgr8')
        output.header = msg.header
        self.image_pub.publish(output)

    def publish_poses(self, header, people):
        payload = {
            'stamp': {'sec': header.stamp.sec, 'nanosec': header.stamp.nanosec},
            'image_frame_id': header.frame_id,
            'schema': 'COCO-WholeBody-133',
            'joints_2d': WHOLEBODY_JOINTS, 'joints_3d': WHOLEBODY_JOINTS,
            'units_3d': 'm', 'coordinates_3d': 'rtmw3d_camera_pelvis_relative',
            'people': people,
        }
        self.pose_pub.publish(String(data=json.dumps(payload, allow_nan=False)))


def main(args=None):
    """Initialize ROS and run the pose estimator until shutdown."""
    rclpy.init(args=args)
    node = None
    try:
        node = PoseEstimator()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
