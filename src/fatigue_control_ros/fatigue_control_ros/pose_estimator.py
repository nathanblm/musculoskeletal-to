import json

from cv_bridge import CvBridge, CvBridgeError
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from mmpose_pipeline import (
    H36M_JOINTS, MMPosePipeline, default_model_settings,
)

class PoseEstimator(Node):

    def __init__(self):
        super().__init__('pose_estimator')

        self.declare_parameter('image_topic', '/camera/image_raw')
        defaults = default_model_settings()
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        settings = {name: self.get_parameter(name).value for name in defaults}
        self.get_logger().info(
            f"Loading RTMDet, RTMPose and MotionBERT on {settings['device']}")
        
        self.pipeline = MMPosePipeline(settings)
        self.bridge = CvBridge()

        self.pose_pub = self.create_publisher(
            String, 
            '~/poses', 
            10
        
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

        self.get_logger().info('Ready: publishing ~/poses and ~/image')

    def image_callback(self, msg):
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Cannot decode image: {error}')
            return
        timestamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        frame = self.pipeline.process(image, timestamp_ns)
        self.publish_poses(msg.header, frame.people)
        output = self.bridge.cv2_to_imgmsg(frame.overlay, encoding='bgr8')
        output.header = msg.header
        self.image_pub.publish(output)

    def publish_poses(self, header, people):
        payload = {
            'stamp': {'sec': header.stamp.sec, 'nanosec': header.stamp.nanosec},
            'image_frame_id': header.frame_id,
            'joints_2d': 'COCO-17', 'joints_3d': H36M_JOINTS,
            'units_3d': 'm', 'coordinates_3d': 'motionbert_pelvis_relative',
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
