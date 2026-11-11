from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import opensim as osim
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState

from mmpose_pipeline import MMPosePipeline, default_model_settings

CAMERA_TO_OPENSIM = np.array([
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
])

class PoseEstimator(Node):

    def __init__(self):
        super().__init__('pose_estimator')

        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('model_path', '')

        settings = default_model_settings()
        self.get_logger().info(
            f"Loading RTMDet and RTMW3D whole-body pose on {settings['device']}")
        
        self.pipeline = MMPosePipeline(settings)
        self.bridge = CvBridge()
        self.frame_count = 0
        model_path = self.get_parameter('model_path').value
        if not model_path:
            raise ValueError('model_path must be provided for joint-state output')
        
        self.model = osim.Model(str(model_path))
        self.state = self.model.initSystem()
        self.markers = tuple(
            self.model.getMarkerSet().get(name)
            for name in ('shoulder', 'elbow', 'wrist', 'index_mcp', 'pinky_mcp')
        )
        self.marker_names = [marker.getName() for marker in self.markers]
        coordinates = self.model.getCoordinateSet()
        self.coordinates = tuple(
            coordinates.get(i) for i in range(coordinates.getSize())
            if not coordinates.get(i).get_locked()
        )
        self.coordinate_names = tuple(c.getName() for c in self.coordinates)

        # free coordinates within configured limits for IKsolver
        for coordinate in self.coordinates:
            coordinate.setClamped(self.state, True)
        self.previous = np.array([
            coordinate.getValue(self.state) for coordinate in self.coordinates
        ])
        self.alignment = CAMERA_TO_OPENSIM

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
            'Ready: publishing ~/joint_states and ~/image')

    def image_callback(self, msg):

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as error:
            self.get_logger().error(f'Cannot decode image: {error}')
            return

        timestamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        people = self.pipeline.process(image, timestamp_ns)
        self.frame_count += 1

        if len(people) >= 1:
            coordinates = self.estimate_from_people(people)

            if coordinates is not None:
                joint_state = JointState()
                joint_state.header = msg.header
                joint_state.name = list(self.coordinate_names)
                joint_state.position = coordinates.tolist()
                self.joint_state_pub.publish(joint_state)


        overlay = self.pipeline.draw_overlay(image, people)
        output = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        output.header = msg.header
        self.image_pub.publish(output)

    def estimate_from_people(self, people):

        # we only estimate for the first person detected in the image
        points = np.asarray(people[0]['keypoints_3d'], dtype=float)
        self.model.realizePosition(self.state)
        shoulder = self.markers[0].getLocationInGround(self.state)

        # convert shoulder-relative MMPose vectors to OpenSim coordinates and
        # reconstruct ground-frame targets from the model's current shoulder.
        origin = np.array([shoulder[i] for i in range(3)])
        points = points[[8, 10, 117, 129]]
        targets = origin + points @ self.alignment.T

        # create a table of the targets required by the IKsolver
        row = osim.RowVectorVec3(5)
        row[0] = osim.Vec3(*origin)
        for i, point in enumerate(targets, 1):
            row[i] = osim.Vec3(*point)
        table = osim.TimeSeriesTableVec3()
        table.setColumnLabels(self.marker_names)
        table.appendRow(0.0, row)

        # currently all markers influence the IK solution equally
        reference = osim.MarkersReference(table, osim.SetMarkerWeights())
        solver = osim.InverseKinematicsSolver(
            self.model, reference, osim.SimTKArrayCoordinateReference())

        # we're solving for static poses, not trajectories.
        solver.setAdvanceTimeFromReference(False)
        solver.setAccuracy(1e-5)
        try:
            solver.assemble(self.state)
        except RuntimeError:
            for coordinate, value in zip(self.coordinates, self.previous):
                coordinate.setValue(self.state, float(value), False)
            return None

        self.previous = np.array([
            coordinate.getValue(self.state) for coordinate in self.coordinates
        ])
        return self.previous.copy()

def main(args=None):
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
