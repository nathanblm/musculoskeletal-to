from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import opensim as osim
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import JointState
from fatigue_control_msgs.msg import MuscleState

from code.mmpose_pipeline import MMPosePipeline, default_model_settings
from code.muscle_state_estimator import MuscleStateEstimator

CAMERA_TO_OPENSIM = np.array([
    [0.0, 0.0, -1.0],
    [0.0, -1.0, 0.0],
    [-1.0, 0.0, 0.0],
])


class LowPassFilter:

    def __init__(self, cutoff_hz):
        cutoff_hz = np.asarray(cutoff_hz, dtype=float)
        if not np.all(np.isfinite(cutoff_hz)) or np.any(cutoff_hz < 0):
            raise ValueError('cutoff_hz must be finite and non-negative')
        self.cutoff_hz = cutoff_hz.copy()
        self.reset()

    def reset(self):
        self.value = None
        self.timestamp_ns = None

    def update(self, sample, timestamp_ns):
        sample = np.asarray(sample, dtype=float)
        if self.timestamp_ns is None or timestamp_ns <= self.timestamp_ns:
            self.value = sample.copy()
        else:
            dt = (timestamp_ns - self.timestamp_ns) * 1e-9
            alpha = np.where(self.cutoff_hz == 0, 1.0,
                             -np.expm1(-2 * np.pi * self.cutoff_hz * dt))
            self.value += alpha * (sample - self.value)
        self.timestamp_ns = timestamp_ns
        return self.value.copy()


class HumanStateEstimator:

    def __init__(self):
        self.landmark_filter = LowPassFilter([
            rospy.get_param('~landmark_xy_cutoff_hz', 1.01),
            rospy.get_param('~landmark_xy_cutoff_hz', 1.01),
            rospy.get_param('~landmark_z_cutoff_hz', 0.5),
        ])
        model_path = rospy.get_param('~model_path', '')
        dataset_path = rospy.get_param('~dataset_path', '')
        if not model_path or not dataset_path:
            raise ValueError('~model_path and ~dataset_path must be provided')

        settings = default_model_settings()
        rospy.loginfo(
            f"Loading RTMDet and RTMW3D whole-body pose on {settings['device']}")
        
        self.pipeline = MMPosePipeline(settings)
        self.bridge = CvBridge()
        self.frame_count = 0
        
        self.model = osim.Model(str(model_path))
        self.muscle_estimator = MuscleStateEstimator(
            self.model,
            dataset_path,
            muscles=rospy.get_param('~muscles', []) or None,
            force_index=rospy.get_param('~force_index', 0),
        )
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

        self.position_filter = LowPassFilter([
            rospy.get_param('~position_cutoffs_hz/' + name, 0.0)
            for name in self.coordinate_names
        ])
        self.velocity_filter = LowPassFilter([
            rospy.get_param('~velocity_cutoffs_hz/' + name, 0.0)
            for name in self.coordinate_names
        ])

        # free coordinates within configured limits for IKsolver
        for coordinate in self.coordinates:
            coordinate.setClamped(self.state, True)

        self.previous = np.array([
            coordinate.getValue(self.state) for coordinate in self.coordinates
        ])
        self.previous_coordinates = None
        self.previous_timestamp_ns = None

        self.alignment = CAMERA_TO_OPENSIM

        self.joint_state_pub = rospy.Publisher(
            '~joint_states', JointState, queue_size=1)
        self.muscle_state_pub = rospy.Publisher(
            '~muscle_states', MuscleState, queue_size=1)
        self.image_pub = rospy.Publisher('~image', Image, queue_size=1)
        self.subscription = rospy.Subscriber(
            rospy.get_param('~image_topic', '/camera/webcam/image_raw'),
            Image, self.image_callback, queue_size=1,
            buff_size=16 * 1024 * 1024, tcp_nodelay=True,
        )
        rospy.loginfo(
            'Ready: publishing ~joint_states, ~muscle_states, and ~image')

    def image_callback(self, msg):

        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as error:
            rospy.logerr(f'Cannot decode image: {error}')
            return

        timestamp_ns = msg.header.stamp.to_nsec()
        people = self.pipeline.process(image, timestamp_ns)
        self.frame_count += 1

        if len(people) >= 1:
            coordinates = self.coordinates_from_people(people, timestamp_ns)

            if coordinates is not None:
                # apply low-pass filter to the joint angles
                coordinates = self.position_filter.update(coordinates, timestamp_ns)
                velocities = self.calculate_velocity(coordinates, timestamp_ns)
                
                if velocities is not None:
                    joint_state = JointState()
                    joint_state.header = msg.header
                    joint_state.name = list(self.coordinate_names)
                    joint_state.position = coordinates.tolist()
                    joint_state.velocity = velocities.tolist()
                    self.joint_state_pub.publish(joint_state)
                    try:
                        _, forces, fatigue = self.muscle_estimator.update(
                            coordinates,
                            timestamp_ns,
                        )
                    except (RuntimeError, ValueError) as error:
                        rospy.logerr(
                            f'Cannot estimate muscle state: {error}')
                    else:
                        muscle_state = MuscleState()
                        muscle_state.header = msg.header
                        muscle_state.name = list(
                            self.muscle_estimator.muscles.names)
                        muscle_state.force = forces.tolist()
                        muscle_state.fatigue = fatigue.tolist()
                        self.muscle_state_pub.publish(muscle_state)

        overlay = self.pipeline.draw_overlay(image, people)
        output = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        output.header = msg.header
        self.image_pub.publish(output)

    def coordinates_from_people(self, people, timestamp_ns):

        # we only estimate for the first person detected in the image
        points = np.asarray(people[0]['keypoints_3d'], dtype=float)
        self.model.realizePosition(self.state)
        shoulder = self.markers[0].getLocationInGround(self.state)

        # convert shoulder-relative MMPose vectors to OpenSim coordinates and
        # reconstruct ground-frame targets from the model's current shoulder.
        origin = np.array([shoulder[i] for i in range(3)])
        points = points[[8, 10, 117, 129]]

        # apply a low-pass filter to the landmark positions
        points = self.landmark_filter.update(points, timestamp_ns)
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
            # return previous coordinates if the IK solver fails
            for coordinate, value in zip(self.coordinates, self.previous):
                coordinate.setValue(self.state, float(value), False)
            return None

        self.previous = np.array([
            coordinate.getValue(self.state) for coordinate in self.coordinates
        ])

        return self.previous.copy()

    def calculate_velocity(self, coordinates, timestamp_ns):
        if self.previous_coordinates is None:
            self.previous_coordinates = coordinates.copy()
            self.previous_timestamp_ns = timestamp_ns
            self.velocity_filter.reset()
            return None

        dt = (timestamp_ns - self.previous_timestamp_ns) * 1e-9
        if dt <= 0.0:
            rospy.logwarn_throttle(5.0, 'Ignoring pose with non-increasing timestamp')
            self.previous_coordinates = coordinates.copy()
            self.previous_timestamp_ns = timestamp_ns
            self.velocity_filter.reset()
            return None

        velocities = (coordinates - self.previous_coordinates) / dt
        self.previous_coordinates = coordinates.copy()
        self.previous_timestamp_ns = timestamp_ns
        
        # apply low-pass filter to the joint velocities
        return self.velocity_filter.update(velocities, timestamp_ns)


def main():
    rospy.init_node('human_state_estimator')
    try:
        node = HumanStateEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
