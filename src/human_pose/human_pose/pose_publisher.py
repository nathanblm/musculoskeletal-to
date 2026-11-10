import math

import matplotlib

# Render the 3D skeleton into an image; OpenCV owns the live display window.
# Loading Matplotlib's Qt backend conflicts with OpenCV's bundled Qt libraries.
matplotlib.use("Agg")

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    HistoryPolicy,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState
from mmpose.apis import MMPoseInferencer


def unit(vector):
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Degenerate landmark geometry")
    return vector / length


def angle_between(a, b):
    return float(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1, 1)))


class PosePublisher(Node):
    # MotionBERT's Human3.6M landmark ordering, not COCO ordering.
    RIGHT_HIP = 1
    LEFT_HIP = 4
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 12
    LEFT_WRIST = 13
    RIGHT_SHOULDER = 14
    RIGHT_ELBOW = 15
    RIGHT_WRIST = 16

    def __init__(self):
        super().__init__("pose_publisher")

        self.declare_parameter("camera_id", 0)
        self.declare_parameter("side", "right")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("smoothing_alpha", 0.3)

        self.camera_id = self.get_parameter("camera_id").value
        self.side = self.get_parameter("side").value
        self.alpha = float(self.get_parameter("smoothing_alpha").value)

        if self.side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        if not 0 < self.alpha <= 1:
            raise ValueError("smoothing_alpha must be in (0, 1]")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.publisher = self.create_publisher(
            JointState, "/human/pose_angles", qos
        )

        self.inferencer = MMPoseInferencer(
            pose3d="human3d",
            device=self.get_parameter("device").value,
        )

        self.filtered = None
        self.previous_plane = None
        self.frame_count = 0

        self.get_logger().info(
            f"Tracking {self.side} arm in 3D; "
            "publishing geometric angles on /human/pose_angles"
        )

    def calculate_angles(self, points):
        left_shoulder = points[self.LEFT_SHOULDER]
        right_shoulder = points[self.RIGHT_SHOULDER]
        shoulder_mid = (left_shoulder + right_shoulder) / 2
        hip_mid = (
            points[self.LEFT_HIP] + points[self.RIGHT_HIP]
        ) / 2

        # Body-relative orthonormal frame.
        up = unit(shoulder_mid - hip_mid)
        right = right_shoulder - left_shoulder
        right = unit(right - np.dot(right, up) * up)
        forward = unit(np.cross(up, right))

        if self.side == "right":
            shoulder = right_shoulder
            elbow = points[self.RIGHT_ELBOW]
            wrist = points[self.RIGHT_WRIST]
            outward = right
        else:
            shoulder = left_shoulder
            elbow = points[self.LEFT_ELBOW]
            wrist = points[self.LEFT_WRIST]
            outward = -right

        upper_arm = unit(elbow - shoulder)
        forearm = unit(wrist - elbow)

        # 0: arm down; pi/2: upper arm horizontal.
        elevation = angle_between(upper_arm, -up)

        lateral = float(np.dot(upper_arm, outward))
        anterior = float(np.dot(upper_arm, forward))

        # Elevation-plane direction is poorly defined with the arm down/up.
        if math.hypot(lateral, anterior) < 0.1:
            if self.previous_plane is None:
                return None
            plane = self.previous_plane
        else:
            # 0: sideways; +pi/2: forward.
            plane = math.atan2(anterior, lateral)
            if self.previous_plane is not None:
                delta = math.atan2(
                    math.sin(plane - self.previous_plane),
                    math.cos(plane - self.previous_plane),
                )
                plane = self.previous_plane + delta
            self.previous_plane = plane

        # 0: straight elbow; pi/2: right-angle bend.
        flexion = angle_between(upper_arm, forearm)

        return np.array([plane, elevation, flexion])

    def run(self):
        # Keep one video generator alive to preserve temporal tracking.
        results = self.inferencer(
            f"webcam:{self.camera_id}",
            show=True,
        )

        try:
            for result in results:
                if not rclpy.ok():
                    break

                rclpy.spin_once(self, timeout_sec=0.0)

                # This is the result-availability time, not capture time.
                stamp = self.get_clock().now().to_msg()
                people = result["predictions"][0]

                # Avoid silently switching between people.
                # Stand alone in view for this initial implementation.
                if len(people) != 1:
                    self.filtered = None
                    self.previous_plane = None
                    continue

                points = np.asarray(
                    people[0]["keypoints"], dtype=float
                )

                # Some versions include an extra singleton dimension.
                if points.shape == (1, 17, 3):
                    points = points[0]

                if points.shape != (17, 3):
                    raise RuntimeError(
                        f"Expected Human3.6M landmarks (17, 3), "
                        f"received {points.shape}"
                    )
                if not np.all(np.isfinite(points)):
                    continue

                try:
                    angles = self.calculate_angles(points)
                except ValueError:
                    continue

                if angles is None:
                    continue

                if self.filtered is None:
                    self.filtered = angles
                else:
                    self.filtered += self.alpha * (
                        angles - self.filtered
                    )

                message = JointState()
                message.header.stamp = stamp
                message.name = [
                    f"{self.side}_plane_elevation_estimate",
                    f"{self.side}_shoulder_elevation_estimate",
                    f"{self.side}_elbow_flexion_estimate",
                ]
                message.position = self.filtered.tolist()
                self.publisher.publish(message)

                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    degrees = np.degrees(self.filtered)
                    self.get_logger().info(
                        f"Plane: {degrees[0]:.1f}, "
                        f"shoulder: {degrees[1]:.1f}, "
                        f"elbow: {degrees[2]:.1f} deg"
                    )
        finally:
            results.close()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = PosePublisher()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
