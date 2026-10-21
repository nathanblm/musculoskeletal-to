import casadi as ca

class CircularPath:
    def __init__(self, r1, tz, theta1, h, r2):
        self.r1 = r1
        self.tz = tz
        self.theta1 = theta1
        self.h = h
        self.r2 = r2


    def evaluate(self, theta2):

        p1_ground = ca.vertcat(
            self.r1 * ca.cos(self.theta1),
            self.h,
            self.tz + self.r1 * ca.sin(self.theta1),
        )

        p2_plane = ca.vertcat(
            0.0,
            -self.r2 * ca.sin(theta2),
            self.r2 * ca.cos(theta2),
        )

        def y_rotation(theta):
            return ca.vertcat(
                ca.horzcat(ca.cos(theta), 0.0, -ca.sin(theta)),
                ca.horzcat(0.0, 1.0, 0.0),
                ca.horzcat(ca.sin(theta), 0.0, ca.cos(theta)),
            )

        p2_in_ground = y_rotation(self.theta1) @ p2_plane + p1_ground
        return p2_in_ground
        
