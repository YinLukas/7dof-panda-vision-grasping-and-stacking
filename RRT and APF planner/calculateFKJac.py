import numpy as np
from math import pi

class FK_Jac():
    def dh_matrix(self, a, alpha, d, theta):
        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        return np.array([
            [cos_t, -sin_t * cos_a, sin_t * sin_a, a * cos_t],
            [sin_t, cos_t * cos_a, -cos_t * sin_a, a * sin_t],
            [0.0, sin_a, cos_a, d],
            [0.0, 0.0, 0.0, 1.0]])

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout

        self.a = np.array([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0,0.0, 0.0])
        self.alpha = np.array([0.0, -pi / 2, pi / 2, pi / 2, -pi / 2, pi / 2, pi / 2, 0.0, 0.0, 0.0])
        self.d = np.array([0.141, 0.192, 0.0, 0.316, 0.0, 0.384, 0.0, 0.21, 0.0, 0.0])
        self.theta_offset = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -pi / 4, 0.0, 0.0])

        self.joint_in_intermediate_frame = [
            np.array([0.0, 0.0, 0.0]),  # J1
            np.array([0.0, 0.0, 0.0]),  # J2
            np.array([0.0, 0.0, 0.195]),  # J3
            np.array([0.0, 0.0, 0.0]),  # J4
            np.array([0.0, 0.0, 0.125]),  # J5
            np.array([0.0, 0.0, -0.015]),  # J6
            np.array([0.0, 0.0, 0.051]),  # J7
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.1, -0.105]),
            np.array([0.0, -0.1, -0.105])# End Effector
        ]

    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        T_joints = np.zeros((10, 4, 4))
        jointPositions = np.zeros((10, 3))
        T0e = np.identity(4)
        T = T0e @ self.dh_matrix(self.a[0], self.alpha[0], self.d[0], self.theta_offset[0])
        T_joints[0, :, :] = T
        p = T @ np.r_[self.joint_in_intermediate_frame[0], 1.0]
        jointPositions[0, :] = p[:3]
        for i in range(1, 10):
            if i < 8:
                theta = self.theta_offset[i] + q[i - 1]
            else:
                theta = self.theta_offset[i]
            T = T @ self.dh_matrix(self.a[i], self.alpha[i], self.d[i], theta)
            p = T @ np.r_[self.joint_in_intermediate_frame[i], 1.0]
            jointPositions[i, :] = p[:3]
            T_joint = T.copy()
            T_joint[:, 3] = p.flatten()
            T_joints[i] = T_joint
        idx_to_move = 7
        indices = list(range(10))
        new_order = [i for i in indices if i != idx_to_move] + [idx_to_move]
        jointPositions = jointPositions[new_order, :]
        T_joints = T_joints[new_order, :, :]
        # Your code ends here
        return jointPositions, T_joints

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE

        return()
    
if __name__ == "__main__":

    fk = FK_Jac()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    joint_positions, T0e = fk.forward_expanded(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
