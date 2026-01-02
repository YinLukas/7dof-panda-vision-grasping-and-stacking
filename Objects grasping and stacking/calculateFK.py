import numpy as np
from math import pi, sin, cos


class FK():
   def dh_matrix(self, a, alpha, d, theta):
       cos_a, sin_a = np.cos(alpha), np.sin(alpha)
       cos_t, sin_t = np.cos(theta), np.sin(theta)
       return np.array([
            [cos_t, -sin_t*cos_a, sin_t*sin_a, a*cos_t],
            [sin_t, cos_t*cos_a, -cos_t*sin_a, a*sin_t],
            [0.0, sin_a, cos_a, d],
            [0.0, 0.0, 0.0, 1.0]])
   def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        self.a = np.array([ 0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0])
        self.alpha = np.array([ 0.0, -pi/2, pi/2, pi/2, -pi/2, pi/2, pi/2, 0.0])
        self.d = np.array([ 0.141, 0.192, 0.0, 0.316, 0.0, 0.384, 0.0, 0.21])
        self.theta_offset = np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -pi/4])

        self.joint_in_intermediate_frame = [
            np.array([0.0, 0.0, 0.0]),  # J1
            np.array([0.0, 0.0, 0.0]),  # J2
            np.array([0.0, 0.0, 0.195]),  # J3
            np.array([0.0, 0.0, 0.0]),  # J4
            np.array([0.0, 0.0, 0.125]),  # J5
            np.array([0.0, 0.0, -0.015]),  # J6
            np.array([0.0, 0.0, 0.051]),  # J7
            np.array([0.0, 0.0, 0.0])   # End Effector
        ]

   def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here
        jointPositions = np.zeros((8, 3))
        T0e = np.identity(4)
        T = T0e @ self.dh_matrix(self.a[0], self.alpha[0], self.d[0], self.theta_offset[0])
        p = T @ np.r_[self.joint_in_intermediate_frame[0], 1.0]
        jointPositions[0, :] = p[:3]
        for i in range(1, 8):
            theta = self.theta_offset[i] + q[i - 1]
            T = T @ self.dh_matrix(self.a[i], self.alpha[i], self.d[i], theta)
            p = T @ np.r_[self.joint_in_intermediate_frame[i], 1.0]
            jointPositions[i, :] = p[:3]
        T0e = T
        # Your code ends here
        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    # This code is for Lab 2, you can ignore it ofr Lab 1
   def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        Ai_list = self.compute_Ai(q)
        axis_of_rotation_list = np.array((3,7))
        for i in range(0,7):
            R_0i = Ai_list[i][:3, :3]
            z_axis = R_0i[:, 2]
            axis_of_rotation_list[:, i] = z_axis

        return axis_of_rotation_list

   def compute_Ai(self, q):
        """
        Return list of cumulative transforms T_0i (i=1..8)
        """
        T_list = []
        T = np.eye(4)
        for i in range(8):
            theta = self.theta_offset[i] + (q[i - 1] if i > 0 and i < 8 else 0)
            T = T @ self.dh_matrix(self.a[i], self.alpha[i], self.d[i], theta)
            T_list.append(T.copy())
        return T_list


if __name__ == "__main__":
    fk = FK()

    # matches figure in the handout
    q = np.array([0, 0, 0, -pi / 2, 0, pi / 2, pi / 4])

    joint_positions, T0e = fk.forward(q)


    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
