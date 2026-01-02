import numpy as np
from lib.calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE
    fk = FK();
    T_list = fk.compute_Ai(q_in)
    joints, T0e = fk.forward(q_in)
    z_list = [T[:3, 2] for T in T_list]
    o_n = joints[-1]
    Jv = np.zeros((3, 7))
    Jw = np.zeros((3, 7))
    for i in range(0, 7):
        z_i = z_list[i]
        o_i = joints[i]
        Jw[:, i] = z_i
        Jv[:, i] = np.cross(z_i, o_n - o_i)
    J = np.vstack((Jv, Jw))
    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, 0, 0, 0, 0])
    print("Jacobian of end-effector:")
    print(np.round(calcJacobian(q),3))
    print("rank of Jacobian:")
    print(np.linalg.matrix_rank(calcJacobian(q)))
    fk = FK()
    joints, T0e = fk.forward(q)
    print("joints:")
    print(joints)
    J = calcJacobian(q)
    w = np.array([0, 0, 0, 0, 0, 0, 1])
    velocity = J@w
    print("velocity:")
    print(velocity)