import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))
    dq_primary = IK_velocity(q_in, v_in, omega_in).reshape((7, 1))

    J_full = calcJacobian(q_in)
    task_mask = ~np.isnan(np.vstack((v_in, omega_in))).flatten()
    J_valid = J_full[task_mask, :]

    # Pseudoinverse and null-space projector based on valid Jacobian
    J_pseudo = J_valid.T @ np.linalg.inv(J_valid @ J_valid.T)
    N = np.eye(7) - J_pseudo @ J_valid

    dq_null = N @ b
    dq_total = dq_primary + dq_null

    return dq_total.reshape((1, 7))

