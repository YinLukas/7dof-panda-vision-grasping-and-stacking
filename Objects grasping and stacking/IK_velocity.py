import numpy as np 
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE
 
    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))
    
    J = calcJacobian(q_in)
    x_dot = np.vstack((v_in, omega_in))
    
    valid_idx = ~np.isnan(x_dot.flatten());
    J_valid = J[valid_idx, :]
    x_valid = x_dot[valid_idx, :]
    
    if J_valid.size == 0:
       dq = np.zeros((7,1))
    else:
       dq = np.linalg.pinv(J_valid) @ x_valid
    return dq
