import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
#from calculateFKJac import FK_Jac
from lib.calculateFKJac import FK_Jac
#from detectCollision import detectCollision
from lib.detectCollision import detectCollision
#from loadmap import loadmap
from lib.loadmap import loadmap

fk = FK_Jac()

class PotentialFieldPlanner:

    # JOINT LIMITS
    tau_norm_last = 1.0
    lower = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upper = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    center = lower + (upper - lower) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self, tol=0.02, max_steps= 800, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size


    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the
        target joint position and the current joint position

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint
        from the current position to the target position
        """

        ## STUDENT CODE STARTS HERE
        # parameters
        zeta = 1  # attractive potential gain
        d_star = 0.12  # distance threshold

        # difference vector
        diff = target - current
        dist = np.linalg.norm(diff)

        # attractive force
        if dist <= d_star:
            # quadratic region
            att_f = zeta * diff
        else:
            # conic region
            att_f = (diff / dist)

        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the
        obstacle and the current joint position

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position
        to the closest point on the obstacle box

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE

        rep_f = np.zeros((3, 1))
        # Convert current joint position to (1,3)
        p = current.reshape(1, 3)

        # Compute distance and direction using the provided helper function
        dist, unit = PotentialFieldPlanner.dist_point2box(p, obstacle)
        dist = dist[0]  # scalar
        unit = unit[0].reshape(3, 1)  # convert to (3,1)

        rho0 = 0.005#0.15
        etta = 0.002
        if dist <= 0 or dist > rho0:
            return rep_f

        magnitude = etta * (1.0 / dist - 1.0 / rho0) * (1.0 / (dist * dist))
        rep_f = - magnitude * unit

        ## END STUDENT CODE

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point

        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit


    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum
        of forces (attactive, repulsive) on each joint.

        INPUTS:
        target - 3x9 numpy array representing the desired joint/end effector positions
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x9 numpy array representing the current joint/end effector positions
        in the world frame

        OUTPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE
        joint_forces = np.zeros((3, 9))
        obstacle = np.asarray(obstacle)
        if obstacle.size == 0:
            obstacle_list = []
        elif obstacle.ndim == 1:
            obstacle_list = [obstacle.reshape(6,)]
        else:
            obstacle_list = []
            for k in range(obstacle.shape[0]):
                box_k = obstacle[k, :].reshape(6,)
                obstacle_list.append(box_k)

        for i in range(9):
            att_f = PotentialFieldPlanner.attractive_force(target[:, i].reshape(3,1), current[:, i].reshape(3,1))
            if i <= 6:
                zeta_i = 30.0
            else:
                zeta_i = 12.0
            att_f = zeta_i * att_f
            rep_sum = np.zeros((3,1))
            for box in obstacle_list:
                dist, unit = PotentialFieldPlanner.dist_point2box(current[:, i].reshape(1,3), box)
                unitvec = unit[0].reshape(3,1)
                rep_f = PotentialFieldPlanner.repulsive_force(box, current[:, i].reshape(3,1), unitvec)
                rep_sum += rep_f

            joint_forces[:, i] = (att_f + rep_sum).reshape(3,)
        ## END STUDENT CODE
        return joint_forces

    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum
        of torques on each joint.

        INPUTS:
        joint_forces - 3x9 numpy array representing the force vectors on each
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x9 numpy array representing the torques on each joint
        """

        ## STUDENT CODE STARTS HERE
        joints, T_list = fk.forward_expanded(q)

        z_axes = [T_list[j][:3,2] for j in range(10)]

        joint_torques = np.zeros(9)
        for i in range(1, 10):  # each force point
            F_i = joint_forces[:, i-1]  # (3,)
            o_i = joints[i]  # (3,)
            J_i = np.zeros((3, 9))
            for j in range(i):
                z_j = z_axes[j]
                o_j = joints[j]
                J_i[:, j] = np.cross(z_j, (o_i - o_j))
            joint_torques += (J_i.T @ F_i)
        ## END STUDENT CODE

        return joint_torques.reshape(1,9)

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets

        """

        ## STUDENT CODE STARTS HERE

        distance = np.linalg.norm(target - current)
        ## END STUDENT CODE

        return distance

    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal
        configuration

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task.
        """

        ## STUDENT CODE STARTS HERE

        dq = np.zeros((1, 7))
        current_pos, _ = fk.forward_expanded(q)  # 10*3
        current_pos = current_pos[1:10, :].T      # shape = (3,9)
        target_pos, _ = fk.forward_expanded(target)  # 10*3
        target_pos = target_pos[1:10, :].T
        obstacles = map_struct.obstacles
        #obstacles = map_struct["obstacles"]
        joint_forces = PotentialFieldPlanner.compute_forces(target_pos,obstacles,current_pos)
        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)
        tau = joint_torques[0]  # shape (9,) → first 7 entries matter
        # Only first 7 torques correspond to revolute joints
        tau = tau[:7]
        alpha = 0.02
        norm = np.linalg.norm(tau)
        if norm > 1e-6:
            univector = tau / norm
            dq = alpha * univector
        else:
            dq = np.zeros((1, 7))  # 或者保持原样
        #univector = tau/norm
        # 保留 torque 大小
        #dq = alpha * univector  # shape (7,)

        # 步长 clip，避免太大跳动
        #max_step = 0.05
        #norm = np.linalg.norm(dq)
        #if norm > max_step:
           # dq = dq / norm * max_step

        dq = dq.reshape(1, 7)
        PotentialFieldPlanner.tau_norm_last = norm
        ## END STUDENT CODE

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def check_config_valid(map_struct, q):
        q = np.asarray(q).flatten()
        jointPositions, _ = fk.forward_expanded(q)

        for i in range(len(jointPositions) - 1):
            p1 = jointPositions[i]
            p2 = jointPositions[i + 1]
            linePt1 = np.array([p1])  # shape (1,3)
            linePt2 = np.array([p2])  # shape (1,3)
            # 遍历所有障碍物 (M×6)
            #for box in map_struct["obstacles"]:
            for box in map_struct.obstacles:
                if detectCollision(linePt1, linePt2, box)[0]:
                    return True  #有碰撞
        return False  #无碰撞

    def check_edge_valid(map_struct, q_near, q_new, joint_step=0.005):
        q_near = np.asarray(q_near)
        q_new = np.asarray(q_new)
        dist = np.linalg.norm(q_new - q_near)
        if dist == 0:
            return PotentialFieldPlanner.check_config_valid(map_struct, q_near)
        num_steps = int(np.ceil(dist / joint_step))
        for k in range(num_steps + 1):
            alpha = k * 1 / num_steps
            q_interp = q_near + alpha * (q_new - q_near)
            if PotentialFieldPlanner.check_config_valid(map_struct, q_interp):
                return True
        return False

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the startng configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes
        start - 1x7 numpy array representing the starting joint angles for a configuration
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles.
        """

        q_curr = start.astype(float).flatten()
        q_path = np.array([q_curr])  # shape (1,7)

        # 参数从 __init__ 读取
        tol = self.tol
        max_steps = self.max_steps
        min_step_size = self.min_step_size

        # 初始距离
        dist_curr = PotentialFieldPlanner.q_distance(goal, q_curr.flatten())

        stagnant_steps = 0  # 用来监测局部极小值
        step = 0
        while step < max_steps:

            ## STUDENT CODE STARTS HERE

            # The following comments are hints to help you to implement the planner
            # You don't necessarily have to follow these steps to complete your code

            # Compute gradient
            # TODO: this is how to change your joint angles
            dq = PotentialFieldPlanner.compute_gradient(q_curr.flatten(), goal, map_struct)
            step_norm = np.linalg.norm(dq)
            # Termination Conditions
            if dist_curr < tol: # TODO: check termination conditions
                break # exit the while loop if conditions are met!
            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            tau_norm = PotentialFieldPlanner.tau_norm_last

            if tau_norm < 0.001 and dist_curr > tol:
                stagnant_steps += 1
            else:
                stagnant_steps = 0

                # ===== random escape if stuck =====
            if stagnant_steps > 10:
                # fixed-length random direction (0.02)
                v = np.random.rand(1, 7) - 0.5
                v = v / np.linalg.norm(v)
                q_next = q_curr + 0.1 * v
            else:
                q_next = q_curr + dq.flatten()

            # 5) joint limits
            q_next = np.clip(q_next, self.lower, self.upper)

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function
            while PotentialFieldPlanner.check_edge_valid(map_struct, q_curr, q_next, 0.005):
                v = np.random.rand(1, 7) - 0.5
                v = v / np.linalg.norm(v)
                q_next = q_curr + 0.1 * v
                q_next = np.clip(q_next, self.lower, self.upper)

            q_path = np.vstack((q_path, q_next.reshape(1, 7)))
            q_curr = q_next
            dist_curr = PotentialFieldPlanner.q_distance(goal, q_curr.flatten())
            step += 1

        if PotentialFieldPlanner.q_distance(goal, q_curr.flatten()) < tol:
            q_path[-1, :] = goal.reshape(1, 7)
            print("🎉 Path Found!")
            ## END STUDENT CODE

        return q_path

################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()

    # inputs
    map_struct = loadmap("C:/Users/11729/Desktop/Penn/MEAM 5200/Lab4/maps/map4.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    #start = np.array([-1,  -1.3  , 1.5 , -2,       -1, 1.57, 0.2])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])

    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))

    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
