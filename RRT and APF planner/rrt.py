import numpy as np
from detectCollision import detectCollision
from loadmap import loadmap
from copy import deepcopy
from calculateFK import FK

fk = FK()

def check_config_valid(map_struct, q):
    jointPositions, _ = fk.forward(q)

    for i in range(len(jointPositions) - 1):
        p1 = jointPositions[i]
        p2 = jointPositions[i+1]
        linePt1 = np.array([p1])  # shape (1,3)
        linePt2 = np.array([p2])  # shape (1,3)
        # 遍历所有障碍物 (M×6)
        for box in map_struct.obstacles:
            if detectCollision(linePt1, linePt2, box)[0]:
                return True #有碰撞
    return False #无碰撞

def check_edge_valid(map_struct, q_near, q_new, joint_step = 0.05):
    q_near = np.asarray(q_near)
    q_new = np.asarray(q_new)
    dist = np.linalg.norm(q_new - q_near)
    if dist == 0:
        return check_config_valid(map_struct, q_near)
    num_steps = int(np.ceil(dist / joint_step))
    for k in range(num_steps + 1):
        alpha = k * 1 / num_steps
        q_interp = q_near + alpha * (q_new - q_near)
        if check_config_valid(map_struct, q_interp):
            return True
    return False


def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """
    # 0. 检查 start / goal 是否本身碰撞
    # =======================================
    start = np.asarray(start, dtype=float).reshape(-1)
    goal = np.asarray(goal, dtype=float).reshape(-1)
    if check_config_valid(map, start):
        print("❌ Start configuration in collision.")
        return np.zeros((0,7))

    if check_config_valid(map, goal):
        print("❌ Goal configuration in collision.")
        return np.zeros((0,7))

    # initialize path
    nodes = [start]
    parents = {0: None} #parent是哈希表

    max_iter = 6000
    step_size = 0.3
    goal_thresh = 0.3

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    for it in range(max_iter):
        #随机采点
        if np.random.rand() < 0.1:  # ⭐10% 概率取 goal
            q_rand = goal.copy()
        else:
            q_rand = lowerLim + (upperLim - lowerLim) * np.random.rand(7)
        #找到最近点
        nodes_arr = np.vstack(nodes)
        idx_near = np.argmin(np.linalg.norm(nodes_arr - q_rand, axis=1))
        q_near = nodes[idx_near]
        #移动步长，计算出新的点
        direction = q_rand- q_near
        dist = np.linalg.norm(direction)
        if dist < step_size:
            q_new = q_rand
        else:
            q_new = q_near + direction / dist * step_size
        #检查碰撞
        if check_edge_valid(map, q_near, q_new, 0.05):
            continue #检测出collision，跳过当此循环

        nodes.append(q_new)
        parents[len(nodes) - 1] = idx_near
        #检查是否接近goal

        if np.linalg.norm(q_new - goal) < goal_thresh:
            if check_edge_valid(map, q_new, goal):
                continue

            print("🎉 Goal Reached!")

            # 回溯路径

            final_path = [goal, q_new]
            cur = len(nodes) - 1
            while parents[cur] is not None:
                cur = parents[cur]
                final_path.append(nodes[cur])

            final_path.reverse()
            return np.vstack(final_path)

    print("❌ No path found.")
    return np.zeros((0,7))


if __name__ == '__main__':
    map_struct = loadmap("C:/Users/11729/Desktop/Penn/MEAM 5200/Lab4/maps/map1.txt")
    start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    goal =  np.array([1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))