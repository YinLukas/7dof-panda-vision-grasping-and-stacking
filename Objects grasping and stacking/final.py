import sys
import numpy as np
from copy import deepcopy
from math import pi
from lib.calculateFK import FK
from lib.IK_position_null import IK
import rospy

# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# For timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# Platform height and block height
Z_PLATFORM = 0.20
BLOCK_H    = 0.05

# ===================== Team-specific offsets =====================

OFFSETS_BY_TEAM = {
    "blue": {
        "static":  (0.01, 0.04, -0.015),   # (x, y, z) meters
        "dynamic": (0.00, 0.00, -0.02),
    },
    "red": {
        "static":  (0.01, -0.04, -0.015),
        "dynamic": (0.00, 0.00, -0.02),
    }
}


x_offset_static  = 0.0
y_offset_static  = 0.0
z_offset_static  = 0.0

x_offset_dynamic = 0.0
y_offset_dynamic = 0.0
z_offset_dynamic = 0.0


def apply_team_offsets(team: str):

    global x_offset_static, y_offset_static, z_offset_static
    global x_offset_dynamic, y_offset_dynamic, z_offset_dynamic

    team = team.lower()
    if team not in OFFSETS_BY_TEAM:
        raise ValueError(f"Unknown team: {team}. Must be 'red' or 'blue'.")

    xs, ys, zs = OFFSETS_BY_TEAM[team]["static"]
    xd, yd, zd = OFFSETS_BY_TEAM[team]["dynamic"]

    x_offset_static, y_offset_static, z_offset_static = xs, ys, zs
    x_offset_dynamic, y_offset_dynamic, z_offset_dynamic = xd, yd, zd


fk_solver = FK()
ik_solver = IK()


def ik_solve(T_target, q_seed):
    # ik_position_null will move the EE to the desired configuration and position
    q_sol, rollout, success, message = ik_solver.inverse(
        T_target,
        q_seed,
        method='J_pseudo',
        alpha=0.5
    )

    if not success:
        print("IK warning:", message)

    return q_sol


# Gripper control
def open_gripper(arm):
    arm.exec_gripper_cmd(0.08, 0.0)


def close_gripper(arm):
    arm.exec_gripper_cmd(0.0, 20.0)


Grip_min = 0.040
Grip_max = 0.060


def block_gripped(arm):
    state = arm.get_gripper_state()
    pos = state["position"]
    width = float(pos[0] + pos[1])   # Compute the distance between the two gripper fingers
    success = bool(Grip_min <= width <= Grip_max)
    return success


def compute_T0_block(q, T_c_block, detector):
    _, T0e_spot = fk_solver.forward(q)
    H_ee_camera = detector.get_H_ee_camera()
    T_e_block = H_ee_camera @ T_c_block
    T0_block = T0e_spot @ T_e_block
    return T0_block


def make_pick_poses_from_block(T0_block, R_grab):
    x = T0_block[0, 3]
    y = T0_block[1, 3]
    z = T0_block[2, 3]

    T_hover = np.eye(4)
    T_pick  = np.eye(4)
    T_lift  = np.eye(4)

    z_pick = Z_PLATFORM + BLOCK_H / 2

    z_hover = z_pick + 0.15
    z_lift  = z_pick + 0.15
    for T in [T_hover, T_pick, T_lift]:
        T[0:3, 0:3] = R_grab

    # Use static offset only
    T_hover[0:3, 3] = np.array([x + x_offset_static, y + y_offset_static, z_hover + z_offset_static])
    T_pick[0:3, 3]  = np.array([x + x_offset_static, y + y_offset_static, z_pick  + z_offset_static])
    T_lift[0:3, 3]  = np.array([x + x_offset_static, y + y_offset_static, z_lift  + z_offset_static])
    print("pick z =", z_pick)
    return T_hover, T_pick, T_lift


def make_stack_poses(R_approach, x_stack, y_stack, current_stack_number):
    """
    current_stack_number: number of blocks already on the tower
    Returns: T_hover_stack, T_place_stack, T_lift_stack
    """
    T_hover = np.eye(4)
    T_stack = np.eye(4)
    T_lift  = np.eye(4)

    z_top   = Z_PLATFORM + BLOCK_H * current_stack_number

    z_hover = z_top +  0.1
    z_stack = z_top + BLOCK_H / 2
    z_lift  = z_hover

    for T in [T_hover, T_stack, T_lift]:
        T[0:3, 0:3] = R_approach

    # Use static offset only
    T_hover[0:3, 3] = np.array([x_stack + x_offset_static, y_stack + y_offset_static, z_hover + z_offset_static])
    T_stack[0:3, 3] = np.array([x_stack + x_offset_static, y_stack + y_offset_static, z_stack + z_offset_static])
    T_lift[0:3, 3]  = np.array([x_stack + x_offset_static, y_stack + y_offset_static, z_lift  + z_offset_static])

    return T_hover, T_stack, T_lift


def move_to_target(arm, T_target, q):
    q_target = ik_solve(T_target, q)
    arm.safe_move_to_position(q_target)
    return q_target


def grab_and_stack(arm, detector, q_spot, q_stack_pose):
    """
    Grab all static blocks and stack them at the world coordinate position
    determined by q_stack_pose (compute the position via FK).

    Strategy:
      1) At q_spot, do a coarse detection to lock in XY
      2) Move with a fixed downward orientation to a hover pose above the block (coarse)
      3) Detect again at hover, refine the same block, and recompute R_grab (principal-axis method)
      4) Generate accurate T_hover/T_pick/T_lift and perform the grasp
      5) Stack with a fixed R_stack
    """
    current_stack_number = 0

    # ---------------------------
    #  (x_stack, y_stack)
    # ---------------------------
    _, T0_stack = fk_solver.forward(q_stack_pose)
    x_stack = T0_stack[0, 3]
    y_stack = T0_stack[1, 3]

    open_gripper(arm)

    while not rospy.is_shutdown():

        # Step 1:
        arm.safe_move_to_position(q_spot)
        q_current = q_spot.copy()

        # Step 2:
        detections = detector.get_detections()
        static_blocks = [(name, pose) for (name, pose) in detections
                         if "static" in name.lower()]

        print("\n[grab_and_stack] Detected blocks:")
        for (name, pose) in detections:
            print("   ", name)

        #
        if len(static_blocks) == 0:
            print("[grab_and_stack] No more static blocks. Done.")
            break

        #
        name, T_c_block = static_blocks[0]
        print(f"\n=== Grabbing (coarse): {name} ===")

        # Step 3:
        T0_block_coarse = compute_T0_block(q_current, T_c_block, detector)
        x_coarse = T0_block_coarse[0, 3]
        y_coarse = T0_block_coarse[1, 3]

        #
        R_down = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0,  0, -1]
        ])

        #
        T_hover_coarse, _, _ = make_pick_poses_from_block(T0_block_coarse, R_down)

        q_current = move_to_target(arm, T_hover_coarse, q_current)

        # ============================
        #
        # ============================
        detections2 = detector.get_detections()
        static_blocks2 = [(n, pose) for (n, pose) in detections2
                          if "static" in n.lower()]

        if len(static_blocks2) == 0:
            print("[grab_and_stack] No static blocks found at hover, retry...")
            continue

        T_c_block_refined = None
        min_dist = float("inf")

        for (n, pose) in static_blocks2:

            T0_tmp = compute_T0_block(q_current, pose, detector)
            dx = T0_tmp[0, 3] - x_coarse
            dy = T0_tmp[1, 3] - y_coarse
            d = np.hypot(dx, dy)

            if n == name:

                T_c_block_refined = pose
                break

            if d < min_dist:
                min_dist = d
                T_c_block_refined = pose

        if T_c_block_refined is None:
            print("[grab_and_stack] Could not refine block pose, skip.")
            continue

        print(f"[grab_and_stack] Refined detection for {name}")

        T0_block = compute_T0_block(q_current, T_c_block_refined, detector)

        # -----------------------------
        #
        # -----------------------------
        R_block_world = T0_block[0:3, 0:3]

        axis_x = R_block_world[:, 0]
        axis_y = R_block_world[:, 1]

        px = np.linalg.norm(axis_x[0:2])
        py = np.linalg.norm(axis_y[0:2])

        #
        main_axis = axis_x if px > py else axis_y

        theta = np.arctan2(main_axis[1], main_axis[0])

        #
        theta_small = (theta + np.pi/4) % (np.pi/2) - np.pi/4
        theta = theta_small

        R_align = np.array([
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta),  np.cos(theta), 0.0],
            [0.0, 0.0, 1.0]
        ])

        #
        R_0_e = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0,  0, -1]
        ])

        #
        R_grab = R_align @ R_0_e

        # Step 4:
        T_hover, T_pick, T_lift = make_pick_poses_from_block(T0_block, R_grab)

        # Step 5: hover → pick
        q_current = move_to_target(arm, T_hover, q_current)
        q_current = move_to_target(arm, T_pick,  q_current)

        # Step 6:
        close_gripper(arm)
        rospy.sleep(0.3)

        # Step 7:
        q_current = move_to_target(arm, T_lift, q_current)

        # Step 8:
        if not block_gripped(arm):
            print("[grab_and_stack] Grabbing failed, retry next block.")
            open_gripper(arm)
            continue

        print("[grab_and_stack] Successfully grabbed", name)

        # ---------------------------
        # Step 9: STACKING (fixed R_stack)
        # ---------------------------
        R_stack = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0,  0, -1]
        ])

        print(f"[grab_and_stack] Stacking block #{current_stack_number} ...")
        T_hover_stack, T_stack, T_lift_stack = make_stack_poses(
            R_stack, x_stack, y_stack, current_stack_number
        )

        q_current = move_to_target(arm, T_hover_stack, q_current)
        q_current = move_to_target(arm, T_stack,      q_current)

        open_gripper(arm)
        rospy.sleep(0.25)

        q_current = move_to_target(arm, T_lift_stack, q_current)

        print("[grab_and_stack] Stacked", name)
        current_stack_number += 1

    print("\n[grab_and_stack] Finished grabbing all static blocks.")
    return current_stack_number


# ==================== Dynamic block grasp parameters ====================
# Turntable center position (world coordinate origin)
TURNTABLE_CENTER_WORLD = np.array([0.0, 0.0, Z_PLATFORM])

# Block radius on the turntable (most blocks are near the rim)
# Adjust this value based on the actual simulation if needed
BLOCK_RADIUS_ON_TURNTABLE = 0.30  # radius from turntable center (meters)

# Gripper open/close cycle timing (tune based on turntable speed)
GRIPPER_CYCLE_TIME = 1.5  # time to wait after opening (seconds)
GRIPPER_CLOSE_DURATION = 0.5  # time to hold closed before checking (seconds)

# Pick height
DYNAMIC_PICK_HEIGHT = Z_PLATFORM + BLOCK_H / 2  # block center height


def get_dynamic_grasp_position(team):
    """
    Compute a fixed grasp position for dynamic blocks (in the base frame).

    The arm waits at a fixed point on the turntable rim and grabs blocks as they pass.

    Args:
        team: 'red' or 'blue'

    Returns:
        grasp_position: [x, y, z] grasp position (base frame)
    """
    # Turntable center in world is (0, 0)
    # Red team: base_y = world_y + 0.99, center is at base (0, 0.99)
    # Blue team: base_y = world_y - 0.99, center is at base (0, -0.99)

    if team == 'red':
        turntable_center_base = np.array([0.0, 0.99])
    else:
        turntable_center_base = np.array([0.0, -0.99])

    # Grasp point: on the rim, facing toward the robot
    if team == 'red':
        grasp_y = turntable_center_base[1] - BLOCK_RADIUS_ON_TURNTABLE
    else:
        grasp_y = turntable_center_base[1] + BLOCK_RADIUS_ON_TURNTABLE

    # Keep x aligned with the turntable centerline
    grasp_x = turntable_center_base[0]

    # Dynamic grasp point uses: static_offset + dynamic_offset (per team)
    gx = grasp_x + x_offset_static + x_offset_dynamic
    gy = grasp_y + y_offset_static + y_offset_dynamic
    gz = DYNAMIC_PICK_HEIGHT + z_offset_static + z_offset_dynamic

    grasp_position = np.array([gx, gy, gz])
    return grasp_position


def get_dynamic_ready_pose(team):
    """
    Get a ready pose facing the turntable (predefined joint configuration).
    This pose is used as an IK seed to ensure the arm faces the turntable.

    Args:
        team: 'red' or 'blue'

    Returns:
        q_ready: 7 joint angles
    """
    if team == 'blue':
        q_ready = np.array([
            -0.6,
            0.0,
            0.0,
            -pi / 2,
            0.0,
            pi / 2,
            pi / 4
        ])
    else:
        q_ready = np.array([
            0.6,
            0.0,
            0.0,
            -pi / 2,
            0.0,
            pi / 2,
            pi / 4
        ])

    return q_ready


def get_dynamic_approach_waypoints(team):
    """
    Get transition waypoints from the ready pose to the turntable grasp pose.
    """
    if team == 'blue':
        waypoint1 = np.array([
            -0.9,
            0.2,
            0.0,
            -1.8,
            0.0,
            1.6,
            pi / 4
        ])

        waypoint2 = np.array([
            -1.2,
            0.3,
            0.0,
            -1.5,
            0.0,
            1.8,
            pi / 4
        ])
    else:
        waypoint1 = np.array([
            0.9,
            0.2,
            0.0,
            -1.8,
            0.0,
            1.6,
            pi / 4
        ])

        waypoint2 = np.array([
            1.2,
            0.3,
            0.0,
            -1.5,
            0.0,
            1.8,
            pi / 4
        ])

    return [waypoint1, waypoint2]


def move_through_waypoints(arm, waypoints, sleep_time=0.1):
    """
    Move the arm through multiple waypoints in sequence.
    """
    for i, wp in enumerate(waypoints):
        print(f"  Moving through waypoint {i + 1}/{len(waypoints)}...")
        arm.safe_move_to_position(wp)
        rospy.sleep(sleep_time)


# ==================== Dynamic block grasping procedure ====================
def grab_dynamic_blocks(arm, detector, q_spot, q_stack_pose, initial_stack_number, team):
    """
    Grab dynamic blocks and stack them using a fixed-position periodic grabbing strategy.
    """
    current_stack_number = initial_stack_number

    # Compute the stacking position in world coordinates via FK
    _, T0_stack = fk_solver.forward(q_stack_pose)
    x_stack = T0_stack[0, 3]
    y_stack = T0_stack[1, 3]

    # Compute fixed grasp position
    grasp_pos = get_dynamic_grasp_position(team)
    print(f"\n{'=' * 50}")
    print(f"Dynamic Grasp Strategy: Fixed Position Periodic Grabbing")
    print(f"{'=' * 50}")
    print(f"Grasp position (base frame): x={grasp_pos[0]:.3f}, y={grasp_pos[1]:.3f}, z={grasp_pos[2]:.3f}")
    print(f"Gripper cycle time: {GRIPPER_CYCLE_TIME}s")

    # Get the ready pose facing the turntable
    q_ready = get_dynamic_ready_pose(team)

    # Get transition waypoints
    waypoints = get_dynamic_approach_waypoints(team)

    # Build grasp orientation (EE pointing down)
    R_grab = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0,  0, -1]
    ])

    # Hover pose (10 cm above grasp point)
    T_hover = np.eye(4)
    T_hover[0:3, 0:3] = R_grab
    T_hover[0, 3] = grasp_pos[0]
    T_hover[1, 3] = grasp_pos[1]
    T_hover[2, 3] = grasp_pos[2] + 0.10

    # Pick pose (grasp point)
    T_pick = np.eye(4)
    T_pick[0:3, 0:3] = R_grab
    T_pick[0, 3] = grasp_pos[0]
    T_pick[1, 3] = grasp_pos[1]
    T_pick[2, 3] = grasp_pos[2]

    # Lift pose (15 cm above grasp point)
    T_lift = np.eye(4)
    T_lift[0:3, 0:3] = R_grab
    T_lift[0, 3] = grasp_pos[0]
    T_lift[1, 3] = grasp_pos[1]
    T_lift[2, 3] = grasp_pos[2] + 0.15

    max_attempts = 50  # Maximum number of attempts
    attempts = 0
    max_dynamic_blocks = 8  # Keep the original value (8)

    # ===== Move to ready pose first =====
    print("Moving to ready position facing turntable...")
    arm.safe_move_to_position(q_ready)
    q_current = q_ready.copy()
    rospy.sleep(0.3)

    while not rospy.is_shutdown() and attempts < max_attempts:
        dynamic_grabbed = current_stack_number - initial_stack_number
        if dynamic_grabbed >= max_dynamic_blocks:
            print(f"\nAll {max_dynamic_blocks} dynamic blocks grabbed!")
            break

        attempts += 1

        print(f"\n{'=' * 50}")
        print(f"Dynamic Grab Attempt {attempts}/{max_attempts}")
        print(f"Dynamic blocks grabbed so far: {dynamic_grabbed}/{max_dynamic_blocks}")
        print(f"{'=' * 50}")

        # ===== Step 1: Move through approach waypoints =====
        print("Moving through approach waypoints...")
        open_gripper(arm)
        rospy.sleep(0.2)
        move_through_waypoints(arm, waypoints)
        q_current = waypoints[-1].copy()  # Use last waypoint as IK seed

        # ===== Step 2: Move to hover pose =====
        print("Moving to hover position above turntable...")
        try:
            q_current = move_to_target(arm, T_hover, q_current)
        except Exception as e:
            print(f"Failed to move to hover: {e}")
            print("Returning to ready position...")
            arm.safe_move_to_position(q_ready)
            q_current = q_ready.copy()
            continue

        # ===== Step 3: Lower to grasp pose =====
        print("Lowering to grasp position...")
        try:
            q_current = move_to_target(arm, T_pick, q_current)
        except Exception as e:
            print(f"Failed to move to pick position: {e}")
            print("Returning to ready position...")
            arm.safe_move_to_position(q_ready)
            q_current = q_ready.copy()
            continue

        # ===== Step 4: Periodic grab cycles =====
        grab_success = False
        cycle_attempts = 0
        max_cycle_attempts = 8

        print(f"Starting periodic grab cycles (max {max_cycle_attempts} cycles)...")

        while cycle_attempts < max_cycle_attempts and not grab_success:
            cycle_attempts += 1

            print(f"  Cycle {cycle_attempts}: Closing gripper...")
            close_gripper(arm)
            rospy.sleep(GRIPPER_CLOSE_DURATION)

            if block_gripped(arm):
                print(f"  >>> Block detected in gripper!")
                grab_success = True
                break
            else:
                print(f"  No block detected, opening gripper...")
                open_gripper(arm)
                rospy.sleep(GRIPPER_CYCLE_TIME)

        if not grab_success:
            print("No block grabbed after all cycles. Retrying...")
            open_gripper(arm)
            arm.safe_move_to_position(q_ready)
            q_current = q_ready.copy()
            continue

        # ===== Step 5: Lift the block =====
        print("Lifting block...")
        try:
            q_current = move_to_target(arm, T_lift, q_current)
        except Exception as e:
            print(f"Failed to lift: {e}")
            open_gripper(arm)
            arm.safe_move_to_position(q_ready)
            q_current = q_ready.copy()
            continue

        if not block_gripped(arm):
            print("Block dropped during lift!")
            open_gripper(arm)
            arm.safe_move_to_position(q_ready)
            q_current = q_ready.copy()
            continue

        print("Successfully grabbed dynamic block!")

        # ===== Step 6: Return through waypoints to a safe pose =====
        print("Returning through waypoints to safe position...")
        reversed_waypoints = waypoints[::-1]
        move_through_waypoints(arm, reversed_waypoints)

        arm.safe_move_to_position(q_ready)
        q_current = q_ready.copy()

        if not block_gripped(arm):
            print("Block dropped during return transition!")
            open_gripper(arm)
            continue

        # ===== Step 7: Move to stack ready pose =====
        print("Moving to stack ready position...")
        arm.safe_move_to_position(q_stack_pose)
        q_current = q_stack_pose.copy()

        if not block_gripped(arm):
            print("Block dropped during stack transition!")
            open_gripper(arm)
            arm.safe_move_to_position(q_ready)
            q_current = q_ready.copy()
            continue

        # ===== Step 8: Stack =====
        R_stack = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0,  0, -1]
        ])

        print(f"Stacking block #{current_stack_number + 1}...")
        T_hover_stack, T_stack, T_lift_stack = make_stack_poses(
            R_stack, x_stack, y_stack, current_stack_number
        )

        try:
            q_current = move_to_target(arm, T_hover_stack, q_current)
            q_current = move_to_target(arm, T_stack, q_current)
        except Exception as e:
            print(f"Failed to move to stack position: {e}")
            open_gripper(arm)
            arm.safe_move_to_position(q_ready)
            q_current = q_ready.copy()
            continue

        print("Releasing block...")
        open_gripper(arm)
        rospy.sleep(0.25)

        q_current = move_to_target(arm, T_lift_stack, q_current)

        print(f"Stacked dynamic block #{current_stack_number + 1} (total: {current_stack_number + 1})")
        current_stack_number += 1

        # ===== Step 9: Return to ready pose =====
        print("Returning to ready position...")
        arm.safe_move_to_position(q_ready)
        q_current = q_ready.copy()

        # Reset attempts after a success (keep original logic)
        attempts = 0

    print(f"\nFinished grabbing dynamic blocks.")
    print(f"Dynamic blocks grabbed: {current_stack_number - initial_stack_number}")
    print(f"Total blocks stacked: {current_stack_number}")
    return current_stack_number


# ==================== Main program ====================
if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    # ===== Apply team-specific static/dynamic offsets =====
    apply_team_offsets(team)
    print(f"[offset] team={team} "
          f"static=({x_offset_static:.3f},{y_offset_static:.3f},{z_offset_static:.3f}) "
          f"dynamic=({x_offset_dynamic:.3f},{y_offset_dynamic:.3f},{z_offset_dynamic:.3f})")

    rospy.init_node("team_script")

    arm = ArmController()
    detector = ObjectDetector()

    # ===== Initial ready joint configuration =====
    start_position = np.array([
        -0.01779206, -0.76012354,  0.01978261,
        -2.34205014,  0.02984053,  1.54119353 + pi/2,
        0.75344866
    ])
    arm.safe_move_to_position(start_position)

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")
    print("\nStarting in 3 seconds...\n")
    rospy.sleep(3.0)
    print("Go!\n")

    # ================================
    # 1) Static: spot pose + stack pose
    # ================================
    if team == 'blue':
        q_static_spot = np.array([
            0.1345, -0.0507, 0.1806,
            -1.6082, 0.0091, 1.5583, 1.1004
        ])
        q_stack_pose = np.array([
            -0.1354, 0.1106, -0.1694,
            -2.0057, 0.0218, 2.1146, 0.4704
        ])
    else:
        q_static_spot = np.array([
            -0.1645, -0.0505, -0.1516,
            -1.6082, -0.0076, 1.5583, 0.4693
        ])
        q_stack_pose = np.array([
            0.2050, 0.1096, 0.0977,
            -2.0057, -0.0125, 2.1147, 1.0939
        ])

    # ================================
    # 2) Static blocks: build a tower on the target platform first
    # ================================
    print("[main] Starting STATIC block stacking...\n")

    static_block_count = grab_and_stack(
        arm=arm,
        detector=detector,
        q_spot=q_static_spot,
        q_stack_pose=q_stack_pose
    )

    if static_block_count is None:
        static_block_count = 4

    print(f"\n[main] Static stacking completed. Static blocks: {static_block_count}\n")

    # ================================
    # 3) Dynamic blocks: grab from the turntable and continue stacking on the same tower
    # ================================
    print("[main] Starting DYNAMIC block stacking...\n")

    final_block_count = grab_dynamic_blocks(
        arm=arm,
        detector=detector,
        q_spot=q_static_spot,            # Not used in the current implementation
        q_stack_pose=q_stack_pose,       # Same stacking pose as the static tower
        initial_stack_number=static_block_count,
        team=team
    )

    print(f"\n[main] Dynamic stacking completed. Total blocks on tower: {final_block_count}\n")
    print("[main] Program finished.")