# functions related to moving the robot's arm (including gripper)
import time
import bosdyn.client.lease
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.frame_helpers import (GRAV_ALIGNED_BODY_FRAME_NAME,
                                         VISION_FRAME_NAME,
                                         ODOM_FRAME_NAME,
                                         get_a_tform_b)

def create_client(conn):
    return conn.ensure_client(RobotCommandClient.default_service_name)

def unstow(conn, command_client):
    """unstow the arm. Blocks until the arm arrives"""
    unstow = RobotCommandBuilder.arm_ready_command()
    return _execute_arm_command(unstow, command_client, conn.lease_client)

def stow(conn, command_client):
    """unstow the arm. Blocks until the arm arrives"""
    stow = RobotCommandBuilder.arm_stow_command()
    return _execute_arm_command(stow, command_client, conn.lease_client)

def open_gripper(conn, command_client, level=None):
    """
    opens the gripper to the given leve. If level=None, then th
    gripper will fuly open
    """
    if level is None:
        gripper_command = RobotCommandBuilder.claw_gripper_open_command()
    else:
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(level)
    return _execute_arm_command(gripper_command, command_client, conn.lease_client, wait=1.0)

def close_gripper(conn, command_client):
    return open_gripper(conn, command_client, level=0.0)

def moveEETo(conn, command_client, robot_state_client, pose, seconds=3.0):
    """Moves arm end-effector to given pose. The pose should be
    with respect to the robot's body frame (+x forward, +y left, +z up).
    The pose could be either x, y, z or x, y, z, qx, qy, qz, qw.

    Note that as done by the Spot SDK example arm_simple, the pose
    sent to the robot would be transformed with respect to the odom
    frame (which is safer). Therefore, we need the robot_state_client
    to get up-to-date robot state."""
    if len(pose) not in {3, 7}:
        raise ValueError("Invalid pose.")

    x, y, z = pose[:3]
    if len(pose) == 3:
        qx = 0
        qy = 0
        qz = 0
        qw = 1
    else:
        qx, qy, qz, qw = pose[3:]

    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)
    # Rotation as a quaternion
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
    flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                            rotation=flat_body_Q_hand)

    robot_state = robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

    odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.position.x, odom_T_hand.position.y, odom_T_hand.position.z,
        odom_T_hand.rotation.w, odom_T_hand.rotation.x,
        odom_T_hand.rotation.y, odom_T_hand.rotation.z,
        VISION_FRAME_NAME, seconds)
    return _execute_arm_command(arm_command, command_client, conn.lease_client, wait=3.0)


def gazeAt(conn, command_client, robot_state_client, x, y, z):
    """Opens the gripper, and gaze at a target pose. The target pose
    can be specified with respect to the robot's body. Note that this
    specifies where to look, not where the hand will be. Reference:
    Spot SDK arm_gaze example. WARNING: ARM MAY MOVE QUICKLY

    For the same reason as moveEETo, will need robot_state_client"""
    robot_state = robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
    gaze_target_in_odom = odom_T_flat_body.transform_point(x=x, y=y, z=z)
    gaze_command = RobotCommandBuilder.arm_gaze_command(gaze_target_in_odom[0],
                                                        gaze_target_in_odom[1],
                                                        gaze_target_in_odom[2],
                                                        VISION_FRAME_NAME)
    # Make the open gripper RobotCommand
    gripper_command = RobotCommandBuilder.claw_gripper_open_command()

    # Combine the arm and gripper commands into one RobotCommand
    synchro_command = RobotCommandBuilder.build_synchro_command(
        gripper_command, gaze_command)

    # Send the request
    print("Requesting gaze.")
    return _execute_arm_command(synchro_command, command_client, conn.lease_client, wait=4.0)





def moveEEToWithBodyFollow(conn, command_client, robot_state_client, pose, seconds=3.0):
    """Moves arm end effector to given pose, with the body following the
    arm and moving to good positions. (See arm_with_body_follow SDK example).
    The pose is expected to be with respect to the body frame at the
    start of this process. Note that because the body follows, the pose
    that is sent to Spot SDK will be with respect to the odom frame, a fixed frame.
    Therefore, we need robot_state_client to fetch the transform to odom frame.

    The pose should be  with respect to the robot's body frame (+x forward,
    +y left, +z up). The pose could be either x, y, z or x, y, z, qx, qy, qz, qw.
    """
    if len(pose) not in {3, 7}:
        raise ValueError("Invalid pose.")

    x, y, z = pose[:3]
    if len(pose) == 3:
        qx = 0
        qy = 0
        qz = 0
        qw = 1
    else:
        qx, qy, qz, qw = pose[3:]

    hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)
    # Rotation as a quaternion
    flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)
    flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                            rotation=flat_body_Q_hand)

    robot_state = robot_state_client.get_robot_state()
    odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                     VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

    odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand)

    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_hand.position.x, odom_T_hand.position.y, odom_T_hand.position.z,
        odom_T_hand.rotation.w, odom_T_hand.rotation.x,
        odom_T_hand.rotation.y, odom_T_hand.rotation.z,
        VISION_FRAME_NAME, seconds)

    follow_arm_command = RobotCommandBuilder.follow_arm_command()
    # Somehow, with follow_arm_command added, will need to create a new lease keepalive object.
    synchro_command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)
    return _execute_arm_command(synchro_command, command_client, conn.lease_client, wait=3.0)


def _execute_arm_command(command, command_client, lease_client, wait=3.0):
    _start_time = time.time()
    # This method of using LeaseKeepAlive automatically handles LeaseUseError,
    # which seems to occur for arm following command
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True):
        command_id = command_client.robot_command(command)
        command_success = block_until_arm_arrives(command_client, command_id, wait)
    _used_time = time.time() - _start_time
    return command_success, _used_time
