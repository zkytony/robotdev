# functions related to moving the robot's arm (including gripper)
import time
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b

def create_client(conn):
    return conn.ensure_client(RobotCommandClient.default_service_name)

def unstow(conn, command_client):
    """unstow the arm. Blocks until the arm arrives"""
    unstow = RobotCommandBuilder.arm_ready_command()
    return _execute_arm_command(unstow, command_client, conn.lease)

def stow(conn, command_client):
    """unstow the arm. Blocks until the arm arrives"""
    stow = RobotCommandBuilder.arm_stow_command()
    return _execute_arm_command(stow, command_client, conn.lease)


def gazeAt(conn, command_client, x, y, z):
    """Opens the gripper, and gaze at a target pose. The target pose
    can be specified with respect to the robot's body. Note that this
    specifies where to look, not where the hand will be. Reference:
    Spot SDK arm_gaze example. WARNING: ARM MAY BEHAVE UNEXPECTEDLY"""
    gaze_command = RobotCommandBuilder.arm_gaze_command(x, y, z,
                                                        GRAV_ALIGNED_BODY_FRAME_NAME)
    # Make the open gripper RobotCommand
    gripper_command = RobotCommandBuilder.claw_gripper_open_command()

    # Combine the arm and gripper commands into one RobotCommand
    synchro_command = RobotCommandBuilder.build_synchro_command(
        gripper_command, gaze_command)

    # Send the request
    print("Requesting gaze.")
    return _execute_arm_command(synchro_command, command_client, conn.lease, wait=4.0)


def moveEETo(conn, command_client, pose, seconds=3.0):
    """Moves arm end-effector to given pose. By default, the pose is
    with respect to the robot's body frame (+x forward, +y left, +z up).
    The pose could be either x, y, z or x, y, z, qx, qy, qz, qw."""
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
    arm_command = RobotCommandBuilder.arm_pose_command(
        flat_body_T_hand.position.x, flat_body_T_hand.position.y, flat_body_T_hand.position.z,
        flat_body_T_hand.rotation.w, flat_body_T_hand.rotation.x,
        flat_body_T_hand.rotation.y, flat_body_T_hand.rotation.z,
        GRAV_ALIGNED_BODY_FRAME_NAME, seconds)
    return _execute_arm_command(arm_command, command_client, conn.lease, wait=3.0)



def _execute_arm_command(command, command_client, lease, wait=3.0):
    _start_time = time.time()
    # note that this gets the graph directly instead of a Response object
    command_id = command_client.robot_command(command, lease=lease.lease_proto)
    block_until_arm_arrives(command_client, command_id, wait)
    _used_time = time.time() - _start_time
    return command_id, _used_time
