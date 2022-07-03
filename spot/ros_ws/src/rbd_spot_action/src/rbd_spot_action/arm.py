# functions related to moving the robot's arm (including gripper)
import time
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
    can be specified with respect to the robot's body. Reference:
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


def _execute_arm_command(command, command_client, lease, wait=3.0):
    _start_time = time.time()
    # note that this gets the graph directly instead of a Response object
    command_id = command_client.robot_command(command, lease=lease.lease_proto)
    block_until_arm_arrives(command_client, command_id, wait)
    _used_time = time.time() - _start_time
    return command_id, _used_time
