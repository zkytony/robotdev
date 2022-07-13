# functions related to moving the robot's arm (including gripper)
import time
import bosdyn.client.lease
from bosdyn.api import arm_command_pb2, geometry_pb2, robot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b

def create_client(conn):
    return conn.ensure_client(RobotCommandClient.default_service_name)

def velocityCommand(conn, command_client, v_x, v_y, v_rot,
                    cmd_duration=0.125, mobility_params=None):
    """
    Send a velocity motion command to the robot.

    Args:
        v_x: Velocity in the X direction in meters
        v_y: Velocity in the Y direction in meters
        v_rot: Angular velocity around the Z axis in radians
        cmd_duration: (optional) Time-to-live for the command in seconds.  Default is 125ms (assuming 10Hz command rate).
    Returns:
        RobotCommandResponse, float (time taken)
    """
    if mobility_params is None:
        mobility_params = RobotCommandBuilder.mobility_params()
    end_time = time.time() + cmd_duration
    synchro_velocity_command = RobotCommandBuilder.synchro_velocity_command(
        v_x=v_x, v_y=v_y, v_rot=v_rot, params=mobility_params)

    with bosdyn.client.lease.LeaseKeepAlive(conn.lease_client, must_acquire=True):
        _start_time = time.time()
        command_id = command_client.robot_command(
            synchro_velocity_command, end_time_secs=end_time,
            timesync_endpoint=conn.timesync_endpoint)
        end_time = time.time() - _start_time
        return command_id, end_time
