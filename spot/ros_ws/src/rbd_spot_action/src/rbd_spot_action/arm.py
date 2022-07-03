# functions related to moving the robot's arm (including gripper)
import time
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)

def create_client(conn):
    return conn.ensure_client(RobotCommandClient.default_service_name)

def unstow(command_client, conn):
    """unstow the arm. Blocks until the arm arrives"""
    unstow = RobotCommandBuilder.arm_ready_command()
    _start_time = time.time()
    # note that this gets the graph directly instead of a Response object
    unstow_command_id = command_client.robot_command(unstow, lease=conn.lease.lease_proto)
    block_until_arm_arrives(command_client, unstow_command_id, 3.0)
    _used_time = time.time() - _start_time
    return unstow_command_id, _used_time
