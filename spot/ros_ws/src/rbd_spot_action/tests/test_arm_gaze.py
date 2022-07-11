#!/usr/bin/env python
# unstow the arm
import time
import rbd_spot
from bosdyn.client.robot_command import blocking_stand

def main():
    conn = rbd_spot.SpotSDKConn(sdk_name="ArmGazeClient",
                                take_lease=True)
    command_client = rbd_spot.arm.create_client(conn)
    robot_state_client = rbd_spot.state.create_client(conn)
    blocking_stand(command_client, timeout_sec=10)
    # NOTE: Don't try 0.0, 0.0, 0.0 ==> the robot will attampt
    # to look at its own center of gravity (body frame origin)
    # which is leads to unpleasant arm motion.
    # Look forward
    rbd_spot.arm.gazeAt(conn, command_client, robot_state_client, 3.0, 0.0, 0)
    time.sleep(2.0)
    # Look to the left
    rbd_spot.arm.gazeAt(conn, command_client, robot_state_client, 3.0, 4.0, 0)
    time.sleep(2.0)

if __name__ == "__main__":
    main()
