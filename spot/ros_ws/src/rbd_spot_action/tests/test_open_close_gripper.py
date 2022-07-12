#!/usr/bin/env python
# unstow the arm
import time
import rbd_spot
from bosdyn.client.robot_command import blocking_stand

def main():
    conn = rbd_spot.SpotSDKConn(sdk_name="UnstowArmClient",
                                take_lease=True)
    command_client = rbd_spot.arm.create_client(conn)
    rbd_spot.arm.open_gripper(conn, command_client)
    time.sleep(2.0)
    rbd_spot.arm.close_gripper(conn, command_client)
    time.sleep(2.0)

if __name__ == "__main__":
    main()
