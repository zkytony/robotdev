#!/usr/bin/env python
# unstow the arm
import time
import rbd_spot
from bosdyn.client.robot_command import blocking_stand

def main():
    conn = rbd_spot.SpotSDKConn(sdk_name="UnstowArmClient",
                                take_lease=True)
    command_client = rbd_spot.arm.create_client(conn)
    rbd_spot.arm.unstow(conn, command_client)
    time.sleep(3.0)
    rbd_spot.arm.stow(conn, command_client)
    time.sleep(3.0)

if __name__ == "__main__":
    main()
