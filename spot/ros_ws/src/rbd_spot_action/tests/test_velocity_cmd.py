#!/usr/bin/env python
# unstow the arm
import time
import rbd_spot
from bosdyn.client.robot_command import blocking_stand

TRANS_VEL = 0.5    # m/s
ROT_VEL = 0.5    # rad/s

def main():
    conn = rbd_spot.SpotSDKConn(sdk_name="VelocityCmdClient",
                                take_lease=True)
    command_client = rbd_spot.body.create_client(conn)
    blocking_stand(command_client, timeout_sec=10)

    rbd_spot.body.velocityCommand(conn, command_client, TRANS_VEL, 0.0, 0.0, duration=1.0)
    time.sleep(2)
    rbd_spot.body.velocityCommand(conn, command_client, 0.0, -TRANS_VEL, 0.0, duration=1.0)
    time.sleep(2)
    rbd_spot.body.velocityCommand(conn, command_client, 0.0, 0.0, ROT_VEL, duration=1.0)
    time.sleep(2)

if __name__ == "__main__":
    main()
