#!/usr/bin/env python
# unstow the arm
import time
import rbd_spot
from bosdyn.client.robot_command import blocking_stand

def main():
    conn = rbd_spot.SpotSDKConn(sdk_name="ArmMoveClient",
                                take_lease=True)
    command_client = rbd_spot.arm.create_client(conn)
    robot_state_client = rbd_spot.state.create_client(conn)
    blocking_stand(command_client, timeout_sec=10)
    # NOTE: Don't try 0.0, 0.0, 0.0 ==> the robot will attampt
    # to look at its own center of gravity (body frame origin)
    # which is leads to unpleasant arm motion.

    # +x: forward
    # +y: left
    # +z: up

    # For reference:
    # x = 0.5 (almost touching the robot itself)
    # x = 1.0 (good number for testing)
    # x = 1.5 (too far ahead)
    print((1.0, 0.0, 0))
    rbd_spot.arm.moveEETo(conn, command_client, robot_state_client, (1.0, 0.0, 0))
    time.sleep(2.0)
    rbd_spot.arm.stow(conn, command_client)
    time.sleep(2.0)
    # y = 0.5 (good number for testing);
    #    note that because the origin of the body frame is at robot's
    #    center, the arm end effector will move to somewhere in the middle of the body,
    #    to the left
    # z = 0.35 (avoids awkward downward movement of z=0)
    print((0.0, 0.5, 0.35))
    rbd_spot.arm.moveEETo(conn, command_client, robot_state_client, (0.0, 0.5, 0.35))
    time.sleep(2.0)
    rbd_spot.arm.stow(conn, command_client)
    time.sleep(2.0)
    # Forward, right, a little up
    print((0.5, -0.5, 0.25))
    rbd_spot.arm.moveEETo(conn, command_client, robot_state_client, (0.5, -0.5, 0.25))
    time.sleep(2.0)
    rbd_spot.arm.stow(conn, command_client)
    time.sleep(2.0)

    # Test if the robot can move itself to reach a farther pose
    # answer: No it can't
    print((2.0, -0.5, 0.25))
    rbd_spot.arm.moveEETo(conn, command_client, robot_state_client, (2.0, 0.1, 0.25))
    time.sleep(2.0)
    rbd_spot.arm.stow(conn, command_client)
    time.sleep(2.0)

    # Now, do the body follow command. It should achieve the goal
    print((2.0, -0.5, 0.25))
    rbd_spot.arm.moveEEToWithBodyFollow(conn, command_client, robot_state_client, (2.0, 0.1, 0.25))
    time.sleep(2.0)
    rbd_spot.arm.stow(conn, command_client)
    time.sleep(2.0)

    # do the body follow command again. It should achieve the goal
    print((-2.0, 0.5, 0.4))
    rbd_spot.arm.moveEEToWithBodyFollow(conn, command_client, robot_state_client, (-2.0, 0.5, 0.4))
    time.sleep(2.0)
    rbd_spot.arm.stow(conn, command_client)
    time.sleep(2.0)

if __name__ == "__main__":
    main()
