#!/usr/bin/env python
# Our own keyboard control script based on spot_ros
# because wasd.py isn't compatible with the spot_ros
# driver setup.
#
# This function 'synchro_se2_trajectory_point_command' is useful too.
# https://github.com/boston-dynamics/spot-sdk/blob/master/python/bosdyn-client/src/bosdyn/client/robot_command.py#L851
import time
import rospy
from geometry_msgs.msg import Twist
from rbd_spot.utils.keys import getch

TRANS_VEL = 0.5    # m/s
ROT_VEL = 0.5    # rad/s

class Move:
    def __init__(self, name, vtrans=(0.0, 0.0, 0.0), vrot=(0.0, 0.0, 0.0)):
        """
        Args:
            name (str): name of movement
            vtrans (tuple): (vx, vy, vz) translational velocity
            vrot (tuple): (vx, vy, vz) rotational velocity

        The geometry of Spot coordinates is documented here:
        https://dev.bostondynamics.com/docs/concepts/geometry_and_frames
        """
        self.name = name
        self.vtrans = vtrans
        self.vrot = vrot

    def to_message(self):
        m = Twist()
        tx, ty, tz = self.vtrans
        m.linear.x = tx
        m.linear.y = ty
        m.linear.z = tz
        rx, ry, rz = self.vrot
        m.angular.x = rx
        m.angular.y = ry
        m.angular.z = rz
        return m


def print_controls(controls):
    print("Controls:")
    for k in controls:
        print(f"[{k}]  [{controls[k].name}]")
    print("\n[h] help")
    print("[c]  quit\n")


def main():
    rospy.init_node("spot_keyboard_control")

    controls = {
        "w": Move("forward", vtrans=(TRANS_VEL, 0.0, 0.0)),
        "a": Move("left", vtrans=(0.0, TRANS_VEL, 0.0)),
        "d": Move("right", vtrans=(0.0, -TRANS_VEL, 0.0)),
        "s": Move("back", vtrans=(-TRANS_VEL, 0.0, 0.0)),
        "q": Move("turn_left", vrot=(0.0, 0.0, ROT_VEL)),
        "e": Move("turn_right", vrot=(0.0, 0.0, -ROT_VEL)),
    }
    print_controls(controls)

    pub = rospy.Publisher("/spot/cmd_vel", Twist, queue_size=10)
    max_rate = rospy.Rate(50)
    _start_time = time.time()
    while True:
        k = getch()
        if k == "c":
            print("bye.")
            break

        if k == "h":
            print_controls(controls)

        if k in controls:
            action = controls[k]
            m = action.to_message()
            pub.publish(m)
            print("%.3fs: %s" % (time.time() - _start_time, action.name))
        max_rate.sleep()

if __name__ == "__main__":
    main()
