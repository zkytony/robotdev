#!/usr/bin/env python
#
# Publish to angular_vel_cmd
# Publish to velocity_vel_cmd
# /author: Kaiyu Zheng
# NOTE 01-14-2022: this script does not seem to be working

import rospy
from std_msgs.msg import Header
from movo_msgs.msg import JacoAngularVelocityCmd7DOF, JacoCartesianVelocityCmd, PVA, PanTiltCmd, LinearActuatorCmd

import argparse

SEQ = 1000


def pose_publisher(msg, arm="right", rate=1, duration=float('inf')):
    """
    duration (float) the duration (seconds) of time the message will be published.
    """
    global SEQ
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.seq = SEQ; SEQ += 1

    if isinstance(msg, JacoCartesianVelocityCmd):
        pub = rospy.Publisher("movo/%s_arm/cartesian_vel_cmd" % arm, JacoCartesianVelocityCmd, queue_size=10)
    else:
        pub = rospy.Publisher("movo/%s_arm/angular_vel_cmd" % arm, JacoAngularVelocityCmd7DOF, queue_size=10)
    pub.publish(msg)
    rospy.loginfo(msg)
    start = rospy.Time.now()
    d = rospy.Duration(duration) if duration < float('inf') else None
    while (duration >= float('inf')\
           or (duration < float('inf') and rospy.Time.now() - start <= d))\
          and not rospy.is_shutdown():
        msg.header.stamp = rospy.Time.now()
        msg.header.seq = SEQ; SEQ += 1
        pub.publish(msg)
        rospy.sleep(0.005)

def angular_vel(indices=[], new_vals=[]):
    msg = JacoAngularVelocityCmd7DOF()
    vals = [0]*7
    if len(indices) > 0 and len(new_vals) > 0 and len(indices) == len(new_vals):
        for indx, i in enumerate(indices):
            vals[i] = new_vals[indx]

    msg.theta_shoulder_pan_joint = vals[0]
    msg.theta_shoulder_lift_joint = vals[1]
    msg.theta_arm_half_joint = vals[2]
    msg.theta_elbow_joint = vals[3]
    msg.theta_wrist_spherical_1_joint = vals[4]
    msg.theta_wrist_spherical_2_joint = vals[5]
    msg.theta_wrist_3_joint = vals[6]
    return msg

def cartesian_vel(indices=[], new_vals=[]):
    # cartesian velocity does not have joint-specific setting. For the entire arm,
    # there are:
    #
    # x, y, z, theta_x, theta_y, theta_z
    vals = [0] * 6
    if len(indices) > 0 and len(new_vals) > 0 and len(indices) == len(new_vals):
        for indx, i in enumerate(indices):
            vals[i] = new_vals[indx]

    msg.x = vals[0]
    msg.y = vals[1]
    msg.z = vals[2]
    msg.theta_x = vals[3]
    msg.theta_y = vals[4]
    msg.theta_z = vals[5]
    return msg


def move_head(pan, tilt):
    """pan and tilt are lists of floats. (potentially empty if no movement intended."""
    global SEQ

    if len(pan) > 0 and len(pan) != 3:
        raise ValueError("pan has three elements (p, v, a)!")
    if len(tilt) > 0 and len(tilt) != 3:
        raise ValueError("tilt has three elements (p, v, a)!")
    # Note that as long as the velocity and acceleration are zero, the pan/tilt won't
    # change from its current position.
    msg = PanTiltCmd()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.seq = SEQ; SEQ += 1

    if len(pan) > 0:
        pva_pan = PVA()
        pva_pan.pos_rad = pan[0]
        pva_pan.vel_rps = pan[1]
        pva_pan.acc_rps2 = pan[2]
        msg.pan_cmd = pva_pan
    if len(tilt) > 0:
        pva_tilt = PVA()
        pva_tilt.pos_rad = tilt[0]
        pva_tilt.vel_rps = tilt[1]
        pva_tilt.acc_rps2 = tilt[2]
        msg.tilt_cmd = pva_tilt

    pub = rospy.Publisher("movo/head/cmd", PanTiltCmd, queue_size=10, latch=True)
    rospy.loginfo("Publishing %s" % msg)
    pub.publish(msg)
    print("Publishing and latching message. Press ctrl-C to terminate")
    while not rospy.is_shutdown():
        rospy.sleep(2)


def move_torso(pos, vel):
    global SEQ

    if pos < 0 or pos > 0.6:
        raise ValueError("Invalid position for torso! (0 ~ 0.6)")
    if vel > 0.1:
        raise ValueError("Too fast!")
    elif vel < 0:
        raise ValueError("Velocity cannot be negative.")

    msg = LinearActuatorCmd()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.seq = SEQ; SEQ += 1
    msg.desired_position_m = pos
    msg.fdfwd_vel_mps = vel
    pub = rospy.Publisher("movo/linear_actuator_cmd", LinearActuatorCmd, queue_size=10, latch=True)
    rospy.loginfo("Publishing %s" % msg)
    pub.publish(msg)
    print("Publishing and latching message. Press ctrl-C to terminate")
    while not rospy.is_shutdown():
        rospy.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description='Publish velocity commands to MOVO arm joints.')
    parser.add_argument("part", type=str, help="Which arm to move. 'left' or 'right' or 'head' or 'torso'")
    parser.add_argument('-i', '--indices', type=int, nargs='+',
                        help='Indices of joints, or indices of coordinates (if using cartesian).')
    parser.add_argument('-v', '--vals', type=float, nargs='+',
                        help='Values to set for the corresponding indices.')
    parser.add_argument('-c', '--cartesian', help='Cartesian velocity commands', action="store_true")
    parser.add_argument('-r', '--rate', type=float, help="Rate for publishing velocity command", default=1.0)
    parser.add_argument('-p', '--pan', type=float, nargs='+',
                        help='a list of 3 floats to control head pan: position (rad), velocity (rad/s)'\
                        'and acceleration (rad/s^2)')
    parser.add_argument('-t', '--tilt', type=float, nargs='+',
                        help='a list of 3 floats to control head tilt: position (rad), velocity (rad/s)'\
                        'and acceleration (rad/s^2)')
    parser.add_argument('-d', '--duration', type=float, help="the duration (seconds) of time the message"\
                        "will be published.", default=float('inf'))
    args = parser.parse_args()

    rospy.init_node("movo_pose_node", anonymous=True)

    if args.part.lower() != "left" and args.part.lower() != "right" and args.part.lower() != "head"\
       and args.part.lower() != "torso":
        raise ValueError("part must be either left or right or head or torso!")

    indices = args.indices if args.indices is not None else []
    vals = args.vals if args.vals is not None else []

    try:
        if args.part.lower() == "left" or args.part.lower() == "right":
            if args.cartesian:
                msg = cartesian_vel(indices, vals)
            else:
                msg = angular_vel(indices, vals)
            pose_publisher(msg, args.part.lower(), rate=args.rate, duration=args.duration)
        elif args.part.lower() == "head":
            if args.pan is None and args.tilt is None:
                raise ValueError("At least one of '--pan' or '--tilt' must be given.")
            move_head(args.pan if args.pan is not None else [],
                      args.tilt if args.tilt is not None else [])
        elif args.part.lower() == "torso":
            if len(args.vals) != 2:
                raise ValueError("Need to supply position and velocity for torso movement.")
            move_torso(args.vals[0], args.vals[1])
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
