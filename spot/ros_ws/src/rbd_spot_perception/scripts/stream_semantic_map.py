#!/usr/bin/env python  
import roslib
import rospy
import math
import tf
import geometry_msgs.msg

import tf

from scipy.signal import savgol_filter

if __name__ == '__main__':
    rospy.init_node('broadcast_semantic_map')

    listener = tf.TransformListener()
    br = tf.TransformBroadcaster()

    fiducial_poses = {}

    rate = rospy.Rate(10.0)
    while not rospy.is_shutdown():
        for fiducial_number in range(9):
            fiducial_frame = "fiducial_52"+str(fiducial_number)
            try:
                (trans,rot) = listener.lookupTransform('/map', fiducial_frame, rospy.Time(0))
                fiducial_poses[fiducial_frame] = [trans,rot]
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        for fiducial_marker_name, fiducial_marker_pose in fiducial_poses.items():
            print(fiducial_marker_name, fiducial_marker_pose)
            br.sendTransform((fiducial_marker_pose[0][0], fiducial_marker_pose[0][1], fiducial_marker_pose[0][2]),
                 (fiducial_marker_pose[1][0],fiducial_marker_pose[1][1],fiducial_marker_pose[1][2],fiducial_marker_pose[1][3]),
                 rospy.Time.now(),
                 "persistant_"+fiducial_marker_name,
                 '/map')
        rate.sleep()
