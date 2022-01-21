#!/usr/bin/env python

import sys
import yaml

import tf2_ros
import rospy
import numpy as np

from ar_track_alvar_msgs.msg import AlvarMarkers
from rbd_movo_motor_skills.utils.ros_utils import tf2_frame
#!!! NEED THIS: https://answers.ros.org/question/95791/tf-transformpoint-equivalent-on-tf2/?answer=394789#post-id-394789
# STUPID ROS PROBLEM.
import tf2_geometry_msgs.tf2_geometry_msgs

class TEST:
    def __init__(self):
        self._tfbuffer = tf2_ros.Buffer()
        self._tflistener = tf2_ros.TransformListener(self._tfbuffer)
        self._base_frame = "base_link"
        ar_topic = "/ar_pose_marker"
        rospy.Subscriber(ar_topic, AlvarMarkers, self.callback)

    def callback(self, m):
        """Check whether the pose of the ar tag detection
        satisfies the specification.
        Arg:
           m (AlvarMarekrs): The AR tag detections
        """
        for d in m.markers:
            artag_id = d.id
            # This ar tag detection has acceptable id.
            artag_pose_stamped = d.pose
            header = d.header
            artag_pose_stamped.header = header
            artag_pose_stamped.header.frame_id = tf2_frame(artag_pose_stamped.header.frame_id)
            print(self._tfbuffer.lookup_transform(tf2_frame(header.frame_id), self._base_frame, rospy.Time()))
            print("MY FRAME", artag_pose_stamped.header.frame_id)
            artag_pose_stamped = self._tfbuffer.transform(artag_pose_stamped, self._base_frame)


def test():
    rospy.init_node("HEY")
    TEST()
    rospy.spin()

if __name__ == "__main__":
    test()
