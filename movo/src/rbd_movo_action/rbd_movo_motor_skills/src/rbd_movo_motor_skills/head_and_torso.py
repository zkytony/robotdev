# Note: adapted from code by Yoon.
# code from MOS3D
# /author: Kaiyu Zheng
import argparse
import sys
from copy import copy
import rospy
import actionlib
import math
import random
import numpy as np
from movo.system_defines import TRACTOR_REQUEST
from geometry_msgs.msg import Twist
from movo_msgs.msg import ConfigCmd
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint,
)
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState

class HeadJTAS(object):
    def __init__(self, head_topic="/movo/head_controller/state"):
        self._client = actionlib.SimpleActionClient(
            'movo/head_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction,
        )
        self._head_topic = head_topic
        self._goal = FollowJointTrajectoryGoal()
        self._goal_time_tolerance = rospy.Time(0.1)
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self.total_time = 0.0
        server_up = self._client.wait_for_server(timeout=rospy.Duration(10.0))
        if not server_up:
            rospy.logerr("Timed out waiting for Joint Trajectory"
                         " Action Server to connect. Start the action server"
                         " before running example.")
            rospy.signal_shutdown("Timed out waiting for Action Server")
            sys.exit(1)
        self.clear()

    def add_point(self, positions, time):
        point = JointTrajectoryPoint()
        point.positions = copy(positions)
        point.velocities = [0.0] * len(self._goal.trajectory.joint_names)
        point.time_from_start = rospy.Duration(time)
        self._goal.trajectory.points.append(point)

    def start(self):
        self._goal.trajectory.header.stamp = rospy.Time(0.0)
        self._client.send_goal(self._goal)

    def stop(self):
        self._client.cancel_goal()

    def wait(self, timeout=15.0):
        self._client.wait_for_result(timeout=rospy.Duration(timeout))

    def result(self):
        return self._client.get_result()

    def clear(self):
        self._goal = FollowJointTrajectoryGoal()
        self._goal.goal_time_tolerance = self._goal_time_tolerance
        self._goal.trajectory.joint_names = ['pan_joint','tilt_joint']

    @staticmethod
    def wait_for_head(head_topic="/movo/head_controller/state"):
        msg = rospy.wait_for_message(head_topic, JointTrajectoryControllerState, timeout=15)
        assert msg.joint_names[0] == 'pan_joint', "Joint is not head joints (need pan or tilt)."
        cur_pan = msg.actual.positions[0]
        cur_tilt = msg.actual.positions[1]
        return cur_pan, cur_tilt

    @classmethod
    def move(cls, desired_pan, desired_tilt,
             head_topic="/movo/head_controller/state",
             v=0.3):
        """desired_pan, desired_tilt (radian) are angles of pan and tilt joints of the head"""
        msg = rospy.wait_for_message(head_topic, JointTrajectoryControllerState, timeout=15)
        assert msg.joint_names[0] == 'pan_joint', "Joint is not head joints (need pan or tilt)."
        cur_pan = msg.actual.positions[0]
        cur_tilt = msg.actual.positions[1]

        traj_head = HeadJTAS()
        total_time_head = 0.0
        traj_head.add_point([cur_pan, cur_tilt], 0.0)
        # First pan
        if desired_pan < cur_pan:
            vel = -v
        else:
            vel = v
        dt = abs(abs(desired_pan - cur_pan) / vel)
        total_time_head += dt
        traj_head.add_point([desired_pan, cur_tilt],total_time_head)
        # then tilt
        if desired_tilt < cur_tilt:
            vel = -v
        else:
            vel = v
        dt = abs(abs(desired_tilt - cur_tilt) / vel)
        total_time_head += dt
        traj_head.add_point([desired_pan, desired_tilt],total_time_head)
        traj_head.start()
        traj_head.wait(total_time_head+3.0)
        return True
