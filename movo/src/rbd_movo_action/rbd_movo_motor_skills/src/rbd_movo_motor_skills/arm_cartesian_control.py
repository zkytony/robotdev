# The following is adapted from movo_demos/scripts/cartesian_control.py

import sys
import copy
import rospy
import moveit_commander
import geometry_msgs.msg

"""
This class takes root in the moveit_ Move Group Python Interface. This code is useful because someone can plan trajectories and movements with only a python code. The link [https://ros-planning.github.io/moveit_tutorials/doc/move_group_python_interface/move_group_python_interface_tutorial.html].

It has been adapted for MOVO.
"""

class ArmCartesianControl(object):
    """
    Self made class used to control Movo in a friendly, non complicated way.
    """
    def __init__(self, side="left"):
        super(ArmCartesianControl, self).__init__()
        moveit_commander.roscpp_initialize(sys.argv)
        # We instantiate a robotCommander
        movo = moveit_commander.RobotCommander()
        # We create a scene, in case we want to visualize that scene in Rviz
        scene = moveit_commander.PlanningSceneInterface()
        # We then initialize our MoveGroupCommander, the group we are going to modify
        group_name = "{}_arm".format(side)
        group = moveit_commander.MoveGroupCommander(group_name)
        self.group = group

    def go_cartesian_pose(self, position, orientation):
        #Specifically for the arms. The group should be changed prior. (left of right)
        x, y, z = position
        qx, qy, qz, qw = orientation

        pose_goal = geometry_msgs.msg.Pose()
        pose_goal.orientation.x = qx
        pose_goal.orientation.y = qy
        pose_goal.orientation.z = qz
        pose_goal.orientation.w = qw
        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z
        self.group.set_pose_target(pose_goal)
        plan = self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()
