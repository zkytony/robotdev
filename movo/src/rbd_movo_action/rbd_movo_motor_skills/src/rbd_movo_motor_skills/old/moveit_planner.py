#!/usr/bin/env python
# Uses Moveit! to control MOVO's parts.
# /author: Kaiyu Zheng
#
# Docs:
# http://docs.ros.org/indigo/api/moveit_tutorials/html/doc/pr2_tutorials/planning/scripts/doc/move_group_python_interface_tutorial.html
#
# See also:
# https://github.com/h2r/ros_reality_bridge/blob/holocontrol_movo/scripts/moveit_movo.py

import os, sys
import math
import copy
import numpy as np
import rospy
import moveit_commander
import moveit_msgs.srv
import moveit_msgs.msg
import geometry_msgs.msg
import std_msgs.msg
import shape_msgs.msg
import actionlib
import copy
import argparse
import yaml
import tf
import threading
import subprocess

from rbd_movo_motor_skills.msg import PlanMoveEEAction, PlanMoveEEGoal, PlanMoveEEResult, PlanMoveEEFeedback, \
    ExecMoveitPlanAction, ExecMoveitPlanGoal, ExecMoveitPlanResult, ExecMoveitPlanFeedback, \
    PlanJointSpaceAction, PlanJointSpaceGoal, PlanJointSpaceResult, PlanJointSpaceFeedback,\
    PlanWaypointsAction, PlanWaypointsGoal, PlanWaypointsResult, PlanWaypointsFeedback, \
    GetStateAction, GetStateGoal, GetStateResult, GetStateFeedback

import rbd_movo_motor_skills.old.common as common
import rbd_movo_motor_skills.old.util as util

common.DEBUG_LEVEL = 1

class ListenEEPose(threading.Thread):
    def __init__(self, planner, group_name, tf_listener,
                 base_frame="/odom", ee_frame="/right_ee_link"):
        threading.Thread.__init__(self)
        self._base_frame = base_frame
        self._ee_frame = ee_frame
        self._poses = []
        self._timestamps = []
        self._tf_listener = tf_listener
        self._planner = planner
        self._group_name = group_name

    def get_poses(self):
        return self._poses

    def run(self):
        try:
            rate = rospy.Rate(100)
            while self._planner._current_goal[self._group_name] is not None\
                  and not rospy.is_shutdown():
                trans, rot = self._tf_listener.lookupTransform(self._base_frame,
                                                               self._ee_frame,
                                                               rospy.Time(0))
                self._poses.append(geometry_msgs.msg.Pose(
                    geometry_msgs.msg.Point(*trans),
                    geometry_msgs.msg.Quaternion(*rot)
                ))
                rate.sleep()

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as ex:
            print("ERROR!")
            return


class MoveitPlanner:

    """SimpleActionServer that takes care of planning."""

    class Status:
        ABORTED = 0
        SUCCESS = 1

    class PlanType:
        CARTESIAN = 1
        JOINT_SPACE = 2
        WAYPOINTS = 3

    def __init__(self, group_names, ee_names, visualize_plan=True,
                 robot_name="movo", joint_limits={}):
        """
        joint_limits (dict) map from joint name to tuple (vel, acc) for velocity and acceleration
                            limits of that joint. The joint name should be full name, including "left"
                            and "right" etc.
        """
        # Initializing npode
        util.info("Initializing moveit commander...")
        moveit_commander.roscpp_initialize(sys.argv)

        # interface to the robot, world, and joint group
        self._robot = moveit_commander.RobotCommander()
        self._scene = moveit_commander.PlanningSceneInterface()
        self._joint_groups = {n:moveit_commander.MoveGroupCommander(n)
                              for n in group_names}
        self._ee_frames = {}

        # Set the planner to be used. Reference: https://github.com/ros-planning/moveit/issues/236

        for i, n in enumerate(group_names):
            self._joint_groups[n].set_planner_id("RRTstarkConfigDefault")
            if ee_names[i] != self._joint_groups[n].get_end_effector_link():
                util.warning("Setting end effector for group %s from %s to %s"
                             % (n, self._joint_groups[n].get_end_effector_link(), ee_names[i]))
                self._joint_groups[n].set_end_effector_link(ee_names[i])
            self._ee_frames[n] = ee_names[i]

        # starts an action server
        util.info("Starting moveit_planner_server...")
        self._plan_server = actionlib.SimpleActionServer("moveit_%s_plan" % robot_name,
                                                         PlanMoveEEAction, self.plan, auto_start=False)
        self._js_plan_server = actionlib.SimpleActionServer("moveit_%s_joint_space_plan" % robot_name,
                                                            PlanJointSpaceAction, self.plan_joint_space, auto_start=False)
        self._wayp_plan_server = actionlib.SimpleActionServer("moveit_%s_wayp_plan" % robot_name,
                                                              PlanWaypointsAction, self.plan_waypoints, auto_start=False)
        self._exec_server = actionlib.SimpleActionServer("moveit_%s_exec" % robot_name,
                                                         ExecMoveitPlanAction, self.execute, auto_start=False)
        self._get_state_server = actionlib.SimpleActionServer("moveit_%s_get_state" % robot_name,
                                                              GetStateAction, self.get_state, auto_start=False)

        self._plan_server.start()
        self._js_plan_server.start()
        self._wayp_plan_server.start()
        self._exec_server.start()
        self._get_state_server.start()

        rospy.wait_for_service("compute_fk")
        self._moveit_fk = rospy.ServiceProxy("compute_fk", moveit_msgs.srv.GetPositionFK)

        self._waypoints = {n:[] for n in group_names}
        self._current_plan = {n:None for n in group_names}
        self._current_goal = {n:None for n in group_names}
        self._plan_type = None
        self._tf_listener = tf.TransformListener()

        # Print current joint positions
        for n in self._joint_groups:
            util.info("Joint values for %s" % n)
            util.info("    " + str(self._joint_groups[n].get_current_joint_values()))
            util.info("Current pose for %s" % n)
            util.info("    " + str(self._joint_groups[n].get_current_pose().pose))

        util.info("Setting joint limits", bold=True)
        for joint_name in joint_limits:
            rospy.set_param("robot_description_planning/joint_limits/%s_joint/max_velocity" % joint_name, joint_limits[joint_name][0])
            rospy.set_param("robot_description_planning/joint_limits/%s_joint/max_acceleration" % joint_name, joint_limits[joint_name][1])

        self.print_joint_limits()

        if visualize_plan:
            self._display_trajectory_publisher = rospy.Publisher(
                '/move_group/display_planned_path',
                moveit_msgs.msg.DisplayTrajectory)

    def __del__(self):
        moveit_commander.roscpp_shutdown()

    # def start_pen_tip_tf(self):
    #     # Run pen tf publisher
    #     if self._pen is not None:
    #         util.info("Starting pen tip tf publisher", bold=True)
    #         pen.publish_transform()


    def print_joint_limits(self):
        # TODO: now only prints left arm
        util.info("Joint Limits", bold=True)
        util.info2("left_shoulder_pan")
        print("vel: %f" % rospy.get_param("robot_description_planning/joint_limits/left_shoulder_pan_joint/max_velocity"))
        print("acc: %f" % rospy.get_param("robot_description_planning/joint_limits/left_shoulder_pan_joint/max_acceleration"))

        util.info2("left_shoulder_lift")
        print("vel: %f" % rospy.get_param("robot_description_planning/joint_limits/left_shoulder_lift_joint/max_velocity"))
        print("acc: %f" % rospy.get_param("robot_description_planning/joint_limits/left_shoulder_lift_joint/max_acceleration"))

        util.info2("left_arm_half")
        print("vel: %f" % rospy.get_param("robot_description_planning/joint_limits/left_arm_half_joint/max_velocity"))
        print("acc: %f" % rospy.get_param("robot_description_planning/joint_limits/left_arm_half_joint/max_acceleration"))

        util.info2("left_elbow")
        print("vel: %f" % rospy.get_param("robot_description_planning/joint_limits/left_elbow_joint/max_velocity"))
        print("acc: %f" % rospy.get_param("robot_description_planning/joint_limits/left_elbow_joint/max_acceleration"))

        util.info2("left_wrist_spherical_1")
        print("vel: %f" % rospy.get_param("robot_description_planning/joint_limits/left_wrist_spherical_1_joint/max_velocity"))
        print("acc: %f" % rospy.get_param("robot_description_planning/joint_limits/left_wrist_spherical_1_joint/max_acceleration"))

        util.info2("left_wrist_spherical_2")
        print("vel: %f" % rospy.get_param("robot_description_planning/joint_limits/left_wrist_spherical_2_joint/max_velocity"))
        print("acc: %f" % rospy.get_param("robot_description_planning/joint_limits/left_wrist_spherical_2_joint/max_acceleration"))

        util.info2("left_wrist_3")
        print("vel: %f" % rospy.get_param("robot_description_planning/joint_limits/left_wrist_3_joint/max_velocity"))
        print("acc: %f" % rospy.get_param("robot_description_planning/joint_limits/left_wrist_3_joint/max_acceleration"))


    def compute_fk(self, ee_link, joint_names, joint_positions, base_frame="odom"):
        """Given joint position, figure out the euclidean coordinates by
        forward kinematics. Uses movo_group's compute_fk service."""
        header = std_msgs.msg.Header(0, rospy.Time.now(), base_frame)
        rs = moveit_msgs.msg.RobotState()
        rs.joint_state.name = joint_names
        rs.joint_state.position = joint_positions
        ee_pose = self._moveit_fk(header, [ee_link], rs).pose_stamped[0].pose
        return ee_pose


    def plan(self, goal):
        group_name = goal.group_name
        if self._current_goal[group_name] is not None:
            rospy.logwarn("Previous goal exists. Clear it first.")
            return
        self._plan_type = MoveitPlanner.PlanType.CARTESIAN
        self._current_goal[group_name] = self._joint_groups[group_name].get_current_pose().pose
        self._current_goal[group_name].position = goal.pose.position
        if not goal.trans_only:
            self._current_goal[group_name].orientation = goal.pose.orientation
        util.info("Generating plan for goal [%s to %s]" % (group_name, self._current_goal[group_name]))

        self._joint_groups[group_name].set_pose_target(self._current_goal[group_name])
        self._current_plan[group_name] = self._joint_groups[group_name].plan()
        result = PlanWaypointsResult()
        if len(self._current_plan[group_name].joint_trajectory.points) > 0:
            util.success("A plan has been made. See it in RViz [check Show Trail and Show Collisions]")
            result.status = MoveitPlanner.Status.SUCCESS
            self._plan_server.set_succeeded(result)
        else:
            util.error("No plan found.")
            result.status = MoveitPlanner.Status.ABORTED
            self._plan_server.set_aborted(result)


    def plan_joint_space(self, goal):
        util.info2("plan_joint_space")
        group_name = goal.group_name
        if self._current_goal[group_name] is not None:
            rospy.logwarn("Previous goal exists. Clear it first.")
            return
        self._plan_type = MoveitPlanner.PlanType.JOINT_SPACE
        self._current_goal[group_name] = goal
        util.info("Generating joint space plan [%s to %s]" % (group_name, goal.joint_values))

        self._joint_groups[group_name].set_joint_value_target(goal.joint_values)
        self._joint_groups[group_name].set_planning_time(1.0)
        self._current_plan[group_name] = self._joint_groups[group_name].plan()
        self._joint_groups[group_name].set_planning_time(60.0)  # set back to default value
        result = PlanJointSpaceResult()
        if len(self._current_plan[group_name].joint_trajectory.points) > 0:
            util.success("A plan has been made. See it in RViz [check Show Trail and Show Collisions]")
            result.status = MoveitPlanner.Status.SUCCESS
            self._js_plan_server.set_succeeded(result)
        else:
            util.error("No plan found.")
            result.status = MoveitPlanner.Status.ABORTED
            self._js_plan_server.set_aborted(result)


    def plan_waypoints(self, goal):
        util.info2("plan_waypoints")
        group_name = goal.group_name
        if self._current_goal[group_name] is not None:
            rospy.logwarn("Previous goal exists. Clear it first.")
            return
        self._plan_type = MoveitPlanner.PlanType.WAYPOINTS
        self._current_goal[group_name] = goal
        util.info("Generating waypoints plan for %s" % (group_name))

        current_pose = self._joint_groups[group_name].get_current_pose().pose
        waypoints = goal.waypoints #[current_pose] +
        self._waypoints[group_name] = waypoints
        self._current_plan[group_name], fraction = self._joint_groups[group_name].compute_cartesian_path(waypoints, 0.01, 0.0)
        result = PlanWaypointsResult()

        if len(self._current_plan[group_name].joint_trajectory.points) > 0:
            util.success("A plan has been made (%d points). See it in RViz [check Show Trail and Show Collisions]"
                         % len(self._current_plan[group_name].joint_trajectory.points))
            result.status = MoveitPlanner.Status.SUCCESS
            self._wayp_plan_server.set_succeeded(result)
        else:
            util.error("No plan found.")
            result.status = MoveitPlanner.Status.ABORTED
            self._wayp_plan_server.set_aborted(result)


    def execute(self, goal):
        util.info2("execute")
        group_name = goal.group_name
        util.info("Received executive action from client [type = %d]" % goal.action)

        result = ExecMoveitPlanResult()
        if goal.action == common.ActionType.EXECUTE:
            if self._plan_type == MoveitPlanner.PlanType.WAYPOINTS:
                base_frame = self._joint_groups[group_name].get_pose_reference_frame()
                eepl = ListenEEPose(self, group_name, self._tf_listener,
                                    base_frame, ee_frame=self._ee_frames[group_name])
                eepl.start()
                success = self._joint_groups[group_name].execute(self._current_plan[group_name])
            else:
                success = self._joint_groups[group_name].go(wait=goal.wait)

            # Print status
            if success:
                util.success("Plan for %s will execute." % group_name)
                result.status = MoveitPlanner.Status.SUCCESS
                rospy.sleep(1)
            else:
                util.error("Plan for %s will NOT execute. Is there a collision?" % group_name)
                result.status = MoveitPlanner.Status.ABORTED

            self.cancel_goal(group_name)  # get rid of this goal since we have completed it
            if self._plan_type == MoveitPlanner.PlanType.WAYPOINTS\
                   and goal.stroke_index >= 0:
                try:
                    util.info("Saving stroke path")
                    char_dir = rospy.get_param("current_writing_character_save_dir")
                    eepl.join()
                    if os.path.exists(char_dir):
                        print(len(self._waypoints[group_name]))
                        if len(self._waypoints[group_name]) > 1:
                            euc_poses = eepl.get_poses()
                            with open(os.path.join(char_dir, "stroke-%d-path.yml" % goal.stroke_index), 'w') as f:
                                yaml.dump(euc_poses, f)
                                util.success("Saved planned trajectory for stroke %d" % goal.stroke_index)
                except KeyError:
                    pass

            util.info("Now %s is at pose:\n%s" % (group_name,
                                                  self._joint_groups[group_name].get_current_pose().pose))
            self._exec_server.set_succeeded(result)

        elif goal.action == common.ActionType.CANCEL:
            self.cancel_goal(group_name)
            result.status = MoveitPlanner.Status.SUCCESS
            self._exec_server.set_succeeded(result)

        else:
            util.error("Unrecognized action type %d" % goal.action)
            result.status = MoveitPlanner.Status.ABORTED
            self._exec_server.set_aborted()

    def get_state(self, goal):
        util.info2("get_state")
        group_name = goal.group_name
        result = GetStateResult()
        # Fill in state attributes
        result.pose = self._joint_groups[group_name].get_current_pose().pose
        result.joint_values = self._joint_groups[group_name].get_current_joint_values()
        result.has_goal = self._current_goal[group_name] is not None
        result.has_plan = self._current_plan[group_name] is not None
        self._get_state_server.set_succeeded(result)

    def cancel_goal(self, group_name):
        self._joint_groups[group_name].clear_pose_targets()
        util.success("Goal for %s has been cleared" % group_name)
        self._current_plan[group_name] = None
        self._current_goal[group_name] = None


def main():
    parser = argparse.ArgumentParser(description='Movo Moveit Planner.')
    parser.add_argument("-j", "--joint_limits_file",
                        type=str, help="Directory to save the collected data", default="../../cfg/arm_joint_limits.yaml")
    parser.add_argument("-e", "--ee-frames", type=str, nargs="+", help="End effector frames for each move_group")
    parser.add_argument('group_names', type=str, nargs="+", help="Group name(s) that the client wants to talk to")
    args = parser.parse_args()

    rospy.init_node("moveit_movo_planner",
                    anonymous=True, disable_signals=True)

    if args.ee_frames is None:
        raise ValueError("ee-frames is required!")

    joint_limits = {}
    with open(args.joint_limits_file) as f:
        joint_limits = yaml.load(f)

    MoveitPlanner(args.group_names, args.ee_frames, joint_limits=joint_limits)
    rospy.spin()

if __name__ == "__main__":
    main()
