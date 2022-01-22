#!/usr/bin/env python
# /author: Kaiyu Zheng

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import actionlib
import yaml
import math
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from rbd_movo_motor_skills.msg import PlanMoveEEAction, PlanMoveEEGoal, PlanMoveEEResult, PlanMoveEEFeedback, \
    ExecMoveitPlanAction, ExecMoveitPlanGoal, ExecMoveitPlanResult, ExecMoveitPlanFeedback, \
    PlanJointSpaceAction, PlanJointSpaceGoal, PlanJointSpaceResult, PlanJointSpaceFeedback, \
    PlanWaypointsAction, PlanWaypointsGoal, PlanWaypointsResult, PlanWaypointsFeedback,\
    GetStateAction, GetStateGoal, GetStateResult, GetStateFeedback
from rbd_movo_motor_skills.old.common import ActionType
import rbd_movo_motor_skills.old.util as util
from rbd_movo_motor_skills.old.moveit_planner import MoveitPlanner

import argparse


class MoveitClient:

    """SimpleActionClient that feeds goal to the Planner,
    and react accordingly based on the feedback and result."""

    class Status:
        HEALTHY = 0
        FAILING = 1

    def __init__(self, robot_name="movo"):
        self._plan_client = actionlib.SimpleActionClient("moveit_%s_plan" % robot_name,
                                                         PlanMoveEEAction)
        self._js_plan_client = actionlib.SimpleActionClient("moveit_%s_joint_space_plan" % robot_name,
                                                         PlanJointSpaceAction)
        self._wayp_plan_client = actionlib.SimpleActionClient("moveit_%s_wayp_plan" % robot_name,
                                                              PlanWaypointsAction)
        self._get_state_client = actionlib.SimpleActionClient("moveit_%s_get_state" % robot_name,
                                                              GetStateAction)
        self._exec_client = actionlib.SimpleActionClient("moveit_%s_exec" % robot_name,
                                                         ExecMoveitPlanAction)

        util.info("Waiting for moveit planner server...")
        up = self._plan_client.wait_for_server(timeout=rospy.Duration(10))
        if not up:
            rospy.logerr("Timed out waiting for Moveit Planner")
            sys.exit(1)
        up = self._js_plan_client.wait_for_server(timeout=rospy.Duration(10))
        if not up:
            rospy.logerr("Timed out waiting for Moveit Joint Space Planner")
            sys.exit(1)
        up = self._wayp_plan_client.wait_for_server(timeout=rospy.Duration(10))
        if not up:
            rospy.logerr("Timed out waiting for Moveit Waypoints Planner")
            sys.exit(1)
        up = self._exec_client.wait_for_server(timeout=rospy.Duration(10))
        if not up:
            rospy.logerr("Timed out waiting for Moveit Executer")
            sys.exit(1)
        up = self._get_state_client.wait_for_server(timeout=rospy.Duration(10))
        if not up:
            rospy.logerr("Timed out waiting for Moveit GetState server")
            sys.exit(1)
        self._internal_status = MoveitClient.Status.HEALTHY

    def is_healthy(self):
        return self._internal_status == MoveitClient.Status.HEALTHY

    def go_fail(self):
        self._internal_status = MoveitClient.Status.FAILING

    def send_goal(self, group_name, pose, done_cb=None):
        """"`pose` can either be a Pose, a list of coordinates for
        end effector pose or a list of joint values"""
        #### PlanMoveEE
        if isinstance(pose, geometry_msgs.msg.Pose):
            goal = PlanMoveEEGoal()
            goal.group_name = group_name
            goal.pose = pose
            goal.trans_only = False
            util.info("Client sending goal [%s, %s]" % (group_name, pose))
            self._plan_client.send_goal(goal, done_cb=done_cb)
            self._plan_client.wait_for_result(rospy.Duration.from_sec(5.0))

        elif type(pose) == tuple:
            trans_only = True
            pose_target = geometry_msgs.msg.Pose()
            pose_target.position.x = pose[0]
            pose_target.position.y = pose[1]
            pose_target.position.z = pose[2]
            if len(pose) > 3:
                if len(pose) == 6:
                    pose_target.orientation = geometry_msgs.msg.Quaternion(
                        *quaternion_from_euler(
                            math.radians(pose[3]),
                            math.radians(pose[4]),
                            math.radians(pose[5])))
                elif len(pose) == 7:
                    pose_target.orientation.x = pose[3]
                    pose_target.orientation.y = pose[4]
                    pose_target.orientation.z = pose[5]
                    pose_target.orientation.w = pose[6]
                trans_only = False
            print(pose_target)

            goal = PlanMoveEEGoal()
            goal.group_name = group_name
            goal.pose = pose_target
            goal.trans_only = trans_only
            util.info("Client sending goal [%s, %s]" % (group_name, pose))
            self._plan_client.send_goal(goal, done_cb=done_cb)
            self._plan_client.wait_for_result(rospy.Duration.from_sec(5.0))

        #### PlanWaypoints and PlanJointSpace
        elif type(pose) == list:
            if isinstance(pose[0], geometry_msgs.msg.Point) \
               or isinstance(pose[0], geometry_msgs.msg.Pose):
                goal = PlanWaypointsGoal()
                goal.waypoints = pose
                goal.group_name = group_name
                util.info("Client sending waypoints goal [%s]" % (group_name))
                self._wayp_plan_client.send_goal(goal, done_cb=done_cb)
                self._wayp_plan_client.wait_for_result(rospy.Duration.from_sec(5.0))
            else:
                goal = PlanJointSpaceGoal()
                goal.joint_values = pose
                goal.group_name = group_name
                util.info("Client sending goal [%s, %s]" % (group_name, pose))
                self._js_plan_client.send_goal(goal, done_cb=done_cb)
                self._js_plan_client.wait_for_result(rospy.Duration.from_sec(5.0))

        else:
            util.error("pose type not understood. Goal unsent.")


    def execute_plan(self, group_name, wait=True, done_cb=None, exec_args={}):
        util.info("Executing plan for %s" % group_name)
        goal = ExecMoveitPlanGoal()
        goal.wait = wait
        goal.action = ActionType.EXECUTE
        goal.group_name = group_name
        for attr in exec_args:
            setattr(goal, attr, exec_args[attr])
        self._exec_client.send_goal(goal, done_cb=done_cb)
        self._exec_client.wait_for_result(rospy.Duration.from_sec(5.0))

    def cancel_plan(self, group_name, wait=True):
        util.warning("Canceling plan for %s" % group_name)
        goal = ExecMoveitPlanGoal()
        goal.wait = wait
        goal.action = ActionType.CANCEL
        goal.group_name = group_name
        self._exec_client.send_goal(goal)
        self._exec_client.wait_for_result(rospy.Duration.from_sec(5.0))

    def get_state(self, group_name, done_cb, wait_time=10.0):
        goal = GetStateGoal()
        goal.group_name = group_name
        self._get_state_client.send_goal(goal, done_cb=done_cb)
        finished = self._get_state_client.wait_for_result(rospy.Duration.from_sec(wait_time))
        if not finished:
            util.error("Client didn't hear from Server in %s seconds." % str(wait_time))
            self._internal_status = MoveitClient.Status.FAILING

    def send_and_execute_goals(self, group_name, goals, wait=True, exec_args={}):
        """
        Send and execute a sequence of goals, one by one.

        The `goals` can be in any format that the `send_goal` function understands,
        so either a Pose, a list of coordinates for the end effector (waypoints),
        or a list of joint values.

        If `wait` is true, wait until the last goal execution has been completed.
        """
        def executing(status, result):
            self.get_state(group_name, check_completed)

        def goal_sent(status, result):
            if result.status == MoveitPlanner.Status.SUCCESS:
                self.execute_plan(group_name, done_cb=executing, exec_args=exec_args)
            else:
                util.error("Oops. Something went wrong :(")
                self._internal_status = MoveitClient.Status.FAILING

        def check_completed(status, result):
            # The previous goal is completed if it has been cleared
            # by the planner, which means has_goal will be false.
            if not result.has_goal:
                if self._goal_indx >= len(goals):
                    self._all_goals_done = True
                    return
                else:
                    util.info("Sending goal [%d]" % (self._goal_indx), bold=True)
                    self.send_goal(group_name, goals[self._goal_indx], done_cb=goal_sent)
                    self._goal_indx += 1
            else:
                # The goal might still be executing. We wait and recheck.
                # TODO: be sure?
                util.info("Waiting for goal to be executed...")
                rospy.sleep(1)
                self.get_state(group_name, check_completed)

        self._all_goals_done = False

        util.info("Sending goal [0]", bold=True)
        self.send_goal(group_name, goals[0],
                       done_cb=goal_sent)
        self._goal_indx = 1
        try:
            while self.is_healthy() and \
                  ((self._goal_indx < len(goals) and not wait) \
                   or (not self._all_goals_done and wait)):
                rospy.sleep(1)
        except KeyboardInterrupt as ex:
            util.warning("Interrupted send_and_execute_goals...")
            raise ex



    def send_and_execute_joint_space_goals_from_files(self, group_name, paths,
                                                      wait=True):
        """
        Each file in `paths` is a YAML file that each contains
        one joint space goal. If `wait` is true, wait until the
        last goal execution has been completed.
        """
        goals = []
        for goal_file in paths:
            with open(goal_file) as f:
                util.info("Loading goal from %s" % goal_file)
                goal = yaml.load(f)
                if type(goal[0]) == list:
                    raise ValueError("Single goal only!")
                else:
                    goals.append(goal)

        self.send_and_execute_goals(group_name, goals, wait=wait)

# End of MoveitClient


def parse_waypoints_diffs(points):
    """points is a list of relative differences in x, y, z directions."""
    diffs = []
    for p in points:
        dp = geometry_msgs.msg.Point()
        dp.x = p[0]
        dp.y = p[1]
        dp.z = p[2]
        diffs.append(dp)
    return diffs


def main():
    parser = argparse.ArgumentParser(description='Movo Moveit Client. Priority (-g > -f > -e > -k > -F --state)')
    parser.add_argument('group_name', type=str, help="Group name that the client wants to talk to")
    parser.add_argument('-g', '--goal', type=float, nargs='+',
                        help='Plans goal, specified as a list of floats (either means end-effector pose,'\
                        ', or a list of joint values')
    parser.add_argument('-f', '--goal-file', type=str,
                        help='Path to a yaml file that contains a goal, specified as a list of floats x y z w'\
                        ', or a list of joint values (more than 4 elements). If it contains multiple points,'\
                        'then they are interpreted as waypoints.')
    parser.add_argument('-F', '--plan-exec-goal-files', type=str, nargs='+',
                        help="Plan AND executes multiple goals specified by given files.")
    parser.add_argument('-e', '--exe', help='Execute the plan.', action="store_true")
    parser.add_argument('-k', '--cancel', help='Cancel the plan.', action="store_true")
    parser.add_argument('--ee', help="goal is end effector pose", action="store_true")
    parser.add_argument('--state', help='Get robot state (joint values and pose)', action="store_true")
    args = parser.parse_args()

    # regarding disable_signals, see:
    # https://answers.ros.org/question/262560/rospy-isnt-catching-keyboardinterrupt-on-the-second-iteration-of-the-loop/
    rospy.init_node("moveit_movo_client",
                    anonymous=True, disable_signals=True)

    client = MoveitClient()

    if args.goal:
        goal = args.goal
        if args.ee:
            goal = tuple(goal)
        client.send_goal(args.group_name, goal)

    elif args.goal_file:
        with open(args.goal_file) as f:
            goal = yaml.load(f)
        if type(goal[0]) != list:
            # Single goal
            if args.ee:
                goal = tuple(goal)
        else:
            # Multiple goals. We need a list of waypoints
            goal = parse_waypoints_diffs(goal)
        client.send_goal(args.group_name, goal)

    elif args.exe:
        client.execute_plan(args.group_name)

    elif args.cancel:
        client.cancel_plan(args.group_name)

    elif args.plan_exec_goal_files:
        client.send_and_execute_joint_space_goals_from_files(args.group_name,
                                                             args.plan_exec_goal_files)

    elif args.state:
        goal = GetStateGoal()

if __name__ == "__main__":
    main()
