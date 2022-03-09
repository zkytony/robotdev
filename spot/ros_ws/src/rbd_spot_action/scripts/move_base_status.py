#!/usr/bin/env python
# this script subscribes to move_base feedback and status and simply prints the status and message
import rospy
from actionlib_msgs.msg import GoalStatusArray, GoalStatus

GOAL_REACHED_PRINTED=False
LAST_PRINT_TIME = None
PRINT_FREQ = 5  # hz

def callback(m):
    global GOAL_REACHED_PRINTED
    global LAST_PRINT_TIME
    if len(m.status_list) > 0:
        status = m.status_list[0].status
        if status == GoalStatus.SUCCEEDED and GOAL_REACHED_PRINTED:
            return  # no need to reprint success

        now = rospy.Time.now()
        if LAST_PRINT_TIME is not None and (now - LAST_PRINT_TIME) < rospy.Duration(1./PRINT_FREQ):
            return  # too frequent

        rospy.loginfo("move_base goal Status: {}; Message: {}".format(status, m.status_list[0].text))
        LAST_PRINT_TIME = now
        if m.status_list[0].status == GoalStatus.SUCCEEDED:
            GOAL_REACHED_PRINTED = True
        else:
            GOAL_REACHED_PRINTED = False  # reset

def main():
    rospy.init_node("move_base_status")
    sub = rospy.Subscriber("/move_base/status", GoalStatusArray, callback)
    rospy.spin()

if __name__ == "__main__":
    main()
