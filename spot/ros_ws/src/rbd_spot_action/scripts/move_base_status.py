#!/usr/bin/env python
# this script subscribes to move_base feedback and status and simply prints the status and message
import rospy
from actionlib_msgs.msg import GoalStatusArray

def callback(m):
    rospy.loginfo("Status: {}; Message: {}".format(m.status_list[0].status, m.status_list[0].text))

def main():
    rospy.init_node("move_base_status")
    sub = rospy.Subscriber("/move_base/status", GoalStatusArray, callback)
    rospy.spin()

if __name__ == "__main__":
    main()
