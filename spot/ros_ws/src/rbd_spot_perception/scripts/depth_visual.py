#!/usr/bin/env python

import sys
import rospy
from rbd_spot_perception.depth_visual import DepthVisualPublisher

def main():
    camera = sys.argv[1]
    rospy.init_node(f"depth_visual_publisher_{camera}")
    p = DepthVisualPublisher(camera)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        p.updateTasks()
        rate.sleep()

if __name__ == "__main__":
    main()
