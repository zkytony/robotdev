#!/usr/bin/env python
#
# Assuming spot's GraphNav localization is running, and transform
# from body to graphnav_map is published, capture the transform
# from "hand" frame to graphnav_map and publish that as PoseStamped;
# This is useful as an observation stream to the SLOOP agent.
import rospy
import tf2_ros
import geometry_msgs, std_msgs
from rbd_spot.utils.ros_utils import transform_to_pose_stamped

def main():
    rospy.init_node("spot_stream_hand_pose")

    tfBuffer = tf2_ros.Buffer()
    listener = tf2_ros.TransformListener(tfBuffer)
    pose_pub = rospy.Publisher("/spot_hand_pose", geometry_msgs.msg.PoseStamped, queue_size=10)

    rate = rospy.Rate(10)
    last_ex = None
    message_printed = False
    while not rospy.is_shutdown():
        try:
            timestamp = rospy.Time(0)
            trans_stamped = tfBuffer.lookup_transform("graphnav_map", "hand", timestamp)
            pose_msg = transform_to_pose_stamped(trans_stamped.transform,
                                                 "graphnav_map",
                                                 stamp=trans_stamped.header.stamp)
            pose_pub.publish(pose_msg)
            if not message_printed:
                rospy.loginfo("publishing hand pose")
                message_printed = True

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rate.sleep()
            if last_ex is None or str(last_ex) != str(ex):
                rospy.logerr(ex)
            last_ex = ex
            continue

if __name__ == "__main__":
    main()
