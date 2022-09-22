#!/usr/bin/env python
#
# Assuming spot's GraphNav localization is running, and transform
# from body to graphnav_map is published, capture the transform
# from "hand" frame to graphnav_map and publish that as PoseStamped;
# This is useful as an observation stream to the SLOOP agent.
import rospy
import tf
import geometry_msgs, std_msgs
from rbd_movo_systems.utils.ros_utils import transform_to_pose_stamped

def main():
    rospy.init_node("movo_stream_camera_pose")

    listener = tf.TransformListener()
    pose_pub = rospy.Publisher("/movo_camera_pose", geometry_msgs.msg.PoseStamped, queue_size=10)

    rate = rospy.Rate(10)
    last_ex = None
    message_printed = False
    while not rospy.is_shutdown():
        try:
            timestamp = rospy.Time(0)
            trans,rot = listener.lookupTransform("map", "kinect2_color_frame", timestamp)
            pose_msg = geometry_msgs.msg.PoseStamped()
            pose_msg.header = std_msgs.msg.Header(stamp=timestamp, frame_id="map")
            pose_msg.pose.position = geometry_msgs.msg.Point(x=trans[0],
                                                             y=trans[1],
                                                             z=trans[2])
            pose_msg.pose.orientation = geometry_msgs.msg.Quaternion(x=rot[0],
                                                                     y=rot[1],
                                                                     z=rot[2],
                                                                     w=rot[3])
            pose_pub.publish(pose_msg)
            if not message_printed:
                rospy.loginfo("publishing camera pose")
                message_printed = True

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as ex:
            rate.sleep()
            if last_ex is None or str(last_ex) != str(ex):
                rospy.logerr(ex)
            last_ex = ex
            continue

if __name__ == "__main__":
    main()
