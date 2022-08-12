#!/usr/bin/env python
# Upload graph to GraphNav
#
# Example usage:
#
#    python graphnav_publish_waypoints.py -p ../maps/bosdyn/cit_first_floor --viz

import rospy
import rbd_spot
import argparse
from rbd_spot_perception.msg import GraphNavWaypoint, GraphNavWaypointArray
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

from bosdyn.client.frame_helpers import get_odom_tform_body, get_a_tform_b, ODOM_FRAME_NAME
from bosdyn.client.math_helpers import SE3Pose


def waypoint_to_msg(waypoint, snapshot, anchoring, frame_id, viz=False):
    # First, obtain the seed frame pose of the waypoint
    cloud = snapshot.point_cloud
    waypoint_tform_odom = SE3Pose.from_obj(waypoint.waypoint_tform_ko)
    odom_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, ODOM_FRAME_NAME,
                                     cloud.source.frame_name_sensor)
    waypoint_tform_cloud = waypoint_tform_odom * odom_tform_cloud
    seed_tform_cloud = SE3Pose.from_obj(anchoring.seed_tform_waypoint) * waypoint_tform_cloud

    # Now, compose the message
    wpmsg = GraphNavWaypoint()
    wpmsg.header.stamp = rospy.Time.now()
    wpmsg.header.frame_id = frame_id
    wpmsg.id = waypoint.id
    wpmsg.snapshot_id = waypoint.snapshot_id
    wpmsg.pose_sf.position.x = seed_tform_cloud.position.x
    wpmsg.pose_sf.position.y = seed_tform_cloud.position.y
    wpmsg.pose_sf.position.z = seed_tform_cloud.position.z
    wpmsg.pose_sf.orientation.x = seed_tform_cloud.rotation.x
    wpmsg.pose_sf.orientation.y = seed_tform_cloud.rotation.y
    wpmsg.pose_sf.orientation.z = seed_tform_cloud.rotation.z
    wpmsg.pose_sf.orientation.w = seed_tform_cloud.rotation.w
    wpmsg.name = waypoint.annotations.name

    if viz:
        # Also make a marker
        marker = Marker()
        marker.header = wpmsg.header
        marker.id = int(waypoint.annotations.name.split("_")[1]);
        marker.type = Marker.CYLINDER;
        marker.action = Marker.ADD;
        marker.pose = wpmsg.pose_sf
        marker.scale.x = 0.7
        marker.scale.y = 0.7
        marker.scale.z = 0.2
        marker.color.a = 1.0   # Don't forget to set the alpha!
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        return wpmsg, marker
    else:
        return wpmsg


def main():
    parser = argparse.ArgumentParser("publish waypoints of a graphnav map; will latch")
    parser.add_argument("-p", "--path", type=str, help="path to graphnav map directory",
                        required=True)
    parser.add_argument("--topic", type=str, help="topic to publish to",
                        default="/graphnav_waypoints")
    parser.add_argument("--frame-id", type=str, help="frame id of the waypoints;"
                        "i.e. what frame are the waypoint poses with respect to.",
                        default="graphnav_map")
    parser.add_argument("--viz", action="store_true", help="visualize waypoints as rviz markers")
    parser.add_argument("--rate", type=float, help="rate to publish", default=4.0)
    args, _ = parser.parse_known_args()

    rospy.init_node("graphnav_waypoint_publisher")

    # load the graph
    (current_graph, current_waypoints, current_waypoint_snapshots,
     current_edge_snapshots, current_anchors, current_anchored_world_objects)\
     = rbd_spot.graphnav.load_map(args.path)

    pub_waypoints = rospy.Publisher(args.topic, GraphNavWaypointArray, queue_size=10)
    if args.viz:
        pub_markers = rospy.Publisher(args.topic + "_markers", MarkerArray, queue_size=10)

    rate = rospy.Rate(args.rate)
    while not rospy.is_shutdown():
        waypoint_msgs = []
        marker_msgs = []
        for wpid in current_waypoints:
            waypoint = current_waypoints[wpid]
            snapshot = current_waypoint_snapshots[waypoint.snapshot_id]
            anchoring = current_anchors[wpid]
            res = waypoint_to_msg(waypoint, snapshot, anchoring, args.frame_id)
            if args.viz:
                waypoint_msg, marker_msg = waypoint_to_msg(waypoint, snapshot, anchoring, args.frame_id, viz=True)
                marker_msgs.append(marker_msg)
            else:
                waypoint_msg = waypoint_to_msg(waypoint, snapshot, anchoring, args.frame_id)
            waypoint_msgs.append(waypoint_msg)

        waypoint_array_msg = GraphNavWaypointArray()
        waypoint_array_msg.header = Header()

        waypoint_array_msg.header.stamp = rospy.Time.now()
        waypoint_array_msg.header.frame_id = args.frame_id
        waypoint_array_msg.waypoints = waypoint_msgs
        pub_waypoints.publish(waypoint_array_msg)
        print("Published waypoints")
        if args.viz:
            marker_array_msg = MarkerArray(marker_msgs)
            pub_markers.publish(marker_array_msg)
            print("Published markers")
        rate.sleep()


    rospy.spin()

if __name__ == "__main__":
    main()
