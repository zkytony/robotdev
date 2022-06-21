#!/usr/bin/env python
# Streams graphnav pose; assumes a graph has been uploaded.
import argparse
import rospy
import rbd_spot

def main():
    parser = argparse.ArgumentParser("stream graphnav pose")
    parser.add_argument("-p", "--pub", action="store_true", help="publish as ROS messages")
    args = parser.parse_args()

    if args.pub:
        rospy.init_node("stream_graphnav_pose")

    conn = rbd_spot.SpotSDKConn(sdk_name="StreamImageClient")
    graphnav_client = rbd_spot.graphnav.create_client(conn)

    while True:
        state_result, _used_time = rbd_spot.graphnav.getLocalizationState(graphnav_client)
        print("GetLocalizationState took %.3fs" % _used_time)
        body_pose, stamp = rbd_spot.graphnav.get_pose(state_result, frame='seed', stamped=True)
        waypoint_id, _ = rbd_spot.graphnav.get_pose(state_result, frame='waypoint')
        print(f"timestamp: {stamp}")
        print("body pose (seed frame):")
        print(body_pose)
        print(f"waypoint id: {waypoint_id}")
        print("----")


if __name__ == "__main__":
    main()
