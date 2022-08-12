#!/usr/bin/env python
#
# Script for navigating spot. In reality, you might want
# to use the graphnav.navigateTo function directly instead
# of running this file because you may have the lease.

import argparse
import rbd_spot
import subprocess
from rbd_spot_perception import graphnav_util
import os

DIR_PATH = os.path.dirname(os.path.abspath(__file__))

def main():
    # Assumes the robot is already localized
    parser = argparse.ArgumentParser("graphnav navigate. If you don't supply any argument,"
                                     "will print current localized pose in seed frame")
    parser.add_argument("--pose", type=float, nargs="+",
                        help="pose to navigate to: x y  or  x y yaw  or  x y z qx qy qz qw")
    parser.add_argument("--waypoint", type=str,
                        help="waypoint to navigate to")
    parser.add_argument("--list", action="store_true", help="list all waypoints in the current graph")
    parser.add_argument("--take", action="store_true", help="take lease (forcefully)")
    parser.add_argument("--slow", action="store_true", help="move slowly")
    parser.add_argument("--fast", action="store_true", help="move quickly (overrides --slow)")
    parser.add_argument("--map-name", type=str, help="upload this map, if provided")
    args, _ = parser.parse_known_args()

    if args.map_name:
        bosdyn_maps_dir = os.path.join(DIR_PATH, "../../rbd_spot_perception/maps/bosdyn")
        subprocess.run(["rosrun", "rbd_spot_perception", "graphnav_clear_graph.py"])
        subprocess.run(["rosrun", "rbd_spot_perception", "graphnav_upload_graph.py", "-p",
                        "{}/{}".format(bosdyn_maps_dir, args.map_name)])

    # Localize the robot
    subprocess.run(["rosrun", "rbd_spot_perception", "graphnav_localize.py"])

    needs_lease = args.pose is not None or args.waypoint is not None
    conn = rbd_spot.SpotSDKConn(sdk_name="GraphNavNavClient",
                                acquire_lease=needs_lease,
                                take_lease=args.take)
    graphnav_client = rbd_spot.graphnav.create_client(conn)

    localization, _ = rbd_spot.graphnav.getLocalizationState(graphnav_client)
    body_pose = rbd_spot.graphnav.get_pose(localization, frame='seed')
    print("Starting pose:")
    print(body_pose)

    # Download graph
    graph, _ = rbd_spot.graphnav.downloadGraph(graphnav_client)
    if graph is None:
        print("No graph uploaded to robot.")
        return

    speed = "medium"
    if args.slow:
        speed = "slow"
    if args.fast:
        speed = "fast"

    # Navigate to waypoint
    if args.waypoint:
        waypoint_id = rbd_spot.graphnav.getWaypointId(graphnav_client, args.waypoint, graph=graph)
        rbd_spot.graphnav.navigateTo(conn, graphnav_client, waypoint_id, speed=speed)

    elif args.pose:
        goal = tuple(args.pose)
        rbd_spot.graphnav.navigateTo(conn, graphnav_client, goal, speed=speed)

    elif args.list:
        rbd_spot.graphnav.listGraphWaypoints(graphnav_client)

    else:
        print("Nothing to do.")

    localization, _ = rbd_spot.graphnav.getLocalizationState(graphnav_client)
    body_pose = rbd_spot.graphnav.get_pose(localization, frame='seed')
    print("Ending pose:")
    print(body_pose)



if __name__ == "__main__":
    main()
