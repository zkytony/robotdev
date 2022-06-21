#!/usr/bin/env python
# localize the robot

import argparse
import rbd_spot

def main():
    parser = argparse.ArgumentParser("graphnav localization")
    parser.add_argument("-m", "--method", type=str, help="type of localization method 'fiducial' or 'waypoint'",
                        choices=['fiducial', 'waypoint'], default='fiducial')
    parser.add_argument("-w", "--waypoint", type=str, help="id of waypoint to localize to")
    args = parser.parse_args()

    conn = rbd_spot.SpotSDKConn(sdk_name="StreamImageClient")
    graphnav_client = rbd_spot.graphnav.create_client(conn)
    robot_state_client = rbd_spot.robot_state.create_client(conn)

    if args.method == "fiducial":
        result, _used_time = rbd_spot.graphnav.setLocalizationFiducial(graphnav_client, robot_state_client)
    else:
        graph, _ = rbd_spot.graphnav.downloadGraph(graphnav_client)
        if graph is None:
            print("No graph uploaded to robot.")
            return
        result, _used_time = rbd_spot.graphnav.setLocalizationWaypoint(graphnav_client, robot_state_client,
                                                                       waypoint_id=args.waypoint,
                                                                       graph=graph)
    print("SetLocalization took %.3fs" % _used_time)


if __name__ == "__main__":
    main()
