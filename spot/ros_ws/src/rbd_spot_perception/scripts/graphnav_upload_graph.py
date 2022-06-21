#!/usr/bin/env python
# Upload graph to GraphNav

import rbd_spot
import argparse

def main():
    parser = argparse.ArgumentParser("stream graphnav pose")
    parser.add_argument("-p", "--path", type=str, help="path to graphnav map directory",
                        required=True)
    args = parser.parse_args()

    conn = rbd_spot.SpotSDKConn(sdk_name="GraphNavPoseStreamer")
    graphnav_client = rbd_spot.graphnav.create_client(conn)

    # Check if there is already a graph. If so, abort
    graph, _ = rbd_spot.graphnav.downloadGraph(graphnav_client)
    if graph is not None:
        print("Robot has graph already. You may want to clear it first.")
        return

    # load the graph
    (current_graph, current_waypoints, current_waypoint_snapshots,
     current_edge_snapshots, current_anchors, current_anchored_world_objects)\
     = rbd_spot.graphnav.load_map(args.path)

    # upload the graph to the robot
    upload_result, _used_time = rbd_spot.graphnav.uploadGraph(graphnav_client, current_graph,
                                                              current_waypoint_snapshots,
                                                              current_edge_snapshots)
    print("UploadGraph (and related) took %.3f" % _used_time)

if __name__ == "__main__":
    main()
