#!/bin/bash
#
# Runs the graph_nav_command_line.py example with a map
# under rbd_spot_perception/maps
#
# Usage: rosrun rbd_spot_perception graphnav_nav lab121
#
# This will upload the map lab121 to Spot GraphNav server.

if [ "$#" -lt 1 ]; then
    echo "Usage: graphnav_nav <map_name>"
    exit 1
fi
repo_root=$REPO_ROOT
source $repo_root/tools.sh
bosdyn_maps_dir=$repo_root/spot/ros_ws/src/rbd_spot_perception/maps/bosdyn
map_name=$1
python $repo_root/spot/spot-sdk/python/examples/graph_nav_command_line/graph_nav_command_line.py -u $bosdyn_maps_dir/$map_name $SPOT_IP
