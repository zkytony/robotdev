#!/bin/bash
#
# For a given map, loads the map and starts the localization service.
# Publish localization as ROS messages (both tf and PoseStamped).

if [ "$#" -lt 1 ]; then
    echo "Usage: graphnav_locserver <map_name>"
    exit 1
fi

repo_root=$REPO_ROOT
source $repo_root/tools.sh
map_name=$1

# Upload the map and localize the robot
bosdyn_maps_dir=$repo_root/spot/ros_ws/src/rbd_spot_perception/maps/bosdyn
rosrun rbd_spot_perception graphnav_clear_graph.py
rosrun rbd_spot_perception graphnav_upload_graph.py -p $bosdyn_maps_dir/$map_name
rosrun rbd_spot_perception graphnav_localize.py

# publish the robot pose
rosrun rbd_spot_perception stream_graphnav_pose.py --pub-tf --pub-pose
