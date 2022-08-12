#!/bin/bash
#
# Usage: rosrun rbd_spot_perception graphnav_mapper lab121
#
# This will run the recording GraphNav map example, and the resulting
# map will be saved at lab121.

if [ "$#" -lt 1 ]; then
    echo "Usage: graphnav_mapper <map_name>"
    exit 1
fi

repo_root=$REPO_ROOT
source $repo_root/tools.sh
map_name=$1
python $repo_root/spot/spot-sdk/python/examples/graph_nav_command_line/recording_command_line.py -d $map_name $SPOT_IP

# We assume the user has downloaded the map. But we will check
map_path=$map_name
if [ ! -d $map_path ]; then
    echo "No map seems to be saved."
    exit 1
fi

# Move the map directory to rbd_spot_perception/maps/bosdyn, and reorganize a little.
bosdyn_maps_dir=$repo_root/spot/ros_ws/src/rbd_spot_perception/maps/bosdyn
mv $map_path $bosdyn_maps_dir
mv $bosdyn_maps_dir/$map_name/downloaded_graph/* $bosdyn_maps_dir/$map_name
rm -r $bosdyn_maps_dir/$map_name/downloaded_graph/

# Ask if you'd like to view it; add an alias anyways.
function viewmap {
    python $repo_root/spot/spot-sdk/python/examples/graph_nav_view_map/view_map.py $1
}

if confirm "view map? "; then
    viewmap $bosdyn_maps_dir/$map_name
fi
