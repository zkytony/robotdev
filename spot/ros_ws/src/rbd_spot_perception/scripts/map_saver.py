#!/usr/bin/env python
#
# save map (point cloud and grid map) Saving the grid map can be easily done
# through the map_saver utility.  Regarding point cloud, here is a useful
# thread:
# https://answers.ros.org/question/255351/how-o-save-a-pointcloud2-data-in-python/
# Also useful:
# - https://github.com/Yelloooowww/point2mesh/blob/main/catkin_ws/src/points2mesh/src/savepcd.py

import os
import subprocess
import argparse
import pickle

import rospy
import rospkg
import sensor_msgs.point_cloud2
import sensor_msgs.msg

import open3d as o3d

POINT_CLOUD_SAVED=False
GRID_MAP_SAVED = False


def save_point_cloud_to_ply(pcl2msg, map_name):
    for point in sensor_msgs.point_cloud2.read_points(pcl2msg):
        print(point)


def point_cloud_callback(m, args):
    global POINT_CLOUD_SAVED
    pcl_save_mode, map_name = args
    if pcl_save_mode == "raw":
        with open(f"{map_name}.point_cloud.pkl", "wb") as f:
            pickle.dump(m, f)
    else:
        save_point_cloud_to_ply(m, map_name)


def main():
    global GRID_MAP_SAVED

    parser = argparse.ArgumentParser("save rtabmap")
    parser.add_argument("-p", "--point-cloud-topic", type=str, default="/rtabmap/cloud_map",
                        help="point cloud topic (type: sensor_msgs/PointCloud2)")
    parser.add_argument("-g", "--grid-map-topic", type=str, default="/rtabmap/grid_map",
                        help="grid map topic (type: nav_msgs/GridMap)")
    parser.add_argument("--pcl-save-mode", type=str, default="open3d", choices=['ply', 'raw'],
                        help="How to save the point cloud."\
                        "If 'raw' then save the message directly as a .pkl file;"\
                        "If 'ply', then save the message as a .ply file using Open3D.")
    args = parser.parse_args()

    rospack = rospkg.RosPack()
    maps_dir = os.path.join(rospack.get_path('rbd_spot_perception'), "maps")
    map_name_file = os.path.join(maps_dir, ".map_name")
    with open(map_name_file) as f:
        map_name = f.readline().strip()
        os.environ['MAP_NAME'] = map_name

    rospy.init_node("map_saver_rtabmap")

    # For point cloud, we make our own subscriber
    point_cloud_sub = rospy.Subscriber(args.point_cloud_topic,
                                       sensor_msgs.msg.PointCloud2,
                                       callback=point_cloud_callback,
                                       callback_args=(args.pcl_save_mode, map_name),
                                       queue_size=10)

    # For grid map, we just call the map_saver node from ROS navigation.
    p = subprocess.Popen(["rosrun", "map_server", "map_saver", "-f", map_name, "map:={args.grid_map_topic}"], cwd=maps_dir)

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        poll = p.poll()
        if poll is not None:
            # subprocess is dead - finished
            GRID_MAP_SAVED = True
        if POINT_CLOUD_SAVED and GRID_MAP_SAVED:
            rospy.loginfo("Map saved. Done.")
            break
        rate.sleep()

if __name__ == "__main__":
    main()
