#!/usr/bin/env python
#
# save map (point cloud and grid map) Saving the grid map can be easily done
# through the map_saver utility.  Regarding point cloud, useful resources:
# - https://answers.ros.org/question/255351/how-o-save-a-pointcloud2-data-in-python/
# - https://answers.ros.org/question/321829/color-problems-extracting-rgb-from-pointcloud2/?answer=395676#post-id-395676

import os
import subprocess
import argparse
import pickle
import numpy as np

import rospy
import rospkg
import ros_numpy
import sensor_msgs.point_cloud2 as pc2
import sensor_msgs.msg

import open3d as o3d

POINT_CLOUD_SAVED=False
GRID_MAP_SAVED = False


def save_point_cloud_to_ply(pcl2msg, map_name, maps_dir):
    """Saves the PointCloud2 message as a .ply file"""
    rospy.loginfo("Saving the point cloud to a .ply file")
    pc = ros_numpy.numpify(pcl2msg)
    pc = ros_numpy.point_cloud2.split_rgb_field(pc)
    points = np.zeros((pc.shape[0], 3))
    rgb = np.zeros((pc.shape[0], 3))
    points[:, 0] = pc['x']
    points[:, 1] = pc['y']
    points[:, 2] = pc['z']
    rgb[:, 0] = pc['r']
    rgb[:, 1] = pc['g']
    rgb[:, 2] = pc['b']
    out_pcd = o3d.geometry.PointCloud()
    out_pcd.points = o3d.utility.Vector3dVector(points)
    out_pcd.colors = o3d.utility.Vector3dVector(rgb.astype(float)/255.0)
    o3d.io.write_point_cloud(
        os.path.join(maps_dir, f"{map_name}_point_cloud.ply"), out_pcd)


def point_cloud_callback(m, args):
    global POINT_CLOUD_SAVED
    if not POINT_CLOUD_SAVED:
        pcl_save_mode, map_name, maps_dir = args
        if pcl_save_mode == "raw":
            with open(os.path.join(maps_dir, f"{map_name}.point_cloud.pkl"), "wb") as f:
                pickle.dump(m, f)
        else:
            save_point_cloud_to_ply(m, map_name, maps_dir)
        POINT_CLOUD_SAVED = True


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
                                       callback_args=(args.pcl_save_mode, map_name, maps_dir),
                                       queue_size=10)

    # For grid map, we just call the map_saver node from ROS navigation.
    p = subprocess.Popen(["rosrun", "map_server", "map_saver", "-f",
                          f"{map_name}_grid_map", f"map:={args.grid_map_topic}"],
                         cwd=maps_dir)  # changes the working directory

    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        poll = p.poll()
        if poll is not None:
            GRID_MAP_SAVED = True

        if POINT_CLOUD_SAVED and GRID_MAP_SAVED:
            rospy.loginfo("Map saved. Done.")
            break
        rate.sleep()

if __name__ == "__main__":
    main()
