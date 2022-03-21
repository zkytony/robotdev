# Mapping

We use [rtabmap](http://introlab.github.io/rtabmap/).

## Strategy when using rtabmap
There is a massive launch file at `rtabmap_ros/launch/rtabmap.launch`.
It is too huge and doesn't seem to support multiple cameras.
However, it does have some nice options. For example,
it supports [loading saved map]((https://github.com/introlab/rtabmap_ros/issues/228#issuecomment-376218928)).

Our strategy is to not use that massive launch file because we
do not know what is going on. We write our own.


## Steps

1. Run spot driver
   ```
   roslaunch rbd_spot_robot driver.launch
   ```

   Do not need to launch any camera streaming through this.

2. Stream camera images. Do this by running :
   ```
   rosrun rbd_spot_perception stream_front_camerasets.sh
   ```

   If you would like to do mapping with other cameras,
   develop a `stream_*_camerasets.sh` file similar to
   the front one, and modify the `mapping.launch` file
   accordingly. Currently, by default, `mapping.launch`
   expects only the front two cameras to be engaged.

   Note that the streaming speed (time per response) will
   be printed. If you are connected to RLAB, this speed
   would increase. Observe and wait till it does so
   (typically around or below 0.7s) before you move on.

3. Run our rtabmap launch file:
   ```
   MAP_NAME="<map_name>" roslaunch rbd_spot_perception dual_mapping.launch
   ```
   Specify map name via `map_name`. Note that the map name is set as an environment variable,
   so that it can be accessed by another program. The resulting map
   will be saved at `$(find rbd_spot_perception)/maps/<map_name>_rtabmap.db)`.

   By default, this will reload the map with the same name. If you
   want to overwrite the existing map, you should do:
   ```
   MAP_NAME="<map_name>" roslaunch rbd_spot_perception dual_mapping.launch reload:=false
   ```

   It by default uses the two front cameras; you can
   configure it to use two other cameras. If you
   want to use a different number of cameras, you
   can develop a launch file based on this one.


4. Run visualization: `roslaunch rbd_spot_perception view_maploc.launch`


## Saving the map

When you launch `dual_camera_mapping.launch` it will by default save
the map in rtabmap's format into a file called `<map_name>_rtabmap.db`.
This file may be very big. If you want to save:

1. the 3D point cloud to a common format (that can be loaded by e.g. Open3D)

2. the 2d grid map as a pgm file (with corresponding .yaml file for ROS navigation)

run the following command:
```
rosrun rbd_spot_perception map_saver.py
```
this will save the map under `$(find rbd_spot_perception)/maps`.
Specifically:
* The point cloud will be saved as `<map_name>_point_cloud.ply`
* The grid map will be saved as `<map_name>_grid_map.pgm` and `<map_name>_grid_map.yaml`

You can visualize the `.ply` point cloud file using Open3D via:
```
rosrun rbd_spot_perception visualize_ply.py <map_name>.ply
```

**Troubleshooting:** If you open the visualizer and you only see a few red dots (same
thing in RVIZ if you visualize the `/rtabmap/grid_map` topic and you see abnormal points),
then you should restart everything (driver, image streaming, rtabmap etc.), and try again.

**Troubleshooting:** TODO it seems like setting `overwrite_existing` to false doesn't
prevent existing map to be removed.


#### _Investigations_
Looking at RVIZ, the point cloud comes from the `/rtabmap/mapData` topic.
The grid map comes from the `/rtabmap/grid_map` topic.

How to save the map? According to [RTABMap's author](https://github.com/introlab/rtabmap_ros/issues/215#issuecomment-357742873);
>To save the cloud, I suggest to open the database afterward with rtabmap
>standalone app ($ rtabmap ~/.ros/rtabmap.db), then do "File->Export 3D
>clouds...", you will have many options to export the point cloud (PLY or PCD,
>or even OBJ if meshing is enabled). If you don't care about cloud density,
>/rtabmap/cloud_map topic can be a good choice too, which is a PointCloud2 topic
>that can be easily save to a PCD using pcl_ros.
My experience after reading the above:
- Yes, "MapCloud" in RVIZ and visualizing `/rtabmap/cloud_map` as PointCloud2
  can both display the point cloud for the map.

  But, `/rtabmap/cloud_map` seems to be a normal rostopic; you can get
  the map by saving a message from it.

- "MapCloud" seems to be an RVIZ "feature;" When you turn it on, the map
  gets incrementally downloaded (this process sometimes seems to make Spot to
  have connection problems with the controller). You can press the "Download Map"
  checkmark to make this process start manually.

Strange (03/02/2022): for once, somehow the map got deleted in the middle of running
rtabmap and RVIZ...? (I got network error again with Spot controller!)

Also, look at [this nice github issue reply](https://github.com/introlab/rtabmap_ros/issues/228#issuecomment-376218928)
on more concretely how to save the map and reload it. There is a very useful
launch file `rtabmap.launch`.

## Resolution

According to [this ROS Answers
post](https://answers.ros.org/question/239760/how-to-get-maps-point-cloud-from-rtab_map/?answer=239768#post-id-239768),
"by default the topic `/rtabmap/cloud_map` is voxelized at 5 cm...You can change
this by setting `cloud_voxel_size` to `0.01` for rtabmap node" (to get higher
resolution as much as shown in rtabmapviz). "You may also want to reduce
cloud_decimation to 1 (default 4) if you want even more points."

To change the resolution of the occupancy grid map,
set the `Grid/CellSize` parameter for the `rtabmap` node.
By default, it is `0.05`. Also, see [this Github issue question](https://github.com/introlab/rtabmap_ros/issues/717#issue-1138872131)
for a dump of parameters for rtabmap_ros.


## APPENDIX: rtabmap Installation

You can directly install rtabmap and rtabmap_ros simply by running
```
sudo apt install ros-noetic-rtabmap-ros
```
You can also install from source, although that is quite unnecessary:
```
git clone https://github.com/introlab/rtabmap.git
cd rtabmap/build
cmake ..
make -j4
sudo make install
```
See `shared/install_rtabmap.sh`. It JUST WORKS in the Docker container we have for spot.


## APPENDIX: Which package to use?

 I would like to try out [ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3),
 a very recent SLAM library. The alternative is [rtabmap](https://github.com/introlab/rtabmap)
 and [cartographer](https://google-cartographer.readthedocs.io/en/latest/).

[ORB_SLAM's original paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7219438) has been probably the most popular
SLAM-package paper. ORB_SLAM3, despite released only in 2020, has
gained wild popularity on github (3.4k stars vs. 1.7k for [rtabmap](https://github.com/introlab/rtabmap)).
So, if you use any other package, you will be left wondering,
"what if I use ORB_SLAM?"

ORB_SLAM3's github says: "ORB-SLAM3 is the **first real-time SLAM library** able to
perform Visual, Visual-Inertial and Multi-Map SLAM with monocular, stereo and
RGB-D cameras, using pin-hole and fisheye lens models. In all sensor
configurations, **ORB-SLAM3 is as robust as the best systems available in the**
**literature, and significantly more accurate.**

There is no other choice than ORB_SLAM3.

Cartographer is not considered because it does not provide RGBD SLAM.

**UPDATE 02/23/2022 10:50PM:** ORB_SLAM3 is very difficult to build and the code base is not maintained.
There does not seem to be [active support for ROS](https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/480)
and the authors have basically left the repository hanging.

**I think to build complicated software like this, you need to use docker. Docker is the way to for "one effort and release stress forever".**

The question is: do you want to use docker for ORB_SLAM3 or RTABMap?

THE ANSWER IS: NEITHER. We will use Open3D.

**NOPE. WE ENDED UP USING RTABMAP.**


## APPENDIX: Resources

* [Useful course material for using rtabmap by U of Chicago](http://people.cs.uchicago.edu/~aachien/Teaching/CS234-W17/CourseMaterials/Lab4.pdf)
