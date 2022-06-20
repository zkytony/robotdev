# rbd_spot_perception

Functionalities are provided mainly through:

- [`rbd_spot_perception.image`](./src/rbd_spot_perception/image.py)


## Publish GraphNav Map as ROS Point Cloud
Run the following launch file
```
roslaunch rbd_spot_perception graphnav_map_publisher.launch
```
Note that you can set `map_path` to be the path to a directory
that is generated after you save the GraphNav map using Spot SDK's
GraphNav mapping command line tool.

In order to visualize the points in RVIZ, you need to
publish a tf transform:
```
rosrun tf static_transform_publisher 1 0 0 0 0 0 1 map graphnav_map 100
```


## Troubleshooting

### RVIZ core dumps with error "The minimum corner of the box must be less than or equal to maximum corner"
**Problem**: I get this when initially testing out the graphnav_map_publisher.
When I start rviz, I get:
```
$ rviz
QStandardPaths: XDG_RUNTIME_DIR not set, defaulting to '/tmp/runtime-kaiyuzh'
[ INFO] [1655685998.217462105]: rviz version 1.14.14
[ INFO] [1655685998.217485986]: compiled against Qt version 5.12.8
[ INFO] [1655685998.217508133]: compiled against OGRE version 1.9.0 (Ghadamon)
[ INFO] [1655685998.229895564]: Forcing OpenGl version 0.
[ INFO] [1655685998.365401470]: Stereo is NOT SUPPORTED
[ INFO] [1655685998.365437109]: OpenGL device: llvmpipe (LLVM 12.0.0, 256 bits)
[ INFO] [1655685998.365449278]: OpenGl version: 3.1 (GLSL 1.4).
rviz: /build/ogre-1.9-kiU5_5/ogre-1.9-1.9.0+dfsg1/OgreMain/include/OgreAxisAlignedBox.h:251: void Ogre::AxisAlignedBox::setExtents(const Ogre::Vector3&, const Ogre::Vector3&): Assertion `(min.x <= max.x && min.y <= max.y && min.z <= max.z) && "The minimum corner of the box must be less than or equal to maximum corner"' failed.
Aborted (core dumped)
```
I found [this useful ROS Answers thread](https://answers.ros.org/question/9961/rviz-window-closes-itself/).
The reason for this, I suspect, is that the numpy array of
the point cloud returned by the Python function `load_map_as_points`
are of double type, while `pcl::PointXYZ` has `float` fields.

I tried defining my own point struct with double fields, but I
get pcl-related compilation errors (they don't accept that type).

Therefore, I tried converting the numpy array to float type.
This fixed it.
