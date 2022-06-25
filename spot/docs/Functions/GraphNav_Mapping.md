# GraphNav Mapping

Official reference: https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_map_structure

## Definition and Terminology

_GraphNav maps_ consist of **waypoints** and **edges** between the waypoints.

Each _waypoint_ consists of:

   - a reference frame
   - a unique ID
   - annotations
   - sensor data (bundled as a **snapshot**)

Sensor data at a waypoint may include feature clouds, AprilTag detections, imagery, terrain maps, etc.

_Edges_ consist of:
   - a directed edge from one waypoint to another
   - a transform that estimates the relationship in 3D space between the two waypoints.


A _seed frame_ in GraphNav means some global reference frame ([doc](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_map_structure#:~:text=An%20anchoring%20is%20a%20mapping%20from%20waypoints%20to%20some%20global%20reference%20frame.%20That%20is%2C%20for%20every%20waypoint%20and%20fiducial%2C%20we%20have%20an%20SE3Pose%20describing%20the%20transform%20from%20a%20seed%20frame%20to%20that%20waypoint%20or%20fiducial.))


## Create a map


Use the [recording_command_line](https://dev.bostondynamics.com/python/examples/graph_nav_command_line/readme#recording-service-command-line)
example. Options:
```
    Options:
    (0) Clear map.
    (1) Start recording a map.
    (2) Stop recording a map.
    (3) Get the recording service's status.
    (4) Create a default waypoint in the current robot's location.
    (5) Download the map after recording.
    (6) List the waypoint ids and edge ids of the map on the robot.
    (7) Create new edge between existing waypoints using odometry.
    (8) Create new edge from last waypoint to first waypoint using odometry.
    (9) Automatically find and close loops.
    (a) Optimize the map's anchoring.
    (q) Exit.
```

Typical procedure:

1. Run the `recording_command_line` script.

2. Press 1 to "Start recording a map"

3. Use the controller to drive the robot around. Preferred if you drive the robot in loops.

4. Press 2 to "Stop recording a map"

5. Press 9 to "Automatically find and close loops" (IMPORTANT! this is loop closure)

6. Press a to "Optimize the map's anchoring"

7. Press 5 to save the map (It will be saved into a folder called 'downloaded_graph'); DO THIS, otherwise the map is not saved.


## Publish GraphNav Map as ROS Point Cloud
**If you just want to only visualize the map point cloud**, run the following launch file
```
roslaunch rbd_spot_perception graphnav_map_publisher.launch
```
The point cloud will live in the `graphnav_map` frame.

Note that you can set `map_path` to be the path to a directory
that is generated after you save the GraphNav map using Spot SDK's
GraphNav mapping command line tool.

Now, if you open RVIZ and set the "Fixed Frame" in "Global Options"
to be "graphnav_map" (you may have to manually type it if it is not
in the drop-down). Or you can launch `roslaunch rbd_spot_perception view_graphnav_point_cloud.launch`.

<img src='https://user-images.githubusercontent.com/7720184/174810099-b73515a1-e92c-4858-8373-4b7e7191bbba.png' width='650px'/>



**If you also want to localize the robot model**, you need to:

1. Run spot_ros driver: `roslaunch rbd_spot_robot driver.launch`

2. Upload the map and localize the robot by calling corresponding gRPC services (we have scripts for that):
   ```
   roscd rbd_spot_perception/scripts
   ./graphnav_upload_graph.py -p ../maps/bosdyn/cit_first_floor
   ./graphnav_localize.py
   ```
   Note that `graphnav_localize.py` can set localization based on fiducial marker (default) or waypoint.
   It is more convenient to use the fiducial marker method - especially when the robot is
   next to its dock, and the map includes the fiducial marker on the dock.

3. Publish the GraphNav map as point cloud (same as above)
   ```
   roslaunch rbd_spot_perception graphnav_map_publisher.launch
   ```

4. Stream the localized poses as tf transforms between `graphnav_map` and `body`.
   ```
   roscd rbd_spot_perception/scripts
   ./stream_graphnav_pose.py --pub
   ```

5. Visualize
   ```
   roslaunch rbd_spot_perception view_graphnav_point_cloud_with_localization.launch
   ```

   <img src='https://user-images.githubusercontent.com/7720184/174810487-d02578e1-7a91-48cc-a4e5-7f0a173be43b.jpeg' width='650px'/>


Note that if you created the map with LiDAR, then you must have LiDAR installed when performing localization. Otherwise, you get "The map was recorded with using a sensor configuration which is incompatible with the robot (for example, LIDAR configuration)."
