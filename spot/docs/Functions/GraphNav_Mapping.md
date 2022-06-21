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
