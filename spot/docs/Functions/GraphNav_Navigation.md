# GraphNav Navigation

## Command Line Example
Use the command line example
```
python -m graph_nav_command_line --upload-f ilepath ../../../../ros_ws/src/rbd_spot_perception/maps/bosdyn/cit_first_floor/ $SPOT_I
```

You need to first press 5 to upload the graph.
Then, you need to press 2 or 3 to localize the robot.
Pressing 2 is more realistic, as you only need the
robot to be able to see a fiducial marker.

The localization procedure does not require movement of the robot.

There is no visualization; You have to run the `view_map` example
to see the waypoints on top of the map.

You can specify several types of navigation goals:
```
(6) Navigate to. The destination waypoint id is the second argument.
(7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
(8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
               [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
               When a value for z is not specified, we use the current z height.
               When only yaw is specified, the quaternion is constructed from the yaw.
               When yaw is not specified, an identity quaternion is used.
```
Recall that _seed frame_ in GraphNav means some global reference frame ([doc](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_map_structure#:~:text=An%20anchoring%20is%20a%20mapping%20from%20waypoints%20to%20some%20global%20reference%20frame.%20That%20is%2C%20for%20every%20waypoint%20and%20fiducial%2C%20we%20have%20an%20SE3Pose%20describing%20the%20transform%20from%20a%20seed%20frame%20to%20that%20waypoint%20or%20fiducial.))
