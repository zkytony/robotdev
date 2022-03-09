# Spot Navigation

To run navigation,

1. Start the spot driver in control mode:

   `roslaunch rbd_spot_robot driver.launch control:=true`

2. Run camera streaming and rtabmap localization (with the correct map; lab121 in this example):

   ```
   rosrun rbd_spot_perception stream_front_camerasets.sh
   MAP_NAME=lab121 roslaunch rbd_spot_perception dual_localization.launch
   ```

3. TODO


4. Run visualization `roslaunch rbd_spot_action view_nav.launch`



#### Problem: if control is given to the spot ros driver, how to localize the robot?
Because to localize the robot you need to drive it to create a loop closure.
If you hijack control from the driver, then the driver may not be able to get
it back and you cannot do navigation.

A solution is to have a keyboard controller that you can use while the spot driver
is running. This can be easily done through the [`wasd.py` example](https://dev.bostondynamics.com/python/examples/wasd/readme)
in Spot SDK. **HOWEVER, I FOUND THIS EXAMPLE TO BE NOT WORKING (03/07/2022); The
robot ESTOPS every time the wasd.py is run.**


So instead, I wrote one, using the spot ros's `/spot/cmd_vel` topic.


## Overall Strategy

Because Spot can handle placement of its feet, achieving navigation can be done in possibly the following ways:

1. Use move_base and pretend the robot is a wheeled robot - although move_base is well-supported, the output velocity commands may not be suitable for Spot (it might be though because technically you just need to control the linear velocities).
2. Use the Autowalk feature (somehow) programmatically
3. Do it yourself - because we have access to the grid map, then given a global pose, it should be straightforward (but not trivial) to implement a global
   path planner, and then code up a program that sends velocity commands intermittedly to make the robot follow the global path. This is hacky and risky.
4. Use some library that supports quadruped navigation? ([towr](http://wiki.ros.org/towr) seemed like a promising choice, but it assumes your robot can't do
   gaits itself and it is not available for ROS noetic).


## Spot SDK GraphNav

Spot SDK provides [the GraphNav service](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_service).

You can record a map "using the GraphNavRecording service".


Spot SDK's GraphNav is intended for autonomous site inspection;
basically, the robot would autonomously navigate from waypoint
to waypoint, designed for the specific site.


## APPENDIX: move_base launch file references:

1. [Section 2.5 "Creating a Launch File for the Navigation Stack" in ROS Navigation tutorial](http://wiki.ros.org/navigation/Tutorials/RobotSetup)

2. [Kinova MOVO's move_base.launch](https://github.com/Kinovarobotics/kinova-movo/blob/master/movo_demos/launch/nav/move_base.launch)

3. [sara_action/sara_move_base's move_base.launch](https://github.com/zkytony/sara_actions/blob/master/sara_move_base/launch/move_base.launch) [private repo]


## APPENDIX: Troubleshooting

### _Stuck at "Requesting the map..."_

When launching the `move_base` node, I saw the first three printed messages:
```
[ WARN] [1646661679.269838083]: global_costmap: Pre-Hydro parameter "static_map" unused since "plugins" is provided
[ INFO] [1646661679.275519048]: global_costmap: Using plugin "static_layer"
[ INFO] [1646661679.297608433]: Requesting the map...
```
and the process hangs.

The fix is to set the costmap parameter `map_topic` to the topic where
"nav_msgs/OccupancyGrid" message is published, a parameter for the [costmap
static map layer](http://wiki.ros.org/costmap_2d/hydro/staticmap) (the
documentation is still hydro but it applies to noetic...); this parameter is
set as:
```
global_costmap:
    ...
    map_topic: /rtabmap/grid_map
```

### _Warning: "Pre-Hydro parameter "static_map" unused since "plugins" is provided"_

I get this warning for both "global costmap" and "local costmap":
```
[ WARN] [1646662114.193319092]: global_costmap: Pre-Hydro parameter "static_map" unused since "plugins" is provided
...
[ WARN] [1646662114.612013356]: local_costmap: Pre-Hydro parameter "static_map" unused since "plugins" is provided
```
According to [this Stackoverflow](https://stackoverflow.com/a/61363290/2893053),
the "static_map" parameter is deprecated.

You can ignore this warning; I find it more explicitly understandable that for
`global_costmap`, the "static_map" parameter is set to `true` and for
`local_costmap`, the "static_map" parameter is set to `false`. Removing
this removes this distinction.


But of course, this parameter **is deprecated.** The actual parameter that
is in effect, as suggested by the warning message, is the "plugins":
```
$ rosparam get /move_base/global_costmap/plugins
- name: static_layer
  type: costmap_2d::StaticLayer
- name: obstacle_layer
  type: costmap_2d::ObstacleLayer
- name: inflation_layer
  type: costmap_2d::InflationLayer
```
It is very clear that static layer is used for global costmap, but not local costmap:
```
$ rosparam get /move_base/local_costmap/plugins
- name: obstacle_layer
  type: costmap_2d::ObstacleLayer
- name: inflation_layer
  type: costmap_2d::InflationLayer
```
You can change this simply by defining your own `plugins` parameter like so ([reference](http://wiki.ros.org/costmap_2d/Tutorials/Configuring%20Layered%20Costmaps)):
```
global_costmap:
    plugins:
        - {name: static_map,       type: "costmap_2d::StaticLayer"}
        ...
```

**So, I removed the 'static_map' parameter.**

I had to manually run `rosparam delete /move_base/{local|global}_costmap/static_map`
so that the next `move_base` launch session is not affected by those parameters.


### _Warning: "Trajectory Rollout planner initialized with param meter_scoring not set. Set it to true to make your settings robust against changes of costmap resolution."_
This warning message:
```
[ WARN] [1646663092.504505520]: Trajectory Rollout planner initialized with param meter_scoring not set. Set it to true to make your settings robust against changes of costmap resolution.
```
appears once when I start the move_base node.
According to [this ROS Answers](https://answers.ros.org/question/188847/hydro-error-in-move_baselaunch/),
it is recommended to set the `TrajectoryPlannerROS/meter_scoring` parameter to either `true` or `false`, and
>It is recommended to use meter scoring (meter_scoring: true) when you expect that the resolution of your map might change.


Note that "TrajectoryPlannerROS" is an object within the [`base_local_planner`](http://wiki.ros.org/base_local_planner#TrajectoryPlannerROS) package.


### _Warning: "global_costmap: Parameter "plugins" not provided, loading pre-Hydro parameters" ..._
If you see warnings like this:
```
[ WARN] [1646680822.376228607]: global_costmap: Parameter "plugins" not provided, loading pre-Hydro parameters
...
[ WARN] [1646680822.478478339]: local_costmap: Parameter "plugins" not provided, loading pre-Hydro parameters
```
Ignore it and restart `move_base`. The parameters `plugins` SHOULD be set if you restart.
If that doesn't happen, then you can define them explicitly in the
costmap parameter files.


### _Warning: Costmap2DROS transform timeout ... Could not get robot pose
I saw this warning once together with the above warning regarding "plugins" not provided.
```
[ WARN] [1646680832.782474276]: Costmap2DROS transform timeout. Current time: 1646680832.7824, global_pose stamp: 1646680832.2648, tolerance: 0.5000
[ WARN] [1646680832.782588799]: Could not get robot pose, cancelling reconfiguration
```
I just restarted `move_base` again and this warning is gone.

But, it does still happen occassionally, from time to time. It does not occur
continuously. This may be normal.


### _move_base receives goal, but does not plan any global path)
When I set a goal in RVIZ, the `move_base/goal` topic gets the goal,
and `move_base/status` shows the following message
```
header:
  seq: 3428
  stamp:
    secs: 1646783604
    nsecs: 739524564
  frame_id: ''
status_list:
  -
    goal_id:
      stamp:
        secs: 1646783596
        nsecs: 300418991
      id: "/move_base-4-1646783596.300418991"
    status: 1
    text: "This goal has been accepted by the simple action server"
```
But I don't see a path on RVIZ.

**FIX:** Did you add visualization for global path? The topic is `/move_base/GlobalPlanner/plan`
