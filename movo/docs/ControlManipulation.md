# Manipulation and Control

I have developed a package for doing "hard-coded" manipulation
and control with MOVO. Refer to the [documentation for it here](./src/rbd_movo_action/rbd_movo_motor_skills/README.md).
It is not perfect but currently it offers one possible solution.


## MOVO joint state topics
```
rostopic echo /movo/left_arm/joint_states
```

## Using Moveit! with MOVO

Refer to official tutorial for Moveit! with ROS kinetic [here](https://docs.ros.org/en/kinetic/api/moveit_tutorials/html/doc/getting_started/getting_started.html).

Possible Moveit planners (settings to `planner_id`):

- SBLkConfigDefault
- ESTkConfigDefault
- LBKPIECEkConfigDefault
- BKPIECEkConfigDefault
- KPIECEkConfigDefault
- RRTkConfigDefault
- RRTConnectkConfigDefault
- RRTstarkConfigDefault
- TRRTkConfigDefault
- PRMkConfigDefault
- PRMstarkConfigDefault


A very nice overview of how Moveit! works ([source](bhttps://moveit.ros.org/documentation/concepts/))

<img src="https://moveit.ros.org/assets/images/diagrams/moveit_pipeline.png" width="500px"/>

## Obtain Moveit! planning feedback

You can get feedback of current plan status by subscribing to
either `/move_group/feedback` or `/move_group/status`. You may get a
message like
```
header:
  seq: 54418
  stamp:
    secs: 1643057072
    nsecs: 696490146
  frame_id: ''
status_list:
  -
    goal_id:
      stamp:
        secs: 1643056997
        nsecs:  12346982
      id: "/ArmPose_Exe-1-1643056997.012"
    status: 3
    text: "Solution was found and executed."
```
This message is of atype [actionlib_msgs/GoalStatusArray](http://docs.ros.org/en/melodic/api/actionlib_msgs/html/msg/GoalStatusArray.html).
You can see from [GoalStatus](http://docs.ros.org/en/melodic/api/actionlib_msgs/html/msg/GoalStatus.html)
message type that status 3 means "SUCCEEDED" (`GoalStatus.SUCCEEDED`), among other status values.

## OctoMap collision avoidance

Long story short:

1. Configure the  `point_cloud_topic` in "movo_7dof_moveit_config/config/sensors.yaml" to be where you want Moveit! to get point clouds. Doesn't have
   to be the original (e.g. `/kinect/sd/points`), which can be really noisy. More often, you set it to your custom topic and you would publish filtered points to that topic. Note that if Moveit! doesn't receive any point cloud from this topic, the OctoMap layer will be non-existent.
   
   Besides changing the topic, the other parameters could be left as is.
   
2. Make sure this parameter file "sensors.yaml" is loaded when MOVO bring up system launch starts. By default it should be loaded. 

3. If the setup is correct, you should be able to receive message if you do
     ```
     rostopic echo /move_group/filtered_cloud
     ```
     You can check if your configuration is correct by
     ```
     rosparam get /move_group/sensors
     - {filtered_cloud_topic: filtered_cloud, max_range: 2.0, padding_offset: 0.05, padding_scale: 1.0,
       point_cloud_topic: /kinect2/sd/points, point_subsample: 1, sensor_plugin: occupancy_map_monitor/PointCloudOctomapUpdater}
     ```

     If you add a PointCloud2 visualization in RVIZ, and set the topic to `movo_group/filtered_cloud`,
     you will be able to see the visualized OctoMap for the obstacles, which the motion planner
     should supposedly know how to avoid.

     ![image](https://user-images.githubusercontent.com/7720184/150757773-1a0c8f7d-89eb-426f-9ca9-0fb1150b9d28.png)

#### Filtering/Clearing up Point Cloud for OctoMap
I encountered a problem due to noisy kinect that there are some
points really close to the robot, which causes the motion
planning to fail. I asked a [question on ROS Answers](https://answers.ros.org/question/395059/noisy-points-from-point-cloud-causes-moveit-to-fail/).

The solution is to publish filtered point cloud and use
that as the point cloud topic for OctoMap. Basically,
run
```
roslaunch rbd_movo_perception process_pointcloud.launch
```
and then set the `point_cloud_topic` in `sensors.yaml` (for movo_7dof_moveit_configs)
to be `kinect2/sd/filtered_points`.
**THIS IS IN FACT WHAT YOU SHOULD DO ON THE REAL MOVO.**

With this, you can leave the OctoMap layer running.
If you can see points coming through `move_group/filtered_cloud` in RVIZ, then
motion planning should now take the point cloud from depth camera into account.

### Temporarily Turn Off OctoMap Collision Checking
If you have launched `process_pointcloud.launch` and configured `sensors.yaml`
as described above, then move_group should take point cloud from
your custom topic `kinect2/sd/filtered_points`. As long as you terminate
this launch file, OctoMap collision checking should be terminated.

To programmatically turn off Octomap collision checking:

1. Configure `kinect2/sd/filtered_points` in `sensors.yaml` to something else, like
   `kinect2/sd/filtered_points_relayed`

2. Restart MOVO bringup system launch so that the new parameters are taken into account.

3. Use a bash script to run `rosrun topic_tools relay kinect2/sd/filtered_points  kinect2/sd/filtered_points_relayed`.

4. Use a Python Subprocess to start and stop that script. This is probably more efficient
   that starting and stopping the `processed_pointcloud.launch` file programmatically.


### How to clear octomap
(1) If you add an appropriate collision object, the octomap will be cleared in the vicinity of the object. You still have to allow the collision with the collision object of course.
`rosservice call /clear_octomap`

(2) If you really want to disable octomap updates, the easiest way to do so is externally by writing a node that forwards point clouds to the topic move_group subscribes to only when required.

[Reference](https://github.com/ros-planning/moveit/issues/1728#issuecomment-553882310).



## Avoiding big weird motions.

TRY FIRST: INCREASE TOLERANCE.

If you just send Moveit! the end effector pose, you may end up something like this:

![image](https://user-images.githubusercontent.com/7720184/150889532-0dfa798d-1a6b-4d68-a23b-1ca3d3df46a7.png)

The end effector goal pose is the same, but the planner comes up with different trajectories, some requiring very dramatic movement of the arm. The motion planner
does not readily make any distinction between the solutions. Furthermore, even though Moveit! outputs a plan and the robot executes the plan, it may
happen that the robot stops keep executing the motion plan. The goal is not reached, but the arm has moved half way. You will get the "ABORTED" status at `move_group/feedback`, and may get `Motion plan was found but it seems to be invalid (possibly due to postprocessing). Not executing.` The left two images of the above figure are in exactly that situation.


Ideally you can set constraints the planner should satisfy in a sequence.
Isn't that equivalent as planning a sequence of smaller subgoals?
In that case, [this ROS Answers post](https://answers.ros.org/question/296994/how-to-set-a-sequence-of-goals-in-moveit/#:~:text=You%20can%20set%20(n)%20targets,on%20the%20sequence%20you%20established.)
and [this tutorial](https://www.theconstructsim.com/ros-qa-138-how-to-set-a-sequence-of-goals-in-moveit-for-a-manipulator/) could be useful.
(In fact these are not useful because I am not using the Moveit Commander thing)

Instead, I noticed [MotionPlanRequest](http://docs.ros.org/en/noetic/api/moveit_msgs/html/msg/MotionPlanRequest.html)
message, which is part of the [MoveGroupGoal](https://github.com/kunal15595/ros/blob/master/moveit/devel/share/moveit_msgs/msg/MoveGroupGoal.msg)
I need to set, contains several useful fields:
```bash
# The possible goal states for the model to plan for. Each element of
# the array defines a goal region. The goal is achieved
# if the constraints for a particular region are satisfied
Constraints[] goal_constraints

# No state at any point along the path in the produced motion plan will
# violate these constraints (this applies to all points, not just waypoints)
Constraints path_constraints

# The constraints the resulting trajectory must satisfy
TrajectoryConstraints trajectory_constraints

# A set of trajectories that may be used as reference or initial trajectories
# for (typically optimization-based) planners
# These trajectories do not override start_state or goal_constraints
GenericTrajectory[] reference_trajectories
```

Here is info I found about these parameters:
says about these parameters:
*  Kinematic constraints for the path given by `path_constraints` will be met for every point along the trajectory, if they are not met, a partial solution will be returned. (Reference: the [documentation (Jade)](http://docs.ros.org/en/jade/api/moveit_ros_planning_interface/html/classmoveit_1_1planning__interface_1_1MoveGroup.html))

* The type [TrajectoryConstraints](http://docs.ros.org/en/noetic/api/moveit_msgs/html/msg/TrajectoryConstraints.html) is simply defined as an array of constraints:
    ```
    # The array of constraints to consider along the trajectory
    Constraints[] constraints
    ```
    **NOTE THIS DOES NOT WORK. SEE BELOW**

* The comments of `goal_constraints` seems to say that it supports defining
  multiple goals (goal regions) and as long as any one of them is satisfied,
  the goal is achieved. That is useful too.

Read [this documentation](https://ros-planning.github.io/moveit_tutorials/doc/planning_with_approximated_constraint_manifolds/planning_with_approximated_constraint_manifolds_tutorial.html)
that explains the constraints interface Moveit! provides.


Side step: This [post](https://answers.ros.org/question/236564/moveit-path-planning-with-constraints-fails/) suggests you should use [track_ik](http://wiki.ros.org/trac_ik) instead of KDL (the default IK solver) with moveit
```
apt-get install ros-kinetic-trac-ik-kinematics-plugin
```
(This is already installed for me on ROS Kinetic).

### More on Trajectory Constraint
From v4dn on [Github](https://github.com/ros-planning/moveit/issues/1707):
>The MotionPlanRequest API allows to set the trajectory_constraints field, but a planner plugin actually has to take them into account.
>The default OMPL interface does not do that.
>Incidentally, I'm not aware of a planner plugin that does that at the moment -
>also because the semantics of these constraints are not very well-defined.
That is very sad! Similar info on [ROS Answers](https://answers.ros.org/question/304342/path_constraints-vs-trajectory_constraints-in-motion_plan_request/)
>...it seems that the function of trajectory_constraints hasn't been clearly defined yet. Hence I >suppose it would be safe to say that you should be using path_constraints until that discussion gets closed.
Refer to [this Github issue](https://github.com/ros-planning/moveit/pull/793)

SO I need to use Path Constraints, or just do waypoints.


### More on Path Constraint
Watch [this Youtube video](https://www.youtube.com/watch?v=qEketOee7_g&feature=emb_title) for a great explanation of what path constraint does. Because trajectory constraint is ignored by planners (as discussed above), if you want to set constraints beside the goal, your bet is path constraint (otherwise you would have to do waypoints).

![image](https://user-images.githubusercontent.com/7720184/151100625-eaafce80-27b5-43c5-ad0a-a3e4d4c6c25e.png)

A tip I figured out is you could first figure out a desirable arm configuration. Then you obtain its end effector pose as the goal. Then you also obtain its
joint positions, which you can use as the path constraint. I added a [-60, 60] degree bound for the big joints (first 4), and I adjust them according to the use case. This has worked well so far. Check out notes on creating skills.

## Caveats
You can control fingers individually with some effort:
Look at [this Github thread](https://github.com/Kinovarobotics/kinova-movo/issues/33).
