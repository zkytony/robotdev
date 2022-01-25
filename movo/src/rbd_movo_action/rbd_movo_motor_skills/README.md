# movo_motor_skill

## Creating this package

```
catkin_create_pkg movo_motor_skills ar_track_alvar std_msgs rospy roscpp pcl_ros pcl_msgs moveitg
catkin_make -DCATKIN_WHITELIST_PACKAGES="movo_motor_skills"
```

## Build
Go to the root of the workspace, i.e. `robotdev/movo`
```
catkin_make -DCATKIN_WHITELIST_PACKAGES="rbd_movo_motor_skills"
```

## Using movo_pose_publisher:

1. Move head. Example:

   ```
   ./movo_pose_publisher.py head -p 0.5 0.1 0.1 -t 0.3 0.1 0.1
   ```

   The values `0.5 0.1 0.1` are the goal angle (radians),
   velocity, and acceleration of the pan.

   The values `0.3 0.1 0.1` are the goal angle (radians),
   velocity, and acceleration of the tilt.

2. Move torso. Example:

    ```
    ./movo_pose_publisher.py torso -v 0.5 0.05
    ```

     The values `0.5 0.05` are the height and velocity
     of the torso.

    `0.5` is pretty tall (up 0.5m from initial torso height).
     The valid numbers are within `0 ~ 0.6`

    `0.05` is pretty fast. The valid numbers are within `0 ~ 0.1`.
     You cannot exceed 0.1.

3. Move arm.

    To rotate the left wrist by 10 degrees per second
      ```
      ./movo_pose_publisher.py left -i 6 -v 10
      ```
      Here, `-i` specifies the index of the joint we want to control.
      Here, 6 refers to the wrist joint.
      The indices are drawn below.
      `-v` specifies the angular velocity of the joint. The positive
      direction is also shown in the drawing below. (Previously I
      was setting this to 0.1 that's why the joint didn't move!!)

      ![arm-indexing](https://i.imgur.com/De61JOy.jpg)

      You can pass in a list of indicies, each with a
      corresponding angular velocity of movement.

      You can also specify a duration (seconds) for how long
      you want to move the joint with the specified velocity.
      For example
      ```
      ./movo_pose_publisher.py left -i 6 -v 10 -d 1

      ```
      will run the velocity command for 1 second, and
      ```
      ./movo_pose_publisher.py left -i 6 -v 10 -d 0.5
      ```
      will do it for half a second.

    -------------------------------

    **Some Investigation to figure out the above command:**
    This is slightly trickier. The way `movo_pose_publisher` deals
    with this is to publish messages to the `/movo/left_arm/angular_vel_cmd` topic.

    According to [this merge request](https://github.com/Kinovarobotics/kinova-movo/pull/24#issue-307543835)
    where the features of controlling by angular and cartesian velocities are added:

    >"If messages are published to one of these topics >1hz, the appropriate
    >SIArmController class will switch from it's current control mode to angular
    >velocity control mode, bypassing the software PID control loop."

    So, if you want velocity control, you need to publish to those topics at >1hz.
    The example command they give worked:
      ```
      rostopic pub /movo/left_arm/angular_vel_cmd movo_msgs/JacoAngularVelocityCmd6DOF '{header: auto, theta_wrist_3_joint: 45.0}' --rate=100
      ```
     This actually publishes at 100Hz. I experimented and indeed, you want >1Hz.
     Otherwise, you are doing **position control** (the joint just moves
     into a position), as indicated above; that may suit you too.

     Another thing is the joint angles are specified in DEGREES not radians.
     Once I realized this, I found that my `movo_pose_publisher.py` script worked!




## Using the moveit_client and moveit_planner

Run the planner
```
./moveit_planner.py [group_names ...] -e [ee_frames ...]
```
A `group_name` is a name of a planning group in Moveit. You can see the list
of planning groups in RVIZ (via MotionPlanning->Planning Request->Planning Group dropdown).
Some groups of interest include: left_arm, left_gripper, right_arm, right_gripper, torso, head.

Each planning group has a corresponding end-effector frame, that is, the frame which you
can specify a world-space coordinate and the IK solver will find the joint space coordinates
for the arm's joints. For example, 'left_arm' planning group has an end-effector frame 'left_ee_link'
You can view all the frames by running `rosrun rqt_tf_tree rqt_tf_tree`.

A valid command to run is:
```
./moveit_planner.py left_arm -e left_ee_link
```
You should see a bunch of messages with no error,
followed by some (likely blue) highlighted texts "Starting moveit_planner_server"...


Now, you can run the client to send motion planning requests.  There are several
ways to move the arm. First, you can set a pose target for the end effector,
with quaternion or not:
```
./moveit_client.py right_arm -g x y z --ee
./moveit_client.py right_arm -g x y z qx qy qz qw --ee
```
If the `--ee` is not supplied, the list of coordinates will be interpreted as joint space goal pose.

Second, you can supply a file of YAML format with a list of pose targets
(waypoints). The YAML file can also be just a list of 7 target joint positions
(joint space goal).
```
./moveit_client.py left_arm -f <path_to_file>
```
For example
```
./moveit_client.py left_arm -f ../../cfg/left_clearaway.yml
```
Now, the planner server should show a message
```
A plan has been made. See it in RViz [check Show Trail and Show Collisions]
```
indicating the plan has been made. You can actually visualize the trajectory
in RVIZ by MotionPlanning -> Planned Path -> check Show Trail. There are other visualization toggles you can play with.

Caveats:
1. The `moveit_planner` plans for only one goal at a time. You must manually
   clear the previous goal by running `./moveit_client.py left_arm -k`
2. If you want to **execute** a motion plan, you need to do
   `./moveit_client.py left_arm -e`. **Works but there is a very long delay.**

You can create custom joint target pose file. First, echo the
joint state topic to observe the format. For example, if you want
to move the left arm,
```
$ rostopic echo /movo/left_arm/joint_states

...
---
header:
  seq: 644954
  stamp:
    secs: 1642818280
    nsecs: 736462116
  frame_id: ''
name: [left_shoulder_pan_joint, left_shoulder_lift_joint, left_arm_half_joint, left_elbow_joint,
  left_wrist_spherical_1_joint, left_wrist_spherical_2_joint, left_wrist_3_joint]
position: [1.577010813198596, 1.428039174457853, 1.096650854742716, 2.6154777233935875, 0.010529069524547907, -0.511974077440103, 1.660100640260742]
velocity: [-0.0003408846195301425, -0.0003408846195301425, -0.0003408846195301425, -0.0003408846195301425, -0.0004958321561296851, 0.0, -0.0004958321561296851]
effort: [-3.3321266174316406, -2.671680450439453, -0.2890584468841553, 2.8224406242370605, -0.032693732529878616, 0.00663731200620532, -0.7412301301956177]
---
...
```
Notice the position array: `[1.577010813198596, 1.428039174457853, 1.096650854742716, 2.6154777233935875, 0.010529069524547907, -0.511974077440103, 1.660100640260742]`.
Each element corresponds to the position of a joint, with name in the `name` field.
What you want to do is to move the arm to a desired position, and then record this position array.
Then, save it in a `yaml`  file under the `cfg` directory.

Previously when I was working on robot writing, I was using the `movo_pose_publisher` script
to control the arm to a desired starting pose. I slowly rotate the joints. See notes above
about movo_pose_publisher for how to do that. Then I get the joint state through `rostopic echo` and then save that into a file.

## Quickly Get Arm Joint State
1. Press the big red E-Stop button
2. Cradle the robot arm to prevent it from falling and hitting obstacles.
3. Move the arm freely as you wish.
4. Reengage the big red E-Stop button. Now, the movo_bringup system is still running, but none of the joint state
   publisher is working anymore. So, do:
     ```
     roslaunch rbd_movo_motor_skills basic_state.launch
     ```
     This will start the (arguably minimal) nodes necessary
     to publish real-time MOVO joint states.

Caveat:
- To prevent the gripper from automatically opening
  upon running
     ```
     roslaunch rbd_movo_motor_skills basic_state.launch
     ```
  I looked into Kinova's codebase. The reason
  the gripper opens is due to the following line in
  `MovoArmJTAS.__init__` under `movo_ros/src/movo_jtas` that says:
    ```python
    self._ctl.api.InitFingers()
    ```
    Here, `ctl` is a `SIArmController` (class in `movo_joint_interface/jaco_joint_controller.py`)
    which has a `api` field that is created by:

    ```python
    self.api = KinovaAPI('left',self.iface,jaco_ip,'255.255.255.0',24000,24024,44000, self.arm_dof)
    ```

   This `KinovaAPI` class (seems very important) under
    `movo_joint_interface/kinova_api_wrapper.py`
    has the following line:
     ```python
      self.InitFingers = self.kinova.Ethernet_InitFingers
     ```
    and `self.kinova` is **proprietory**:
     ```python
     self.kinova=CDLL('Kinova.API.EthCommandLayerUbuntu.so')
     ```
     (interesting; CDLL is a function in [ctypes](https://docs.python.org/3.8/library/ctypes.html),
         a library that allows python to call functions in C?!)
     There isn't anything you can do about the `InitFingers` function. **COMMENTING OUT THE InitFingers() Line breaks the movo bringup system launch.**

## Get Arm End Effector Pose
You want to get the end effector pose in the base_link frame:
So the base_link is the parent/source, left_ee_link is the child/target.
```
rosrun tf tf_echo base_link left_ee_link
```
Note that `tf_echo <source_frame> <target_frame>`


## Obtain AR tag transform:
Each ar tag gets its own tf frame. Do:
```
rosrun tf tf_echo /kinect2_color_optical_frame ar_marker_4
```
This will get you the pose that matches the output of
```
rostopic echo /ar_pose_marker
```

## Troubleshooting

### AR detector frequency too low (<1hz)
Found a related question:
https://answers.ros.org/question/275598/ar_track_alvar-running-too-slow1hz/

Solution:
Adding `<param name="max_frequency" type="double" value="10" />` to the node
tag in the launch file allowed me to control the publication frequency of
ar_track_alvar. **NO THIS IS NOT USEFUL**

Also, run `ar_track_alvar` directly on the robot! **THIS IS MORE USEFUL. YOU GET 2HZ**


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

### Obtain feedback

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

### OctoMap collision avoidance

1. First, edit "movo_7dof_moveit_config/launch/sensor_manager.launch.xml" and filling
   the paramter "octomap_frame" with "base_link" (which is the parent frame for motion planning); Set to base_link when there is no navigation and motion is with respect to the robot's base frame.
   ```xml
   <param name="octomap_frame" type="string" value="base_link" />
   ```

   **Note (01/25/22) that by default, the MOVO bringup system launch will starts the `move_group.launch`.**
   You can find this by checking `movo2.launch` --(starts)-> `movo_bringup/manipulation/movo_moveit.launch` --(starts)-> `movo_moveit_planning_execution.launch` which starts `move_group.launch` under
   the `movo_7dof_moveit_config` package. The reason by default movo bringup doesn;t
   load the sensors is because of the following line in `move_group.launch`
   ```
   <rosparam command="delete" param="move_group/sensors" />
   ```
   I commented it out. I am not sure why Kinova has it there.


2. Then, you need to launch another launch file to have this. That launch file should contain:
     ```
       <include file="$(find movo_7dof_moveit_config)/launch/move_group.launch">
         <rosparam command="load" file="$(find movo_7dof_moveit_config)/config/sensors.yaml" />
       </include>
     ```
     You can do `roslaunch rbd_movo_motor_skills manipulation_system.launch` to launch this.
     It is preferred to do this on the robot to reduce the need to transmit depth data off the robot.

     Note that make sure under `sensors.yaml`, you set the correct topic for point cloud:
     ```yaml
     sensors:
       - sensor_plugin: occupancy_map_monitor/PointCloudOctomapUpdater
         point_cloud_topic: /kinect2/sd/points
         max_range: 2.0
         point_subsample: 1
         padding_offset: 0.05
         padding_scale: 1.0
         filtered_cloud_topic: filtered_cloud
     ```
     Basically Moveit! has made it really convenient to use Octomap for dynamic collision avoidance.
     How practical!
     Also, make sure you are editing the right `sensors.yaml` file. You might be
     editing the one under `movo_moveit_config`, which is for the 6DOF arm.

     **Note that after the fix (01/25/22), the above should not be necessary;**
     **The octomap stuff and sensor configs should be loaded by default**

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

I encountered a problem due to noisy kinect that there are some
points really close to the robot, which causes the motion
planning to fail. I asked a [question on ROS Answers](https://answers.ros.org/question/395059/noisy-points-from-point-cloud-causes-moveit-to-fail/).
**I think the only solution is to publish your own, filtered point cloud for Moveit! that cleans such things up.**

#### How to clear octomap
(1) If you add an appropriate collision object, the octomap will be cleared in the vicinity of the object. You still have to allow the collision with the collision object of course.
`rosservice call /clear_octomap`

(2) If you really want to disable octomap updates, the easiest way to do so is externally by writing a node that forwards point clouds to the topic move_group subscribes to only when required.

[Reference](https://github.com/ros-planning/moveit/issues/1728#issuecomment-553882310).


### Avoiding big weird motions.

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
