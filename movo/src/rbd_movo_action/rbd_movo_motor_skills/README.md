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


## (UNUSED) Using the moveit_client and moveit_planner

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
to control the arm to a desired starting pose. I slowly rotate the joints. See notes below
for how to do that. Then I get the joint state through `rostopic echo` and then save that into a file.




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



MOVO arm joint indexing:

![arm-indexing](https://i.imgur.com/De61JOy.jpg)


## Troubleshooting

### AR detector frequency too low (<1hz)
Found a related question:
https://answers.ros.org/question/275598/ar_track_alvar-running-too-slow1hz/

Solution:
Adding `<param name="max_frequency" type="double" value="10" />` to the node
tag in the launch file allowed me to control the publication frequency of
ar_track_alvar. **NO THIS IS NOT USEFUL**

Also, run `ar_track_alvar` directly on the robot! **THIS IS MORE USEFUL. YOU GET 2HZ**


# Appendix



## MOVO joint state topics


## MOVO control topics

Look at [this Github thread](https://github.com/Kinovarobotics/kinova-movo/issues/33).
