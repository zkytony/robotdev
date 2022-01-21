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
./moveit_client right_arm -g x y z --ee
./moveit_client right_arm -g x y z qx qy qz qw --ee
```
If the `--ee` is not supplied, the list of coordinates will be interpreted as joint space goal pose.

Second, you can supply a file of YAML format with a list of pose targets
(waypoints). The YAML file can also be just a list of 7 target joint positions
(joint space goal).
```
./moveit_client left_arm -f <path_to_file>
```
For example
```
./moveit_client left_arm -f ../../cfg/left_clearaway.yml
```
Now, the planner server should show a message
```
A plan has been made. See it in RViz [check Show Trail and Show Collisions]
```
indicating the plan has been made. You can actually visualize the trajectory
in RVIZ by MotionPlanning -> Planned Path -> check Show Trail. There are other visualization toggles you can play with.

Caveats:
1. The `moveit_planner` plans for only one goal at a time. You must manually
   clear the previous goal by running `./moveit_client left_arm -k`
2. If you want to **execute** a motion plan, you need to do
   `./moveit_client left_arm -e`. **NOTE: NOT WORKING. (Works but there is a weird delay?)**


## Troubleshooting

### AR detector frequency too low (<1hz)
Found a related question:
https://answers.ros.org/question/275598/ar_track_alvar-running-too-slow1hz/

Solution:
Adding `<param name="max_frequency" type="double" value="10" />` to the node
tag in the launch file allowed me to control the publication frequency of
ar_track_alvar. **NO THIS IS NOT USEFUL**

Also, run `ar_track_alvar` directly on the robot! **THIS IS MORE USEFUL. YOU GET 2HZ**
