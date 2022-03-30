moveit_client and moveit_planner were developed for the 3D writing project with Atsu.

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
