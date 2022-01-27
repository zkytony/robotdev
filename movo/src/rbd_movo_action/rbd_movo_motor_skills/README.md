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

## Writing a Skill

You can define a skill by creating a `.skill` file, which is essentially a YAML file.
Look at an example [here general.left_arm_movement.skill](cfg/skills/https://github.com/zkytony/robotdev/blob/master/movo/src/rbd_movo_action/rbd_movo_motor_skills/cfg/skills/general.left_arm_movement.skill).

Refer to the documentation at [motion_planning/framework.py](https://github.com/zkytony/robotdev/blob/master/shared/python/motion_planning/framework.py)
for the definition of the skill file and how the framework works. Note that that framework is not specific to MOVO.

Check out [these steps](cfg/skills/README.md) for creating a checkpoint that contains a goal arm configuration.

## Brief Example to Load and Run a Skill
After you have written a Skill file, you could create a `SkillManager` to load it. Once you call `.run()` of the SkillManager, the skill will begin to execute.
As a simplest example (basically how stuff in `tests/` are written):
```python
# test_framework_grasp_bottle.py
import rospy
from rbd_movo_motor_skills.motion_planning.framework import SkillManager

def test():
    rospy.set_param("skill/pkg_base_dir", "../")
    rospy.set_param("skill/pkg_name", "rbd_movo_motor_skills")
    skill_file_path = "scili8.livingroom.pickup_bottle_from_chair.skill"
    mgr = SkillManager(skill_file_path)
    mgr.run()

if __name__ == "__main__":
    test()
```
Replace `scili8.livingroom.pickup_bottle_from_chair.skill` with your skill file (you should place it under `cfg/skills`).

## What to Run
Before running the skill, make sure:
1. MOVO system has started
2. AR Tag detector is running. You can run `roslaunch rbd_movo_perception artag_tracker.launch` (Do it on MOVO)
3. Point cloud processing is running (necessary to enable Moveit! OctoMap collision avoidance to work properly): `roslaunch rbd_movo_perception process_pointcloud.launch`.
4. Moveit! OctoMap collision avoidance is working (Open RVIZ, and visualize the topic `move_group/filtered_points`).

If the above all pass, we can run a skill.
Suppose you have written a script as in the example above. Then you will need to run:
```
python test_framework_grasp_bottle.py
```
This will start the SkillManager which will monitor the process of skill execution.
At any point, you can hit `Ctrl+C` which will terminate all processes.



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

3. Move arm. **This is extremely useful for carefully getting a desired arm configuration**

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

For convenience of setting AR Tag pose in a skill, you may want to check the artag pose with respect to base_link:
```
rosrun tf tf_echo base_link ar_marker_4
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

### Motion plan was found but it seems to be invalid (possibly due to postprocessing).
This happens when the arm seems to be close to Point Cloud obstacles.

According to [this discussion](https://groups.google.com/g/moveit-users/c/3ey_8A8mwsE):
>It is likely the path computed is grazing obstacles, due to the collision checking resolution being to coarse.
>You should open ompl_planning.yaml and change:
>longest_valid_segment_fraction: 0.05
>to something like
>longest_valid_segment_fraction: 0.02



## Gripper Seems Open, but Gripper command verifier does not terminate

The reason this happens is unclear. What happens is that the server for
the action `GripperCommandAction` does not report the goal is success or reached
even though the gripper is in the desired state. I have only observed this
happening when opening the gripper.

To deal with this, one nasty way is to reboot MOVO. A simpler way is
to just set different values for `Open Gripper` (try something between 0.9 to 1.0)
and repeatedly run the open gripper skill `general.left_arm.open_gripper.skill`. You
can do that conveniently by running
```
python test_framework_gripper_open.py
```
Just keep trying until this skill successfully finishes. It should be
a very quick skill to finish. So if it takes longer than 3 seconds,
stop this skill, reset the Gripper opening position, and try again.

**NOTE the gripper_verifier has been modifed to check if the gripper is moving at all. If not, then it will pass.**
