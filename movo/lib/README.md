Note: These files are copied from `/usr/lib` on MOVO.
Even though you may copy these files to the `/usr/lib` directory
of your docker image, it might not be a good idea to run
code that use them on your computer (not tested and not sure what will happen).
If you want to control the arm or gripper etc., MOVO provides you
some action services for example:
```
/movo/head_controller/follow_joint_trajectory
/movo/head_controller/point_head_action
/movo/left_arm_controller/follow_joint_trajectory
/movo/left_gripper_controller/gripper_cmd
/movo/right_arm_controller/follow_joint_trajectory
/movo/right_gripper_controller/gripper_cmd
/movo/torso_controller/follow_joint_trajectory
```
