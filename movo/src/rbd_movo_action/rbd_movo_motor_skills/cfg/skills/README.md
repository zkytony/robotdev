Method for setting arm poses in skill:

We prefer to set end-effector poses (with respect to robot base)
as goals instead of joint positions for the purpose of ego-centric
down-stream manipulation planning.

To do that:

1. Sit beside the robot. Use `./movo_pose_publisher.py` to slowly move
   desired joints until the arm is in desired configuration.
   This involves, for example, running command like the following
    ```
    ./movo_pose_publisher.py -i 5 -v 5
    ```
   You can find `movo_pose_publisher.py` under `rbd_movo_motor_skills/scripts`.
   which rotates joint index 5 (theta_wrist_spherical_2_joint)
   clockwise by 5 degrees per second.

2. Obtain the end effector pose in the base frame:
     ```
     rosrun tf tf_echo base_link left_ee_link
     ```
    Note that "base_link" is the source frame (parent) in this case.

    You get output like:
      ```
      At time 1643156736.136
      - Translation: [0.946, 0.232, 0.850]
      - Rotation: in Quaternion [-0.202, -0.154, 0.000, 0.967]
                  in RPY (radian) [-0.422, -0.302, 0.065]
                  in RPY (degree) [-24.155, -17.286, 3.745]
      ```
      The pose of the end effector with respect to base_link is `[0.946, 0.232, 0.850, -0.202, -0.154, 0.000, 0.967]

3. Obtain the joint configuration's joint positions:
    ```
    rostopic echo /move_group/filtered/left_arm/joint_states
    ```
    You get output like:
       ```
       ---
       header:
         seq: 564368
         stamp:
           secs: 1643155583
           nsecs: 573668956
         frame_id: ''
       name: [left_shoulder_pan_joint, left_shoulder_lift_joint, left_arm_half_joint, left_elbow_joint,
         left_wrist_spherical_1_joint, left_wrist_spherical_2_joint, left_wrist_3_joint]
       position: [1.4514936341781812, 1.320691145658394, 0.08205465634643261, 1.3611895700391201, 0.06178333575783146, -0.252946507703508, 0.46026624158366536]
       velocity: [0.0, 0.0, 0.0, -0.0003408846195301425, -0.0004958321561296851, 0.0, -0.0004958321561296851]
       effort: [-3.5621814727783203, -10.962505340576172, 0.4460943639278412, 8.262934684753418, 0.1420287787914276, -1.7437231540679932, -0.6641568541526794]
       ---
       ```
      You should copy and paste the "position" and "name" fields.

4. Now, in the skill file, create a new checkpoint. Add as an `actuation_cue`
   the robot's EE pose as a 'goal' and the joint positions as a `path_constraint`,
   with some tolerance at the big joints. This is to avoid dramatic big rotations
   of the arms (even if you turned on collision checking with octomap such
   dramatic motion plans can happen and is not pleasant to watch).

   Example:
     ```yaml

     skill:
        ...

        - name: "Motion Plan to Prep"
          actuation_cues:
            - type: ArmPose
              args:
                side: left
                frame: "base_link"
                goal:
                  type: "pose"
                  pose:
                    position: [0.812, 0.466, 0.939]
                    orientation: [-0.596, -0.426, -0.030, 0.680]
                  tolerance: 0.1
                path_constraints:
                  - type: "joints"
                    positions: [1.6199996926727103, 1.365087905242778, 0.6605017672271316, 1.4191538037969127]
                    joint_names: [left_shoulder_pan_joint, left_shoulder_lift_joint, left_arm_half_joint, left_elbow_joint]
                    tolerance: [[-1.05, 1.05], [-1.05, 1.05], [-1.05, 1.05], [-1.05, 1.05]]
     ```
     Notice all the tolerances are set to within `[-1.05, 1.05]` which is
     between around [-60, 60] degrees with respect to the corresponding
     joint position constraint (just a heuristic).

     If you must rotate arm dramatically to achieve some kind of configuration,
     you should set that goal using joint positions instead of EE poses. And
     you should break this trajectory down to waypoints.
