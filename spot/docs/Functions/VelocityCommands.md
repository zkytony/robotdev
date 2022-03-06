# Velocity Commands

The main gRPC service is the RobotCommand service, and the request
is created and sent via the [synchro_velocity_command](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command#bosdyn.client.robot_command.RobotCommandBuilder.synchro_velocity_command) SDK function.

Spot ROS provides a more ROS-friendly interface to send velocity commands. See [documentation](https://www.clearpathrobotics.com/assets/guides/melodic/spot-ros/ros_usage.html#controling-the-velocity). To use this:

1. Release control from the tablet. To do this, press "X" to sit the robot ->  press the power button icon -> Under "CONTROL", press "RELEASE CONTROL".
2. Set parameters "auto_claim", "auto_power_on" and "auto_stand" to be all True in `spot_driver/config/spot_ros.yaml`.
3. Now, (re)start the driver `roslaunch rbd_spot_robot driver.launch control:=true`. The robot should now power its motor on, and then stand up by itself. This means now, the spot driver has _control_ of spot!
4. You can now run a velocity command, for example:
    ```
    rostopic pub /spot/cmd_vel geometry_msgs/Twist "linear:
      x: 0.0
      y: 0.0
      z: 0.0
    angular:
      x: 0.0
      y: 0.0
      z: 0.3" -r 10
    ````
    and you will see spot rotating in place 0.3rad/s. If you press `Ctrl+C` to stop this `rostopic pub` command, then the robot will stop rotating.

**Note, if you are done, don't just kill the spot ROS driver. It is preferred to go to the tablet and do "HIJACK CONTROL", and then control the robot (dock it or whatever) as needed.**
