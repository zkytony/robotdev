# Velocity Commands

The main gRPC service is the RobotCommand service, and the request
is created and sent via the [synchro_velocity_command](https://dev.bostondynamics.com/python/bosdyn-client/src/bosdyn/client/robot_command#bosdyn.client.robot_command.RobotCommandBuilder.synchro_velocity_command) SDK function.

Spot ROS provides a more ROS-friendly interface to send velocity commands. See [documentation](https://www.clearpathrobotics.com/assets/guides/melodic/spot-ros/ros_usage.html#controling-the-velocity).

