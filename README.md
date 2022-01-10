# robotdev

My repository for robot-related development.


## Structure

Each robot gets a corresponding directory.  The directory is a designated ROS
workspace for that robot. The robot-specific code is maintained in
submodules (i.e. other repositories). This repository serves as the
hub for all the robot-related development efforts.

To setup your Linux machine (Ubuntu 20.04) for a robot:
```
source setup_{robot_name}.bash
```
