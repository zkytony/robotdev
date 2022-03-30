# Structure

`rbd_movo_action` is contains packages related to make movo
perform motion-planning or manipulation-involved actions.

`rbd_movo_perception` is related to pure perception such as
AR tag detection based on sensors.

`rbd_movo_system` deals with launching multi-component systems such as navigation and mapping.


Note that all packages are designed to be used on Ubuntu 16.04 and ROS kinetic.

These packages are also present on the MOVO robot.
Individual symbolic links to all of them are created under `movo@movo2:~/movo_ws/src/rbd_movo/<pkd_name>`,
that point to `movo@movo2:~/kaiyu/repo/robotdev/movo/src/<pkd_name>`.


## Create package
```
catkin_create_pkg <your_package_name> std_msgs roscpp
```
