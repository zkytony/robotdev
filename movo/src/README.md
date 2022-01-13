# Structure

`rbd_movo_action` is contains packages related to make movo
perform motion-planning or manipulation-involved actions.

`rbd_movo_perception` is related to pure perception such as
AR tag detection based on sensors.

`rbd_movo_navigation` deals with navigation with movo.


Note that all packages are designed to be used on Ubuntu 16.04 and ROS kinetic.

## Create package
```
catkin_create_pkg <your_package_name> std_msgs roscpp
```
