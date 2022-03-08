# Spot Functions

This folder documents how you may DO something with Spot, for example, mapping, localization, navigation, keyboard control, etc.

Much of this functionality is built upon ROS and the [Spot ROS wrapper](https://github.com/clearpathrobotics/spot_ros).
You need to launch the Spot ROS driver. Here are the different ways to run the driver:

1. Observation mode (you do not need to control the robot programmatically; For example, for mapping, you will use the controller to joystick the robot, while you
    only need to stream sensor data from it to build the map).
   ```
   roslaunch rbd_spot_robot driver.launch
   ```
    
2. Control mode (if a lease is available, e.g. the controller has "RELEASED CONTROL", then you can control the robot using Spot ROS's acquired lease)
   ```
   roslaunch rbd_spot_robot driver.launch control:=true
   ```

3. Force control mode (EVEN IF a lease is NOT available, e.g. 
   the controller IS IN CONTROL, you can still force spot ROS to TAKE the lease. See [Lease.md](../Lease.md). This is equivalent to the HIJACK CONTROL button on the controller).
   ```
   roslaunch rbd_spot_robot driver.launch control:=true force:=true
   ```
