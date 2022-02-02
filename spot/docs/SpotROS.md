# Spot ROS

Clearpath robotics has a [github repo](https://github.com/clearpathrobotics/spot_ros)
that contains ROS packages for Spot.

The [documentation is here](https://www.clearpathrobotics.com/assets/guides/melodic/spot-ros/index.html).
The documentation is a little outdated at some places (e.g. they use python 3.6), but overall seems
to be usable with minor fixes. ROS Noetic on Ubuntu 20.04 works.

* Check out [this article for setting up ROS on Spot](https://www.clearpathrobotics.com/assets/guides/melodic/spot-ros/ros_setup.html#setup-spot-core)

   (You may skip Setup Networking if you have figured out networking using [our Networking notes](./Networking.md).
   
   Note that when you build the packages, you should run the following `catkin_make` command if you are using Python 3.8.
   ```
   catkin_make --cmake-args -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so
   ```
   Make sure that the paths are all valid.
   
   
* Refer to [this article for Spot Driver usage](https://www.clearpathrobotics.com/assets/guides/melodic/spot-ros/ros_usage.html). Spot driver is the
  main functionality of this suite of Spot ROS packages. It allows you to communicate with Spot and receive its sensor data, and control the body pose.

  When you first get started, run:
  ```
  roslaunch rbd_spot_robot driver.launch
  ```
  (this basically runs `spot_driver driver.launch` but with arguments such as username and password filled)
  
  Then run RVIZ:
  ```
  roslaunch spot_viz view_robot.launch
  ```
  You get something like:
  
  <img src="https://user-images.githubusercontent.com/7720184/152255159-e666b6ef-4038-41e6-b77e-72e4dc1cca78.png" width="500px"/>


  TODO: how to control the arm? (That is not the standard of Spot; would this Spot ROS stack still work?)

