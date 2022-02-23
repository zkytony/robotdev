cmake_minimum_required(VERSION 3.0.2)
project(rbd_spot)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package()
catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
