cmake_minimum_required(VERSION 3.0.2)
project(rbd_movo_perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
