cmake_minimum_required(VERSION 3.0.2)
project(rbd_movo_motor_skills)

find_package(catkin REQUIRED COMPONENTS
  pcl_msgs
  pcl_ros
  roscpp
  rospy
  std_msgs
)
catkin_python_setup()
catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
)
