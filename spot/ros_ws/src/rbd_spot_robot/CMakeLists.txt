cmake_minimum_required(VERSION 3.0.2)
project(rbd_spot_robot)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

catkin_package()

catkin_python_setup()

# install(
#   PROGRAMS
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

include_directories(
  ${catkin_INCLUDE_DIRS}
)
