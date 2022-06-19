cmake_minimum_required(VERSION 3.0.2)
project(rbd_spot_perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  spot_driver
  std_msgs
  sensor_msgs
)

catkin_package()
catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Make Python headers available
find_package(PythonLibs ${PYTHON_MIN_VERSION} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# c api utility functions
add_library(c_api_utils
  src/rbd_spot_perception/utils/c_api_utils.cpp)
target_include_directories(c_api_utils
  PUBLIC
  include/rbd_spot_perception)

# graphnav map publisher; needs python embedding
add_executable(graphnav_map_publisher
  src/rbd_spot_perception/graphnav_map_publisher.cpp)
target_include_directories(graphnav_map_publisher
  PUBLIC
  include/rbd_spot_perception)
target_link_libraries(graphnav_map_publisher
  ${PYTHON_LIBRARIES}
  c_api_utils)
