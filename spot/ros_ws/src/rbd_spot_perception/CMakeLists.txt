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

# graphnav map publisher; needs python embedding
find_package(PythonLibs ${PYTHON_MIN_VERSION} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
add_executable(graphnav_map_publisher
  src/rbd_spot_perception/graphnav_map_publisher.cpp)
target_include_directories(graphnav_map_publisher
  PUBLIC
  include/rbd_spot_perception)
target_link_libraries(graphnav_map_publisher
  ${PYTHON_LIBRARIES})
