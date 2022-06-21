cmake_minimum_required(VERSION 3.0.2)
project(rbd_spot_perception)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  spot_driver
  std_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
)

catkin_package()
catkin_python_setup()

include_directories(
  ${catkin_INCLUDE_DIRS}
)

################## Setting up dependencies ##############################
# Make Python headers available
find_package(PythonLibs ${PYTHON_MIN_VERSION} REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_MIN_VERSION 1.71)
find_package(Boost ${Boost_MIN_VERSION} REQUIRED)
set(Boost_LIBRARIES_USED
  boost_numpy
  boost_python)

# PCL
set(PCL_MIN_VERSION 1.10)
find_package(PCL ${PCL_MIN_VERSION} REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
message(STATUS "PCL_LIBRARIES: ${PCL_LIBRARIES}")

# roscpp and std_msgs and sensor_msgs
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(${sensor_msgs_INCLUDE_DIRS})



################ Building our libraries and executables ###################
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
message(STATUS "Boost_LIBRARIES_USED: ${Boost_LIBRARIES_USED}")
target_link_libraries(graphnav_map_publisher
  ${PYTHON_LIBRARIES}
  ${catkin_LIBRARIES}
  c_api_utils)
