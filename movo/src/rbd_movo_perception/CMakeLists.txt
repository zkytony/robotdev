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

add_executable(ar_pattern_detector src/ar_pattern_detector.cpp)
target_link_libraries(ar_pattern_detector    ${catkin_LIBRARIES})
