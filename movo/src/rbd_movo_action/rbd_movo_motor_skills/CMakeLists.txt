cmake_minimum_required(VERSION 3.0.2)
project(rbd_movo_motor_skills)

# allow c++11
add_compile_options(-std=c++11)

set(CMAKE_EXPORT_COMPILE_COMMANDS "export compile_commands.json" ON)

find_package(catkin REQUIRED COMPONENTS
  pcl_msgs
  pcl_ros
  roscpp
  rospy
  std_msgs
)
catkin_python_setup()

add_action_files(
  DIRECTORY action
  FILES
  ExecMoveitPlan.action
  GetState.action
  PlanJointSpace.action
  PlanMoveEE.action
  PlanWaypoints.action
)

generate_messages (
  DEPENDENCIES
  geometry_msgs
  actionlib_msgs
  std_msgs
)

catkin_package()


install(
  PROGRAMS
  scripts/run_skill.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  shared/include
)

# shared motor skills library
add_library(motor_skills
  shared/src/framework.cpp
)

add_executable(hello_world shared/src/hello_world.cpp)
target_link_libraries(hello_world
  ${catkin_LIBRARIES}
  motor_skills
  yaml-cpp
)

# Test
add_executable(test_motor_skill_framework shared/tests/test_motor_skill_framework.cpp)
target_link_libraries(test_motor_skill_framework
  ${catkin_LIBRARIES}
  motor_skills
  yaml-cpp
)
