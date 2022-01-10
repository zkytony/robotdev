## Could not find a configuration file for package "OpenCV" that is compatible with requested version "3".
This error happens when building `aruco`
```
-- ==> add_subdirectory(kinova-movo/movo_common/movo_third_party/aruco_ros/aruco)
CMake Error at kinova-movo/movo_common/movo_third_party/aruco_ros/aruco/CMakeLists.txt:6 (find
_package):
  Could not find a configuration file for package "OpenCV" that is compatible
  with requested version "3".

  The following configuration files were considered but not accepted:

    /usr/lib/x86_64-linux-gnu/cmake/opencv4/OpenCVConfig.cmake, version: 4.2.0
```
I changed the following line in `kinova-movo/movo_common/movo_third_party/aruco_ros/aruco/CMakeLists.txt`
```
find_package(OpenCV 3 REQUIRED)
```
to
```
find_package(OpenCV 4 REQUIRED)
```
This error is passed.

## Could NOT find `moveit_msgs` (missing: `moveit_msgs_DIR`)

This one is due to missing installation of Moveit.
Get it from [here](https://moveit.ros.org/install/).
```
sudo apt install ros-noetic-moveit
```
This resolved it.

## Could not find a package configuration file provided by "boost_signals"
```
-- +++ processing catkin package: 'face_detector'
-- ==> add_subdirectory(kinova-movo/movo_common/movo_third_party/people/face_detector)
-- Using these message generators: gencpp;geneus;genlisp;gennodejs;genpy
CMake Error at /usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0/BoostConfig.cmake:117 (find_packag
e):
  Could not find a package configuration file provided by "boost_signals"
  (requested version 1.71.0) with any of the following names:

    boost_signalsConfig.cmake
    boost_signals-config.cmake

dd the installation prefix of "boost_signals" to CMAKE_PREFIX_PATH or set
  "boost_signals_DIR" to a directory containing one of the above files.  If
  "boost_signals" provides a separate development package or SDK, be sure it
  has been installed.
Call Stack (most recent call first):
  /usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0/BoostConfig.cmake:182 (boost_find_component)
  /usr/share/cmake-3.16/Modules/FindBoost.cmake:443 (find_package)
  kinova-movo/movo_common/movo_third_party/people/face_detector/CMakeLists.txt:23 (find_packag
e)
...
```
I saw that at `kinova-movo/movo_common/movo_third_party/people/face_detector/CMakeLists.txt`
there is a line:
```
find_package(Boost REQUIRED COMPONENTS signals system thread)
```

According to [this ROS Answers post](https://answers.ros.org/question/333142/boost_signals-library-not-found/)
and [this github issue](https://github.com/ros/geometry2/pull/354),
`signals` is no longer required.

This resolved it.

## Could not find a package configuration file provided by "freenect2" with

```
-- +++ processing catkin package: 'kinect2_bridge'
-- ==> add_subdirectory(kinova-movo/movo_common/movo_third_party/iai_kinect2/kinect2_bridge)
CMake Error at kinova-movo/movo_common/movo_third_party/iai_kinect2/kinect2_bridge/CMakeLists.
txt:22 (find_package):
  Could not find a package configuration file provided by "freenect2" with
  any of the following names:

    freenect2Config.cmake
    freenect2-config.cmake

  Add the installation prefix of "freenect2" to CMAKE_PREFIX_PATH or set
  "freenect2_DIR" to a directory containing one of the above files.  If
  "freenect2" provides a separate development package or SDK, be sure it has
  been installed.

```
Solution: Install [libfreenect2](https://github.com/OpenKinect/libfreenect2).

My bash script code:
```bash
function install_libfreenect2
{
    # follow instructions here: https://github.com/OpenKinect/libfreenect2#linux
    if [ ! -d "thirdparty/libfreenect2/" ]; then
        cd thirdparty
        git clone https://github.com/OpenKinect/libfreenect2.git
        cd ..
    fi

    cd thirdparty/libfreenect2
    if [ ! -d "build" ]; then
        sudo apt-get install build-essential cmake pkg-config
        sudo apt-get install libusb-1.0-0-dev
        sudo apt-get install libturbojpeg0-dev
        sudo apt-get install libglfw3-dev
        # OpenCL, CUDA, skipped. Assume CUDA is already installed.
        sudo apt-get install libva-dev libjpeg-dev
        sudo apt-get install libopenni2-dev
        mkdir build && cd build
        # Note: You need to specify cmake
        # -Dfreenect2_DIR=$HOME/freenect2/lib/cmake/freenect2 for CMake based
        # third-party application to find libfreenect2.
        cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2
        make
        make install
    fi
    cd $repo_root
}
```
This resolved the issue.

## Could not find a package configuration file provided by "realsense2" with any of the following names

```
-- +++ processing catkin package: 'realsense2_camera'
-- ==> add_subdirectory(kinova-movo/movo_common/movo_third_party/realsense2_camera)
-- Using these message generators: gencpp;geneus;genlisp;gennodejs;genpy
CMake Warning at kinova-movo/movo_common/movo_third_party/realsense2_camera/CMakeLists.txt:27
(find_package):
  By not providing "Findrealsense2.cmake" in CMAKE_MODULE_PATH this project
  has asked CMake to find a package configuration file provided by
  "realsense2", but CMake did not find one.

  Could not find a package configuration file provided by "realsense2" with
  any of the following names:

    realsense2Config.cmake
    realsense2-config.cmake

  Add the installation prefix of "realsense2" to CMAKE_PREFIX_PATH or set
  "realsense2_DIR" to a directory containing one of the above files.  If
  "realsense2" provides a separate development package or SDK, be sure it has
  been installed.


CMake Error at kinova-movo/movo_common/movo_third_party/realsense2_camera/CMakeLists.txt:29 (m
essage):




   Intel RealSense SDK 2.0 is missing, please install it from https://github.com/IntelRealSens
e/librealsense/releases
```
Solution: install realsense.
```bash
function install_librealsense
{
    # Follow instructions here: https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md
    sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
    sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
    sudo apt-get install librealsense2-dkms
    sudo apt-get install librealsense2-utils
    sudo apt-get install librealsense2-dev
    sudo apt-get install librealsense2-dbg
}
```
The key is to install the last two packages for developers.
This is resolved.

**UPDATE 01/09/2022 19:38: realsense is removed (by kinova themselves). My repo has been updated.**

## Assorted Catkin Build Errors

1. A bunch of errors regarding migration from OpenCV 3 to 4. The key points appears to be:

  * Use
     ```
     #include <opencv2/opencv.hpp>
     ```
     and several other `<opencv2>` header files.

  * Change names, e.g. `CV_AA` to `LINE_AA`, etc. Mostly can be done through googling.

2. Errors about migrating TF to TF2.

   This happened for the package `movo_assisted_teleop`.

   The first thing is to figure out what to do. I decided to port over to TF2
   instead of figuring out how to make TF work, because it is noetic, because
   we are moving forward.

   First, there is a `tf2_ros` namespace. You need to include the tf2 header
   files correctly. Refer to documentation online for noetic.

   The notable one, regarding passing in a pointer to `tf2_ros::Buffer`
   to `base_local_planner`, I asked a question on [RosAsk](https://answers.ros.org/question/393967/cannot-convert-tf2_rostransformlistener-to-tf2_rosbuffer/).
   Later I figured out a way to deal with it, by:
   ```cpp
   // ./movo_navigation/movo_assisted_teleop/include/movo_assisted_teleop/movo_assisted_teleop.h
   tf2_ros::Buffer *tfb_;
   tf2_ros::TransformListener tf_;
   ...
   ```
   and then initialize the `tf_` by passing the dereference of `*tfb_`. I
   came up with this solution based on reading the documentation.


3. Remove realsense

4. Gazebo migration (7? to 11)

   This was a lot of effort. Check out the following commits:

   * bf924d2e32fb8374bb75bc28804386dafcbfdbd4
   * 5cfc15a7eb49b3804a76c6a8d2e3d1676233e6db


5. `boost::shared_ptr` to `std::shared_ptr` in `joint_trajectory_controller`

    The error is:
    ```
     /home/kaiyu/repo/robotdev/movo/src/kinova-movo/movo_common/movo_third_party/joint_trajectory_controller
    /include/joint_trajectory_controller/joint_trajectory_controller_impl.h:109:49: error: conversion from
    ‘urdf::JointConstSharedPtr’ {aka ‘std::shared_ptr<const urdf::Joint>’} to non-scalar type ‘joint_trajec
    tory_controller::internal::UrdfJointConstPtr’ {aka ‘boost::shared_ptr<const urdf::Joint>’} requested
      109 |     UrdfJointConstPtr urdf_joint = urdf.getJoint(joint_names[i]);
          |                                    ~~~~~~~~~~~~~^~~~~~~~~~~~~~~~
      ```

    [This RosAsk thread](https://answers.ros.org/question/366073/urdfmodel-boostshared_ptrurdflink-boostshared_ptrurdfjoint/)
    basically asked about the same error. The solution is to change
    `boost::shared_ptr` to `std::shared_ptr` when declaring the type
    of `UrdfJointConstPtr`. The solution referenced [this commit](https://github.com/RethinkRobotics/baxter_simulator/pull/130/commits/e2f874ddb937077a7b92e4f5fe9cde3b64446cc0).


6. Advanced C++ abstract class error

   ```
  In file included from /opt/ros/noetic/include/class_loader/class_loader_core.hpp:45,
                   from /opt/ros/noetic/include/class_loader/class_loader.hpp:46,
                   from /opt/ros/noetic/include/class_loader/multi_library_class_loader.hpp:42,
                   from /opt/ros/noetic/include/pluginlib/class_loader.hpp:38,
                   from /opt/ros/noetic/include/costmap_2d/costmap_2d_ros.h:50,
                   from /opt/ros/noetic/include/nav_core/base_local_planner.h:42,
                   from /home/kaiyu/repo/robotdev/movo/src/kinova-movo/movo_common/movo_third_party/eband
  _local_planner/include/eband_local_planner/eband_local_planner_ros.h:44,
                   from /home/kaiyu/repo/robotdev/movo/src/kinova-movo/movo_common/movo_third_party/eband
  _local_planner/src/eband_local_planner_ros.cpp:38:
  /opt/ros/noetic/include/class_loader/meta_object.hpp: In instantiation of ‘B* class_loader::impl::MetaO
  bject<C, B>::create() const [with C = eband_local_planner::EBandPlannerROS; B = nav_core::BaseLocalPlan
  ner]’:
  /opt/ros/noetic/include/class_loader/meta_object.hpp:196:7:   required from here
  /opt/ros/noetic/include/class_loader/meta_object.hpp:198:12: error: invalid new-expression of abstract
  class type ‘eband_local_planner::EBandPlannerROS’
    198 |     return new C;
        |
   ```

    The problem is the class `eband_local_planner::EBandPlannerROS` is no longer
    part of noetic. It was in [indigo](http://docs.ros.org/en/indigo/api/eband_local_planner/html/classeband__local__planner_1_1EBandPlannerROS.html),
    but the corresponding [noetic link](http://docs.ros.org/en/noetic/api/eband_local_planner/html/classeband__local__planner_1_1EBandPlannerROS.html) is 404.
    In fact, this class was still present in melodic. It looks like
    [teb_local_planner](http://wiki.ros.org/teb_local_planner), however, survived
    till noetic.
