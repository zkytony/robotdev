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
