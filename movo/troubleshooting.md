# PART 1: Issues encountered when trying to build kinova-movo in ROS Noetic
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


   Solution: copy and paste the latest code from [eband_local_planner](https://github.com/utexas-bwi/eband_local_planner)
   to replace the existing, outdated code for:
   - eband_local_planner.h and .cpp
   - eband_local_planner_ros.h and .cpp
   - conversions_and_types.h and .cpp

   After this issue, the build is successful!!!!!


## roslaunch: error: no such option: --sigint-timeout

```
started roslaunch server http://zephyr:34041/
remote[movo1-0] starting roslaunch
remote[movo1-0]: creating ssh connection to movo1:22, user[movo]
launching remote roslaunch child with command: [env ROS_MASTER_URI=http://movo2:11311 /home/movo/env.sh roslaunch -c movo1-0 -u http://zephyr:34041/ --run_id d27329ee-723c-11ec-92ff-00215cbdfd44 --sigint-timeout 15.0 --sigterm-timeout 2.0]
remote[movo1-0]: ssh connection created
remote[movo1-0]: Usage: roslaunch [options] [package] <filename> [arg_name:=value...]
       roslaunch [options] <filename> [<filename>...] [arg_name:=value...]

If <filename> is a single dash ('-'), launch XML is read from standard input.

roslaunch: error: no such option: --sigint-timeout

[movo1-0] killing on exit
RLException: remote roslaunch failed to launch: movo1
The traceback for the exception was written to the log file
[mapping_bringup-2] process has died [pid 41432, exit code 1, cmd /home/kaiyu/repo/robotdev/movo/src/kinova-movo/movo_common/si_utils/src/si_utils/timed_roslaunch 10 movo_demos assisted_teleop.launch local:=false __name:=mapping_bringup __log:=/home/kaiyu/.ros/log/d27329ee-723c-11ec-92ff-00215cbdfd44/mapping_bringup-2.log].
log file: /home/kaiyu/.ros/log/d27329ee-723c-11ec-92ff-00215cbdfd44/mapping_bringup-2*.log
```
The problem is `roslaunch` tries to run a command on movo.
The command is composed on my local machine running Noetic, but
movo's `roslaunch` is only Kinetic. So it doesn't recognize
the options `--sigint-timeout` because it is only [recently added](http://docs.ros.org/en/latest-available/changelogs/roslaunch/changelog.html).

Relevant [ROS Ask](https://answers.ros.org/question/372195/remote-roslaunch-from-host-noetic-machine-to-remote-melodic-machine-failed/).

I found that in `/opt/ros/noetic/lib/python3/dist-packages/roslaunch/remoteprocess.py`:
```python
class SSHChildROSLaunchProcess(roslaunch.server.ChildROSLaunchProcess):
    """
    Process wrapper for launching and monitoring a child roslaunch process over SSH
    """
    def __init__(self, run_id, name, server_uri, machine, master_uri=None, sigint_timeout=DEFAULT_TIMEOUT_SIGINT, sigterm_timeout=DEFAULT_TIMEOUT_SIGTERM):
        if not machine.env_loader:
            raise ValueError("machine.env_loader must have been assigned before creating ssh child instance")
        args = [machine.env_loader, 'roslaunch', '-c', name, '-u', server_uri, '--run_id', run_id,
                '--sigint-timeout', str(sigint_timeout), '--sigterm-timeout', str(sigterm_timeout)]
        ...
```
You can see that `--sigint-timeout` is appended into the args.

**WARNING: I REMOVED THIS.** It now looks like this:
```python
class SSHChildROSLaunchProcess(roslaunch.server.ChildROSLaunchProcess):
    def __init__(self, run_id, name, server_uri, machine, master_uri=None, sigint_timeout=DEFAULT_TIMEOUT_SIGINT, sigterm_timeout=DEFAULT_TIMEOUT_SIGTERM):
        if not machine.env_loader:
            raise ValueError("machine.env_loader must have been assigned before creating ssh child instance")
        args = [machine.env_loader, 'roslaunch', '-c', name, '-u', server_uri, '--run_id', run_id]
                # '--sigint-timeout', str(sigint_timeout), '--sigterm-timeout', str(sigterm_timeout)]
   ...
```


## Unable to run Teleop

I cannot start teleop with movo and use joy stick to control it,
if I do the roslaunch command on my remote computer.

The error is really I think a result of using Noetic.

Everywhere I looked people suggest using ROS Kinetic in docker.

**NOETIC DOES NOT WORK. ROLL BACK TO 16.04**


# PART 2: Issues encountered when trying to setup movo stack on 20.04 through a Docker container of Ubuntu 16.04 and ROS Kinetic

**Motivation:** Although I could build kinova-movo in 20.04 with ROS Noetic, the packages
don't work as expected and wierd errors are thrown. So I think the right way is to
set up a Ubuntu 16.04 environment and install ROS Kinetic. I am unable to boot into
USB on this laptop "Zephyr" I am using, and also, using a container is a more long term
solution, as Deemer points out. As a result, I am going to set up a Docker container
based on a Ubuntu 16.04 image in which I will install ROS Kinetic.

I have resolved all the problems without taking a note.


# PART 3: Working on MOVO remotely using Docker container

##  Server u'movo1' not found in known_hosts

   When running `roslaunch movo_demos robot_assisted_teleop.launch` inside the docker container to launch the joystick nodes, I get an error:
   ```
   ...
   remote[movo1-0]: failed to launch on movo1:

   Unable to establish ssh connection to [movo@movo1:22]: Server u'movo1' not found in known_hosts


   [movo1-0] killing on exit
   unable to start remote roslaunch child: movo1-0
   The traceback for the exception was written to the log file
   [mapping_bringup-2] process has died [pid 2163, exit code 1, cmd /home/kaiyu/repo/robotdev/movo/src/kinova-movo/movo_common/si_utils/src/si_utils/timed_roslaunch 10 movo_demos assisted_teleop.launch local:=false __name:=mapping_bringup __log:=/home/kaiyu/.ros/log/76870f54-7333-11ec-92ff-00215cbdfd44/mapping_bringup-2.log].
   log file: /home/kaiyu/.ros/log/76870f54-7333-11ec-92ff-00215cbdfd44/mapping_bringup-2*.log

   ```
## "libGL error: No matching fbConfigs or visuals found", "libGL error: failed to load driver: swrast", and "Could not initialize OpenGL for RasterGLSurface, reverting to RasterSurface."

When I am starting `rviz` on my office computer inside the same docker container, I get:
```
$ rviz
[ INFO] [1642208663.492000075]: rviz version 1.12.17
[ INFO] [1642208663.492042508]: compiled against Qt version 5.5.1
[ INFO] [1642208663.492050232]: compiled against OGRE version 1.9.0 (Ghadamon)
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
Could not initialize OpenGL for RasterGLSurface, reverting to RasterSurface.
libGL error: No matching fbConfigs or visuals found
libGL error: failed to load driver: swrast
Segmentation fault (core dumped)
```


   Summary of solution:

   1. Install `nvidia-docker`. That will provide you with the ability to use the tag `--runtime=nvidia` for `docker run`. Follow the [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). It is just a few steps. Make sure their test works.

   2. Use `nvidia/cudagl:9.0-base-ubuntu16.04` as base image instead of `ros:kinetic`. Install ROS Kinetic through Dockerfile. Essentially do:
     ```Dockefile
     FROM nvidia/cudagl:9.0-base-ubuntu16.04

     ENV UBUNTU_RELEASE=xenial
     RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $UBUNTU_RELEASE main" > /etc/apt/sources.list.d/ros-latest.list'
     RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

     RUN apt-get update && apt-get install -y \
         ros-kinetic-desktop-full \
      && rm -rf /var/lib/apt/lists/*
     ```
    [Credit.](https://github.com/ros-visualization/rviz/issues/1170#issuecomment-632188358)

   3. Use the `--runtime=nvidia` flag when doing `docker run` to start your
      container. This follows the command in the second approach mentioned in the ROS [Hardware Acceleration tutorial](http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration).

   Note that all of these changes have been incoporated into our setup.
   There is now `Dockerfile.kinetic.nvidia`, and you provide the
   `--nvidia` flag to both the `build.kinetic.sh` and `run.kinetic.sh`
   if you are running this on a computer with NVidia GPU.
   
## Able to see /tf under `rostopic list` but gets no message when doing `rostopic echo tf`

This happened after I ran `setup_movo.bash` successfully and I can ping both MOVO1 and MOVO2.

The solution is to add the IP addresses to MOVO1 and MOVO2 in your `/etc/hosts` file:
```
...
# for movo
10.66.171.2 movo1
10.66.171.1 movo2
138.16.161.17 movo
```
This [ROS Ask answer](https://answers.ros.org/question/48240/rostopic-list-works-but-rostopic-echo-does-not/?answer=213729#post-id-213729) helped me.


# UNABLE TO IMPORT BASIC ROS PYTHON PACKAGE
Today (02/26/2022), Eric tried to setup this MOVO docker stack on his computer.
Everything built fine. However, he could not import `rbd_movo_motor_skills.motion_planning`. The symptoms are:
0. The import fails.
1. When we try to build a basic ROS package with a pythom module (called the same name as the ROS package itself) using `catkin_make -DCATKIN_WHITELIST_PACKAGES="<pkg_name>"`, the `pkg_name` does not show up in `devel/lib/python2.7/site-packages`. 
2. The `__init__.py` file for `rbd_movo_motor_skills` became empty.

**INVESTIGATION.**
In the beginning, my setup was fine. When Eric ran into the issue, I wanted to reproduce it on my computer.
So I removed my `build/` and `devel` directories. I also removed `.DONE_SETUP`. I ran `source setup_movo.bash` again and it rebuilt everything.
Everything finished 100% without an error. But, I then ran into the same symptoms as Eric.

I noticed, as soon as I ran `source setup_movo.bash`, two new packages were pip-installed:

![image](https://user-images.githubusercontent.com/7720184/155858716-d2cfa7c8-9e53-4365-a4e2-4e22727f3dc4.png)

This is not normal because I had everything working so nothing should need to be installed again.

And then, I found on [this Github issue thread response](https://github.com/Azure/azure-functions-python-worker/issues/233#issuecomment-436819580)
that `pathlib` is ONLY for Python 3.4+. 
Another reason to support this is the problem is that I added the line `pip install pathlib` to `setup_movo.bash` AFTER I had the package structure (rbd_movo_motor_skills) done and built successfully. And that was added for "need to install pathlib & pyyaml for setup movo dev (python2.7)" (see commit [6cf489da34b3895de6](https://github.com/zkytony/robotdev/commit/6cf489da34b3895de611bfdcce46b59d78d530db)). This is not for some critical reason as it appears. I have no idea why I needed to even install `pathlib` at all. I know that `pyyaml` might have been done to parse the skill file. And it should be fine to install it. It is indeed the case that [pathlib is not available for Python 2.7: [this link](https://docs.python.org/2.7/library/pathlib.html) gives 404). So!

*THE ABOVE issue with pathlib is NOT THE PROBLEM.* The Github post said `pathlib` is part of the standard libary after Python 3.4 (so you shouldn't install it yourself separately). But before that, you could. So, installing pathlib for Python 2.7 itself is not a problem.

I realized the issue could most likely be in `setup.py` of `rbd_movo_motor_skills`. Instead of saying `packages=find_packages()` you should do `packages=['rbd_movo_motor_skills']`. I tried building a new docker container and inside it, just create an empty ROS workspace again and a basic ROS packkage with the understood way to set it up to make it python-importable. However, when I used `find_packages()`, I saw the exact same symptoms - the package is not built inside `devel`. But after I changed it to explicitly listing the package name, then it is built and I saw it in `devel`. WTF!! 

[This commit](https://github.com/zkytony/robotdev/commit/3f3011cd11c5521a5162100b180b76ae12a83c89) is proof that this setup.py mishap is most likely why we had our issues.

*UPDATE*: THIS IS THE PROBLEM!!!
