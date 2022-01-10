# Run this script by source setup_movo.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
fi
repo_root=$PWD
MOVO2_IP="138.16.161.17"
MOVO_INTERNAL_NETWORK="10.66.171.0"

# Always assume at the start of a function,
# or any if clause, the working directory is
# the root directory of the repository.

function build_movo_stack
{
    cd movo
    if catkin_make; then
        echo "MOVO SETUP DONE." >> src/.DONE_SETUP
    else
        rm src/.DONE_SETUP
    fi
    cd $repo_root
}

function setup_move_remote_pc
{
    if confirm "Are you working on the real robot (i.e. setup ROS_MASTER_URI) ?"; then
        echo -e "OK"
        export ROS_HOSTNAME=$(hostname)
        export ROS_MASTER_URI="http://movo2:11311"
        export ROS_IP="138.16.161.191"
        echo -e "You computer has been configured. Now do:"
        echo -e "- ssh movo@movo2"
        echo -e "- ssh into movo1 from movo2 (run ssh movo1)"
        echo -e "- sudo ifconfig wlan0 down in movo1"
        echo -e "- go back to movo2, run:"
        echo -e "   roslaunch movo_bringup movo_system.launch"
        cd $repo_root
    fi
}

# Returns true if this is the first time
# we build the ROS packages related to a robot
function first_time_build
{
    if [ ! -e "movo/src/.DONE_SETUP" ]; then
        # has not successfully setup
        true && return
    else
        false
    fi
}

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

##------------------- Main Setup Logic ------------------ ##
# use ros
if ! useros; then
    echo "Cannot use ROS. Abort."
    exit 1
fi

# Creates movo workspace.
# create the movo workspace directory
if [ ! -d "movo/src" ]; then
    mkdir -p movo/src
fi
# create a dedicated virtualenv for movo workspace
if [ ! -d "movo/venv/movo" ]; then
    cd movo/
    virtualenv -p python3 venv/movo
    cd ..
fi

# activate virtualenv; Note that this is the only
# functionality of this script if turtlebot has been setup
# before.
source movo/venv/movo/bin/activate
export ROS_PACKAGE_PATH=$repo_root/movo/src/:${ROS_PACKAGE_PATH}
echo -e "Adding route to movo's internal network. Requires sudo rights"
sudo route add -net ${MOVO_INTERNAL_NETWORK}\
     netmask 255.255.255.0\
     gw ${MOVO2_IP}\
     dev $(get_rlab_interface)

# Install necessary packages
if first_time_build; then
    # ros python packages
    pip uninstall em
    pip install empy catkin-pkg rospkg defusedxml
    # other necessary packages
    pip install numpy
    pip install netifaces
    if ubuntu_version_equal 20.04; then
        sudo apt install ros-noetic-moveit
    else
        echo -e "Unable to install required movo packages due to incompatible Ubuntu version."
        exit 1
    fi
    install_libfreenect2
    sudo apt-get install libopencv-core-dev
fi

# Download the kinova movo stack; first try to do submodule update;
if [ ! -d "movo/src/kinova-movo" ]; then
    cd movo/src
    git submodule add git@github.com:zkytony/kinova-movo.git
    cd ../..
fi

# add movo_motor_skills to kinova-movo/movo_apps
if [ ! -d "movo/src/kinova-movo/movo_apps/movo_motor_skills" ]; then
    cd movo/src/kinova-movo/movo_apps
    git submodule add git@github.com:zkytony/movo_motor_skills.git
    cd $repo_root
fi

# Start building
if first_time_build; then
    build_movo_stack
elif confirm "rebuild?"; then
     build_movo_stack
else
    echo -e "If you want to build the movo project, run 'build_movo_stack'"
fi

if [ -e movo/src/.DONE_SETUP ]; then
    setup_move_remote_pc

    if [ -e movo/functions.sh ]; then
        source movo/functions.sh
    fi
fi

cd $repo_root
