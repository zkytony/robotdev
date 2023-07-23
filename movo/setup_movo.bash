# Run this script by source
user_pwd=$PWD

# root directory of robotdev
repo_root=${ROBOTDEV_PATH:-$HOME/repo/robotdev}
. "$repo_root/tools.sh"

MOVO_PATH="$repo_root/movo"

MOVO2_IP="138.16.161.17"
MOVO_INTERNAL_NETWORK="10.66.171.0"


#------------- FUNCTIONS  ----------------
function install_libfreenect2
{
    # follow instructions here: https://github.com/OpenKinect/libfreenect2#linux
    if [ ! -d "thirdparty/libfreenect2/" ]; then
        cd thirdparty
        git clone https://github.com/OpenKinect/libfreenect2.git
        cd ..
    fi

    cd thirdparty/libfreenect2
    if [ -d "build" ]; then
        if confirm "libfreenect2 build exists. Rebuild?"; then
            rm -rf build
        else
            cd $repo_root
            return
        fi
    fi
    sudo apt-get install build-essential cmake pkg-config
    sudo apt-get install libusb-1.0-0-dev

    if ubuntu_version_equal 16.04; then
        sudo apt-get install libturbojpeg libjpeg-turbo8-dev
    elif ubuntu_version_equal 20.04; then
        sudo apt-get install libturbojpeg0-dev
    fi

    sudo apt-get install libglfw3-dev
    mkdir build && cd build
    # Note: You need to specify cmake
    # -Dfreenect2_DIR=$HOME/freenect2/lib/cmake/freenect2 for CMake based
    # third-party application to find libfreenect2.
    cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/freenect2
    make
    make install
    cd $repo_root
}

function setup_movo_remote_pc
{
    echo -e "Adding route to movo's internal network. Requires sudo rights"
    sudo route add -net ${MOVO_INTERNAL_NETWORK}\
         netmask 255.255.255.0\
         gw ${MOVO2_IP}\
         dev $(get_rlab_interface)
    export ROS_MASTER_URI="http://movo2:11311"
    export ROS_IP="$(get_rlab_ip)"
    echo -e "You computer has been configured. Now do:"
    echo -e "- ssh movo@movo2"
    echo -e "- ssh into movo1 from movo2 (run ssh movo1)"
    echo -e "- sudo ifconfig wlan0 down in movo1"
    echo -e "- go back to movo2, run:"
    echo -e "   roslaunch movo_bringup movo_system.launch"
    cd $repo_root
}


#------------- Main Logic  ----------------
if first_time_build $MOVO_PATH; then
    if ubuntu_version_equal 16.04; then
        pip install netifaces
        pip install pathlib
        pip install pyyaml==3.11
        install_libfreenect2
        build_ros_ws $MOVO_PATH
    fi
fi

if ubuntu_version_equal 16.04; then
    useros
    export ROS_PACKAGE_PATH=$repo_root/${MOVO_PATH}/src/:${ROS_PACKAGE_PATH}
    source $repo_root/${MOVO_PATH}/devel/setup.bash
fi

# Even though MOVO is for Ubuntu 16.04, we may want
# to use it from a computer that runs ROS Noetic
# because we may have code in Python 3. The workspace
# that contains packages we want to use under 20.04
# are at '$MOVO_PATH/noetic_ws'
if ubuntu_version_equal 20.04; then
    if ! command -v ip &> /dev/null
    then
        echo "ip could not be found; installing..."
        sudo apt install iproute2
        pip install netifaces
        sudo apt-get install -y python3-pykdl
        sudo apt-get install ros-noetic-moveit-msgs
    fi
    if [ ! -d "${MOVO_PATH}/venv/movo" ]; then
        cd ${MOVO_PATH}/
        virtualenv -p python3 venv/movo
        cd ..
    fi
    source $repo_root/${MOVO_PATH}/venv/movo/bin/activate

    useros
    export ROS_PACKAGE_PATH=$repo_root/${MOVO_PATH}/noetic_ws/src/:$repo_root/${MOVO_PATH}/src/:${ROS_PACKAGE_PATH}
    source $repo_root/${MOVO_PATH}/noetic_ws/devel/setup.bash
    export PYTHONPATH="$repo_root/${MOVO_PATH}/venv/movo/lib/python3.8/site-packages:${PYTHONPATH}:/usr/lib/python3/dist-packages"
fi

if confirm "Are you working on the real robot (i.e. setup ROS_MASTER_URI, packet forwarding etc) ?"; then
    echo -e "OK"
    setup_movo_remote_pc
fi
cd $user_pwd
