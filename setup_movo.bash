# Path to MOVO workspace, relative to repository root;
# No begin or trailing slash.
MOVO_PATH="movo"

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

function install_pcl_1_11_1
{
    # https://pcl.readthedocs.io/projects/tutorials/en/pcl-1.11.1/compiling_pcl_posix.html
    if [[ ! $PWD = *robotdev ]]; then
        echo "You must be in the root directory of the robotdev repository."
        return
    fi

    if [ ! -d "thirdparty/pcl-1.11.1/" ]; then
        cd thirdparty
        mkdir pcl-1.11.1
        cd pcl-1.11.1
        wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.11.1/source.tar.gz
        tar xzvf source.tar.gz
        mv pcl/* .
        rm source.tar.gz
        cd ../..
    fi

    cd thirdparty/pcl-1.11.1/
    if [ -d "build" ]; then
        if confirm "pcl 1.11.1 build exists. Rebuild?"; then
            rm -rf build
        else
            cd $repo_root
            return
        fi
    fi
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j8
    sudo make -j8 install
}

function install_python_3_8
{
    ## Download and install Python
    PYTHON_VERSION=3.8.14
    if [ ! -d "$HOME/software/Python-$PYTHON_VERSION" ]
    then
        # Install prerequisites; https://github.com/pypa/packaging-problems/issues/573#issuecomment-1040587425
        sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
                                  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
                                  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

        mkdir -p ~/software
        cd ~/software/
        echo -e "Installing Python ${PYTHON_VERSION}"
        if [ ! -f "Python-${PYTHON_VERSION}.tgz" ]
        then
            wget "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
        fi
        tar xzvf "Python-${PYTHON_VERSION}.tgz"
        cd "Python-${PYTHON_VERSION}"
        ./configure
        make
        make test
        sudo make install

        # Install packages
        sudo apt install virtualenv
        sudo apt-get install python3-pip python3-yaml
        sudo pip3 install rospkg catkin_pkg
        sudo apt-get install python-catkin-tools python3-dev python3-numpy
        sudo apt install python3-catkin-pkg-modules python3-rospkg-modules python3-empy
        cd $repo_root
    fi
}

function install_tf2_python3
{
    # https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/?answer=326302#post-id-326302
    cd $repo_root/$MOVO_PATH/
    wstool init
    wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
    catkin_make --cmake-args\
                -DCMAKE_BUILD_TYPE=Release\
                -DPYTHON_EXECUTABLE=/usr/bin/python3\
                -DPYTHON_INCLUDE_DIR=/usr/include/python3.5m\
                -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.5m.so\
                -DCATKIN_WHITELIST_PACKAGES="tf2_py;tf2_ros;tf2;tf2_msgs;geometry2;tf2_eigen;tf2_geometry_msgs;tf2_sensor_msgs;tf2_tools;tf2_bullet"
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
# Run this script by source setup_movo.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
fi
repo_root=$PWD
MOVO2_IP="138.16.161.17"
MOVO_INTERNAL_NETWORK="10.66.171.0"


if first_time_build movo; then
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
cd $repo_root/$MOVO_PATH
