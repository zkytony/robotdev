# Run this script by source setup_movo.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
fi
repo_root=$PWD

# Always assume at the start of a function,
# or any if clause, the working directory is
# the root directory of the repository.

function build_movo_stack
{
    # Run the automatic installation script, if not already
    if confirm "Run movo setup script?"; then
        echo -e "OK"
        box_out "As the setup_remote_pc script runs,"\
                "it will prompt you at different points."\
                "Pay attention to the question when it asks."
        cd movo/src/kinova-movo/movo_pc_setup
        chmod +x setup_remove_pc
        echo -e "***** Executing setup_remove_pc script *****"
        ./setup_remove_pc
        echo "MOVO PC SETUP DONE." >> movo/kinova-movo/DONE_SETUP
        echo -e "Setup done."
        echo -e "Note: To run any of the sim_ functions please disconnect the remote PC from the robot."
        cd $repo_root
    fi
    cd movo
    catkin_make
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


##------------------- Main Setup Logic ------------------ ##
# use ros
if ! useros; then
    echo "Cannot use ROS. Abort."
    exit 1
fi

# update submodules (clone necessary stuff)
git submodule update --init --recursive

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

# Install necessary packages
if first_time_build; then
    # ros python packages
    pip uninstall em
    pip install empy catkin-pkg rospkg defusedxml
    # other necessary packages
    pip install numpy
    if ubuntu_version_equal 20.04; then
        sudo apt install ros-noetic-moveit
    else
        echo -e "Unable to install required movo packages due to incompatible Ubuntu version."
        exit 1
    fi
    install_libfreenect2
    install_librealsense
    sudo apt-get install libopencv-core-dev
fi

# Download the kinova movo stack; first try to do submodule update;
if [ ! -d "movo/src/kinova-movo" ]; then
    cd movo/src
    git submodule add git@github.com:zkytony/kinova-movo.git
    cd ../..
fi

if first_time_build; then
    build_movo_stack
else
    echo -e "If you want to build the movo project, run 'build_movo'"
fi
cd $repo_root
