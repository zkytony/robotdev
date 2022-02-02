# Path to Spot workspace, relative to repository root;
# No begin or trailing slash.
SPOT_PATH="spot"

# The default IP address of Spot when it is connected
# to your remote computer via ethernet.
# Reference: https://support.bostondynamics.com/s/article/Spot-network-setup
#            (section Ethernet)
SPOT_ETH_IP="10.0.0.3"
SPOT_RLAB_IP="138.16.161.12"

function build_spot
{
    cd $repo_root/${SPOT_PATH}

    if catkin_make\
        --cmake-args\
        -DCMAKE_BUILD_TYPE=Release\
        -DPYTHON_EXECUTABLE=/usr/bin/python3\
        -DPYTHON_INCLUDE_DIR=/usr/include/python3.8\
        -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so; then
        echo "SPOT SETUP DONE." >> src/.DONE_SETUP
    else
        rm src/.DONE_SETUP
    fi
}

# Add a few alias for pinging spot.
#------------- Main Logic  ----------------

# Run this script by source setup_movo.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
fi

# We have only tested Spot stack with Ubuntu 20.04.
if ! ubuntu_version_equal 20.04; then
    echo "MOVO development requires Ubuntu 16.04 and ROS kinetic. Abort."
    return 1
fi

# Creates spot workspace.
# create the spot workspace directory
if [ ! -d "${SPOT_PATH}/src" ]; then
    mkdir -p ${SPOT_PATH}/src
fi

# create a dedicated virtualenv for spot workspace
if [ ! -d "${SPOT_PATH}/venv/spot" ]; then
    cd ${SPOT_PATH}/
    virtualenv -p python3 venv/spot
    cd ..
fi
repo_root=$PWD

# activate virtualenv; Note that this is the only
# functionality of this script if turtlebot has been setup
# before.
source ${SPOT_PATH}/venv/spot/bin/activate
export ROS_PACKAGE_PATH=$repo_root/${SPOT_PATH}/src/:${ROS_PACKAGE_PATH}

if first_time_build spot; then
    pip uninstall em
    pip install empy catkin-pkg rospkg defusedxml
    # other necessary packages
    pip install numpy
    # rosdep install dependencies
    rosdep update
    rosdep install --from-paths src --ignore-src -y
fi

# catkin make and end.
if first_time_build spot; then
    build_spot
else
    echo -e "If you want to build the spot project, run 'build_spot'"
fi
cd $repo_root
