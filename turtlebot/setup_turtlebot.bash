# Run this script by source
user_pwd=$PWD

# root directory of robotdev
repo_root=${ROBOTDEV_PATH:-$HOME/repo/robotdev}
. "$repo_root/tools.sh"

# Path to Turtlebot workspace
TURTLEBOT_PATH="$repo_root/turtlebot"

#------------- FUNCTIONS  ----------------
# Always assume at the start of a function,
# or any if clause, the working directory is
# the root directory of the repository.

function build_turtlebot
{
    cd $TURTLEBOT_PATH
    if catkin_make; then
        # cmake build is successful. Mark
        echo "TURTLEBOT SETUP DONE." >> $TURTLEBOT_PATH/src/.DONE_SETUP
    else
        rm $TURTLEBOT_PATH/src/.DONE_SETUP
    fi
}

#------------- Main Logic  ----------------
# use ros
if ! useros; then
    echo "Cannot use ROS. Abort."
    exit 1
fi

# Creates turtlebot workspace.
# create the turtlebot workspace directory
if [ ! -d "${TURTLEBOT_PATH}/src" ]; then
    mkdir -p ${TURTLEBOT_PATH}/src
fi
# create a dedicated virtualenv for turtlebot workspace
if [ ! -d "${TURTLEBOT_PATH}/venv/turtlebot" ]; then
    cd ${TURTLEBOT_PATH}/
    virtualenv -p python3 venv/turtlebot
    cd ..
fi

# activate virtualenv; Note that this is the only
# functionality of this script if turtlebot has been setup
# before.
source ${TURTLEBOT_PATH}/venv/turtlebot/bin/activate
export ROS_PACKAGE_PATH=$repo_root/${TURTLEBOT_PATH}/src/turtlebot3_simulations:${ROS_PACKAGE_PATH}

if first_time_build $TURTLEBOT_PATH; then
    sudo apt-get update
    # ros python packages
    pip uninstall em
    pip install empy catkin-pkg rospkg defusedxml
    # other necessary packages
    pip install numpy
    sudo apt install libignition-physics2-dev
    if ubuntu_version_equal 20.04; then
        sudo apt-get install ros-noetic-gazebo-ros
        sudo apt-get install ros-noetic-turtlebot3-msgs
        sudo apt-get install ros-noetic-turtlebot3
    else
        echo -e "Unable to install required turtlebot packages due to incompatible Ubuntu version."
        exit 1
    fi
fi

if [ ! -d "${TURTLEBOT_PATH}/src/turtlebot3_simulations" ]; then
    # Follow the instructions here: https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/
    cd ${TURTLEBOT_PATH}/src/
    git submodule add https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
    cd ../..
fi

# catkin make and end.
if first_time_build $TURTLEBOT_PATH; then
    build_turtlebot
else
    echo -e "If you want to build the turtlebot project, run 'build_turtlebot'"
fi
source $TURTLEBOT_PATH/devel/setup.bash
cd $user_pwd
