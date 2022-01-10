# Run this script by source setup_turtlebot.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
fi
repo_root=$PWD

# Always assume at the start of a function,
# or any if clause, the working directory is
# the root directory of the repository.

function build_turtlebot
{
    cd $repo_root/turtlebot
    if catkin_make; then
        # cmake build is successful. Mark
        echo "TURTLEBOT SETUP DONE." >> src/.DONE_SETUP
    else
        rm src/.DONE_SETUP
    fi
}

function first_time_build
{
    if [ ! -e "turtlebot/src/.DONE_SETUP" ]; then
        # has not successfully setup
        true && return
    else
        false
    fi
}

##------------------- Main Setup Logic ------------------ ##
# use ros
if ! useros; then
    echo "Cannot use ROS. Abort."
    exit 1
fi

# Creates turtlebot workspace.
# create the turtlebot workspace directory
if [ ! -d "turtlebot/src" ]; then
    mkdir -p turtlebot/src
fi
# create a dedicated virtualenv for turtlebot workspace
if [ ! -d "turtlebot/venv/turtlebot" ]; then
    cd turtlebot/
    virtualenv -p python3 venv/turtlebot
    cd ..
fi

# activate virtualenv; Note that this is the only
# functionality of this script if turtlebot has been setup
# before.
source turtlebot/venv/turtlebot/bin/activate
export ROS_PACKAGE_PATH=$repo_root/turtlebot/src/turtlebot3_simulations:${ROS_PACKAGE_PATH}

if first_time_build; then
    # ros python packages
    pip uninstall em
    pip install empy catkin-pkg rospkg defusedxml
    # other necessary packages
    pip install numpy
    if ubuntu_version_equal 20.04; then
        sudo apt-get install ros-noetic-turtlebot3-msgs
        sudo apt-get install ros-noetic-turtlebot3
    else
        echo -e "Unable to install required turtlebot packages due to incompatible Ubuntu version."
        exit 1
    fi
fi

# obtain code; first try to do submodule update;
# if doesn't work (only the first time), then add the submodule
if [ ! -e "turtlebot/src/turtlebot3_simulations/LICENSE" ]; then
    git submodule update --init --recursive
fi

if [ ! -d "turtlebot/src/turtlebot3_simulations" ]; then
    # Follow the instructions here: https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/
    cd turtlebot/src/
    git submodule add https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
    cd ../..
fi

# catkin make and end.
if first_time_build; then
    build_turtlebot
else
    echo -e "If you want to build the turtlebot project, run 'build_turtlebot'"
fi
cd $repo_root
