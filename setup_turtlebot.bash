# Run this script by source setup_turtlebot.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
fi
repo_root=$PWD

function build_turtlebot
{
    cd $repo_root/turtlebot
    if catkin_make; then
        # cmake build is successful. Mark
        echo "TURTLEBOT PC SETUP DONE." >> turtlebot3_simulations/DONE_SETUP
    else
        rm turtlebot3_simulations/DONE_SETUP
    fi
}

function first_time_build
{
    if [ ! -e "turtlebot/src/turtlebot3_simulations/DONE_SETUP" ]; then
        # has not successfully setup
        true && return
    else
        false
    fi
}

# use ros
if ! useros; then
    echo "Cannot use ROS. Abort."
    exit
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

if first_time_build; then
    pip uninstall em
    pip install empy
    pip install catkin-pkg
fi

# Follow the instructions here: https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/
if [ ! -d "turtlebot/src/turtlebot3_simulations" ]; then
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
