# This is a template for you to write a setup script for a new robot. Please
# fill in commands or values as necessary at all the TODO locations. Angle
# brackets '<..>' indicate a placeholder you should fill in.

# Note that this script is designed to be run every time you
# want to work on a robot platform. The first time you run
# this script, it will install necessary packages and run the
# build commands to build the development environment for the
# robot of interest. Subsequent runs will:
#
#   1. activate virtualenv
#   2. ask if you want to update submodules
#   3. ask if you want to rebuild (that is most likely re-run catkin_make)
#   4. run any setup script that should be run every time you
#      work on this robot.
#

# Run this script by source setup_<robot_name>.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
fi
repo_root=$PWD

# TODO: change
robot_name="<your robot name>"

# Always assume at the start of a function,
# or any if clause, the working directory is
# the root directory of the repository.

# TODO: finish this function. This is run
# every time you want to build the ROS workspace
# for this robot. A preliminary code block is placed,
# but is subject to your edits.
function build_<robot_name>_stack
{
    cd ${robot_name}  # required
    if catkin_make; then
        echo "${ROBOT_NAME} SETUP DONE." >> src/.DONE_SETUP
    else
        rm src/.DONE_SETUP
    fi
    cd $repo_root # required
}

# Returns true if this is the first time
# we build the ROS packages related to a robot
function first_time_build
{
    if [ ! -e "${robot_name}/src/.DONE_SETUP" ]; then
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

# update submodules (clone necessary stuff)
if confirm "Update git submodules?"; then
    # TODO: add submodule update commands
    git submodule update --init --recursive <path_to_submodule>
fi

# creates workspace for the robot
if [ ! -d "${robot_name}/src" ]; then
    mkdir -p ${robot_name}/src
fi
# create a dedicated virtualenv for <robot_name> workspace
if [ ! -d "${robot_name}/venv/${robot_name}" ]; then
    cd ${robot_name}/
    virtualenv -p python3 venv/${robot_name}
    cd ..
fi

# activate virtualenv; Note that this is the only
# functionality of this script if turtlebot has been setup
# before.
source ${robot_name}/venv/${robot_name}/bin/activate
export ROS_PACKAGE_PATH=$repo_root/${robot_name}/src/:${ROS_PACKAGE_PATH}

# Install necessary packages
if first_time_build; then
    pip uninstall em  # required
    pip install empy catkin-pkg rospkg defusedxml  # required
    pip install numpy  # required
    # TODO: install necessary packages
    if ubuntu_version_equal 20.04; then
        # TODO: ubuntu-specific installations
    else
        echo -e "Unable to install required ${robot_name} packages due to incompatible Ubuntu version."
        exit 1
    fi
    # TODO: more non-ubuntu-specific installations
fi

# TODO: add submodule. supply the path to submodule. Copy this
# codeblock if more submodules are needed
if [ ! -d "${robot_name}/src/<submodule repo name>" ]; then
    cd ${robot_name}/src
    git submodule add <submodule repo git path>
    cd ../..
fi

# Start building
if [ first_time_build ] || [ confirm "rebuild?" ] ; then
    # build the workspace
    eval "build_${robot_name}_stack"
else
    echo -e "If you want to build the ${robot_name} project, run 'build_${robot_name}_stack'"
fi

if [ -e ${robot_name}/src/.DONE_SETUP ]; then
    # TODO: run any setup script that should be run every time you work on this
    # robot. additional setup to run that is not build-related
fi

cd $repo_root
