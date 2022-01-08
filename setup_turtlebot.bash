# Run this script by source setup_turtlebot.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
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
