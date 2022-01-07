# Run this script by source setup_movo.bash
# Creates movo workspace.

# create the movo workspace directory
if [ ! -d "movo" ]; then
    mkdir -p movo/src
fi
# create a dedicated virtualenv for movo workspace
if [ ! -d "movo/venv/movo" ]; then
    cd movo/
    virtualenv -p python3 venv/movo
    cd ..
fi
cd movo
catkin_make
