# Run this script by source setup_movo.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
else
    . "./tools.sh"
fi

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
# Download the kinova movo stack
if [ ! -d "movo/src/kinova-movo" ]; then
    cd movo/src
    git submodule add git@github.com:zkytony/kinova-movo.git
    cd ../..
fi

# Run the automatic installation script, if not already
if [ ! -e "movo/src/kinova-movo/DONE_SETUP" ]; then
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
    fi
fi

cd movo
catkin_make
