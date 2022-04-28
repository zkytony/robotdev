# Setup viam development environment with Python 3.9
#
# Note that when working with Viam, we do not assume
# access to any ROS stuff. It is just pure python plus
# Viam's Python SDK.
if [[ ! $PWD = *viam ]]; then
    echo "You must be in the spot/viam directory."
    return 1
else
    . "../../tools.sh"
fi
venv_path="../venv"
# create a dedicated virtualenv for spot workspace
if [ ! -d "${venv_path}/spot39" ]; then
    echo -e "Creating virtualenv for python 3.9..."
    python3.9 -m venv ${venv_path}/spot39
fi

source ${venv_path}/spot39/bin/activate
export PYTHONPATH=""
# TODO: For now, we are relying on some rbd_spot_* packages
# that involve importing 'rospy'. So must add ROS packages to path.
source ../ros_ws/devel/setup.bash

if first_time_setup ./; then
    # install boston dynamics
    pip install bosdyn-client==3.1.0
    pip install bosdyn-mission==3.1.0
    pip install bosdyn-api==3.1.0
    pip install bosdyn-core==3.1.0
    pip install bosdyn-choreography-client==3.1.0

    # other useful packages
    pip install numpy
    pip install pydot
    pip install graphviz
    pip install opencv-python
    pip install pandas
    pip install open3d
    echo "SPOT SETUP DONE." >> .DONE_SETUP

    # Install viam
    ACCESS_TOKEN="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"
    pip install git+https://$ACCESS_TOKEN@github.com/viamrobotics/python-sdk.git

    # TODO: some ros packages to keep things running for now
    pip install rospkg
fi
