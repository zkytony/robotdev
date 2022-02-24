# Install Open3D from source
# https://github.com/isl-org/Open3D
# Note:
#  - you should activate the desired robot python virtualenv (e.g. spot)
#    before you run this script.
# Run this script by source setup_movo.bash
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
    return 1
else
    . "./tools.sh"
fi
repo_root=$PWD

if ! in_venv; then
    echo "You must activate the robot-specific virtualenv."
    return 1
fi

OPEN3D_INSTALL_PATH=$repo_root/thirdparty/Open3D/install

cd thirdparty
git clone https://github.com/isl-org/Open3D
cd Open3D

# Install dependencies
util/install_deps_ubuntu.sh

# build
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$OPEN3D_INSTALL_PATH
make -j$(nproc)
make install

# install python package
# from open3d docs https://github.com/isl-org/Open3D/blob/master/docs/arm.rst
pip install cmake
make install-pip-package
make python-package
make pip-package
python -c "import open3d"
