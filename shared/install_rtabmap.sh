# Install rtabmap.
# The code here references the Dockerfile for rtabmap:focal
# https://hub.docker.com/r/introlab3it/rtabmap/dockerfile
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
    return 1
else
    . "./tools.sh"
fi
repo_root=$PWD

cd thirdparty
git clone https://github.com/introlab/rtabmap.git
cd rtabmap/build
cmake ..
make -j4
sudo make install
