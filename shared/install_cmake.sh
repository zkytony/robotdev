# This script installs cmake separately from
# your system cmake, without removing the
# system cmake. Reference:
# https://answers.ros.org/question/293119/how-can-i-updateremove-cmake-without-partially-deleting-my-ros-distribution/
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
    return 1
else
    . "./tools.sh"
fi
repo_root=$PWD

# CMake version
version=3.23.0-rc2

cd thirdparty
mkdir cmake && cd cmake
wget https://github.com/Kitware/CMake/releases/download/v$version/cmake-$version-linux-x86_64.tar.gz
tar xzvf cmake-$version-linux-x86_64.tar.gz
cd cmake-$version-linux-x86_64
export PATH=$repo_root/thirdparty/cmake/cmake-$version-linux-x86_64/bin:$PATH
