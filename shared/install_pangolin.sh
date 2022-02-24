# Install Pangolin
# https://github.com/stevenlovegrove/Pangolin
# run this script from repository root
#
# *NOTE*: Somehow during `cmake --build .`, the computer shutsdown at around
# 65%. I am not sure why. I had to then go to the build/ directory, and then
# manually run 'cmake ..'  and 'cmake --build .' again. That started the build
# process from around 65%, and it eventually finished successfully.
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
    return 1
else
    . "./tools.sh"
fi
repo_root=$PWD

cd thirdparty
git clone --recursive https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin

# Install dependencies (as described above, or your preferred method)
./scripts/install_prerequisites.sh recommended

# Configure and build
mkdir build && cd build
cmake ..
cmake --build .

# GIVEME THE PYTHON STUFF!!!! (Check the output to verify selected python version)
cmake --build . -t pypangolin_pip_install

# Run me some tests! (Requires Catch2 which must be manually installed on Ubuntu.)
ctest
