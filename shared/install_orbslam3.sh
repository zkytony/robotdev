# Install ORB SLAM3
# https://github.com/UZ-SLAMLab/ORB_SLAM3
# run this script from repository root
# 02/23/2022: tested on Ubuntu 20.04
if [[ ! $PWD = *robotdev ]]; then
    echo "You must be in the root directory of the robotdev repository."
    return 1
else
    . "./tools.sh"
fi
repo_root=$PWD

if [ ! -d "thirdparty/ORB_SLAM3" ]; then
    cd thirdparty
    # clone
    git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git ORB_SLAM3
    cd ..
fi

if [ ! -d "thirdparty/ORB_SLAM3/build" ]; then
    # build
    # *Note*: you may get an error regarding OpenCV version 4.4 not found.
    # In my case, my system has installed OpenCV 4.2. I would like to not
    # change that. This github issue https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/456
    # says you could just edit the file ORB_SLAM3/CMakeLists.txt and change
    # the OpenCV version to 4.2.
    #
    # *Note*: the second build problem I encountered is "Pangolin" not found.
    # I need to install Pangolin according to the docs on its github:
    # https://github.com/stevenlovegrove/Pangolin
    # If you need to install Pangolin, run 'source install_pangolin.sh'
    #
    # *Note*: after successfully building Pangolin (cmake finishes 100%),
    # when I rerun ./build.sh inside ORB_SLAM3, I get a bunch of C++ error
    # messages. This thread https://github.com/UZ-SLAMLab/ORB_SLAM3/issues/387
    # says that you need to change the C++ version to C++14 in CMakeLists.txt
    # The fix is to run 'sed -i 's/++11/++14/g' CMakeLists.txt' and then
    # rerun ./build.sh.
    # The compilation time for ORB_SLAM3 is quite long.
    cd thirdparty/ORB_SLAM3
    chmod a+x build.sh
    ./build.sh
fi
