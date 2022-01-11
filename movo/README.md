This is the ROS workspace for movo.
The main repository for the kinova stack is [here](https://github.com/zkytony/kinova-movo);
It is my fork of their official repo (following Yoon's fork).

See [troubleshooting](troubleshooting.md) for overcoming
issues encountered while setting up this workspace.

## Build

1. After cloning the robotdev repo, run
    ```
    git submodule update --init --recursive
    ```
    This will clone and update all the submodules,
    including ones relevant to using MOVO.

2. Then, you need to build (i.e. compile) the source code of
   the MOVO stack. Because MOVO is designed for ROS Kinetic, you need an Ubuntu 16.04 environment with ROS Kinetic installed in order to build MOVO's source code properly.

   To do that, use Docker. First, go to the root of the repository (e.g. `cd robotdev`). Then:
   ```
   source docker/build.kinetic.sh
   ```
   This script will create a docker image using the [Dockerfile](../Dockerfile.kinetic) provided.

   After the docker image is created, start a container by running the following (still from the root of the repository)
   ```
   source docker/run.kinetic.sh
   ```
   This will drop you into a bash shell. You are, in fact, in an Ubuntu 16.04 environment, with necessary ROS-related software packages installed.

3. To build MOVO source code, go to the repository's root and run:
   ```
   source setup_movo.bash
   ```
   Equivalently, you can run `catkin_make` while inside `robotdev/movo`, that is, the workspace of MOVO. Note that the `catkin_make` might not directly succeed because additional packages, such as libfreenect2, must be installed. The bash script `setup_movo.bash` will take care of that.

   Once the `catkin_make` (a CMake process) finishes the build successfully at 100%, you are good to go.


## Working on MOVO

1. Run the container:
   ```
   source docker/run.kinetic.sh --gui
   ```
   You should supply the `--gui` option if you want to run RVIZ, for example.

   It is recommended that you create one container and use this one every time you need to work on MOVO. This is not absolutely necessary because the container shares the directory of the `robotdev` repository with your host machine, so changes there will persist.

2. Inside the container, run
   ```
    source setup_movo.bash
   ```

   Check if you can ping movo2.

3. SSH into MOVO2. Then SSH into MOVO1 from within MOVO2, and then run
   ```
   sudo ifconfig wlan0 down
   ```
   Check if you can ping movo1.


## Usage and more

Refer to [docs](./docs) for detailed documentation. In particular:

 * [The Bootup Guide](./docs/Bootup.md)
 * [Networking](./docs/Networking.md)
 * [Assorted Usage Guide](./docs/MiscUsage.md)
