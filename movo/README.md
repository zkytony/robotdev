This is the ROS workspace for movo. **Note that MOVO uses ROS kinetic on Ubuntu 16.04**
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
   This script will create a docker image using the [Dockerfile](../Dockerfile.kinetic) provided. If you are on a computer with NVidia GPU,
   ```
   source docker/build.kinetic.sh --nvidia
   ```

   After the docker image is created, start a container by running the following (still from the root of the repository)
   ```
   source docker/run.kinetic.sh
   ```
   This will drop you into a bash shell. You are, in fact, in an Ubuntu 16.04 environment, with necessary ROS-related software packages installed. If you are on a computer with NVidia GPU,
   ```
   source docker/run.kinetic.sh --nvidia
   ```
   
   **Caveat (09/21/2022):** If you use `--nvidia` with the kinetic docker image, RVIZ may fail to start up with error "Unable to create a suitable GLXContext in GLXContext::GLXContext". To be safe, use the non-nvidia image. You can still connect to the same ROS network (as ROS master is run on MOVO).

3. To build MOVO source code, go to the repository's root and run:
   ```
   source setup_movo.bash
   ```
   Equivalently, you can run `catkin_make` while inside `robotdev/movo`, that is, the workspace of MOVO. Note that the `catkin_make` might not directly succeed because additional packages, such as libfreenect2, must be installed. The bash script `setup_movo.bash` will take care of that.

   Once the `catkin_make` (a CMake process) finishes the build successfully at 100%, you are good to go.


4. Make sure that movo's IP addresses are in your `/etc/hosts` file. Otherwise, you may get the issue where you can do `rostopic list` but you cannot do `rostopic echo tf` (no message is received). In my computer, I have my `/etc/hosts` file like this:
   ```
    127.0.0.1       localhost
    127.0.1.1       zephyr

    # The following lines are desirable for IPv6 capable hosts
    ::1     ip6-localhost ip6-loopback
    fe00::0 ip6-localnet
    ff00::0 ip6-mcastprefix
    ff02::1 ip6-allnodes
    ff02::2 ip6-allrouters

    # for movo
    10.66.171.2 movo1
    10.66.171.1 movo2
    138.16.161.17 movo
   ```


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

4. Make sure that the SSH public key of your container is added
   to the `.ssh/authorized_keys` file on both MOVO1 and MOVO2.

    **IMPORTANT** When you SSH into MOVO1 and MOVO2 for **the first time**
    from within the container, please run the following command
    ```
    ssh movo@movo1 -oHostKeyAlgorithms='ssh-rsa'
    ssh movo@movo2 -oHostKeyAlgorithms='ssh-rsa'
    ```
    If you don't provide the option, you will run into an issue when starting a node on MOVO1 from your remote PC, due to a ROS-specific issue. See [this ROS Ask answer](https://answers.ros.org/question/244060/roslaunch-ssh-known_host-errors-cannot-launch-remote-nodes/?answer=244064#post-id-244064).

    If you happened to SSH into MOVO1 and MOVO2 without running the `ssh` command with the option as above, remove the corresponding entries from your `.ssh/known_hosts` file.

5. For convenience, you can add the following to your container's `.bashrc` file:
    ```bash
    alias domovo="cd ~/repo/robotdev/; source setup_movo.bash"
    ```
    
6. (NEW: 09/21/2022) It may happen that you cannot start a GUI program after `setup_movo.bash`, in particular, after sourcing the `devel/setup.bash`. In this case, open another terminal, and start a shell for the same container. Check if GUI programs work there. If so, run `echo $DISPLAY`. Compare that with the `$DISPLAY` variable in the shell where GUI programs fail. Likely, they are different. Now, manually change the `$DISPLAY` variable in the broken shell to the value in the working shell (e.g. I did `export DISPLAY=:1`). Then, GUI programs should work again.

## Usage and more

Refer to [docs](./docs) for detailed documentation. In particular:

 * [The Bootup Guide](./docs/Bootup.md)
 * [Networking](./docs/Networking.md)
 * [Assorted Usage Guide](./docs/MiscUsage.md)
