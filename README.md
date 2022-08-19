# robotdev

My repository for robot-related development.


## Structure

Each robot gets a corresponding directory.  The directory is typically (at least contains)
a designated ROS workspace for that robot. The robot-specific code is maintained in
submodules (i.e. other repositories). This repository serves as the
hub for all the robot-related development efforts.

To setup your Linux machine (Ubuntu 20.04) for a robot:
```
git submodule update --init --recursive  # optional
source setup_{robot_name}.bash
```


## Docker
By default, we assume you use ROS Noetic on Ubuntu 20.04.
If you are using a different version of ROS, please use
the Docker container for that ROS version.

**Note:** If your computer has NVidia GPU and you would like NVidia GPU support inside the docker container, please read [this wiki](https://github.com/zkytony/robotdev/wiki/Enabling-Nvidia-Support-in-Docker) instead of step 1 and 2.

Otherwise, as an example, to start the container for robotdev using ROS Kinetic:

1. Build the docker image. Replace 'kaiyu' to the username of yourself on your host machine
   ```
   source docker/build.kinetic.sh --hostuser=kaiyu
   ```

   If you run `docker images`, you should see:
     ```
     REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
     robotdev     kinetic   3293f13a9b25   7 seconds ago   1.13GB
     ```

   When using the `docker/build.noetic.sh`, you can provide a custom string as the suffix of the image tag name:
   ```
   source docker/build.kinetic.sh --tag-suffix=joy
   ```
   Then the tab of the image will become `noetic:joy`. (TODO for kinetic)

2. Run the docker container
   ```
   source docker/run.kinetic.sh
   ```
   To run it so that GUI (X11 forwarding) is supported:
   ```
   source docker/run.kinetic.sh --gui
   ```

   Note that if you exit from a container, you can restart it as follows.
   First list the containers that are stopped:
   ```
   docker ps --filter "status=exited"
   CONTAINER ID   IMAGE              COMMAND                  CREATED         STATUS                            PORTS     NAMES
   b4c0f9d589f0   robotdev:kinetic   "/ros_entrypoint.sh …"   4 minutes ago   Exited (130) About a minute ago             vigilant_keller
   ```
   Then, run `docker restart`. Note that it supports autocompletion by container names.
   ```
   docker restart vigilant_keller
   ```
   Then, run the following
   ```
   docker attach vigilant_keller
   ```
   You will regain the shell (may need to press enter once).

3. To have access to convenient docker functions such as `dsh`, `dls`, etc. run
   ```
   source docker_functions.sh
   ```
   Note that you should do this outside docker (of course you don't want to run docker inside docker!)
