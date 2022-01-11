# robotdev

My repository for robot-related development.


## Structure

Each robot gets a corresponding directory.  The directory is a designated ROS
workspace for that robot. The robot-specific code is maintained in
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

For example, to start the container for robotdev using ROS Kinetic:

1. Build the docker image. Replace 'kaiyu' to the username of yourself on your host machine.
   Update the 'password' as well.
    ```
    # assume you are at the repository's root
    docker build -f Dockerfile.kinetic\
        -t robotdev:kinetic\
        --build-arg hostuser=kaiyu\
        --rm\
        .
    ```
    The `--rm` option is for you to more conveniently rebuild the image.

    If you run `docker images`, you should see:
     ```
     REPOSITORY   TAG       IMAGE ID       CREATED         SIZE
     robotdev     kinetic   3293f13a9b25   7 seconds ago   1.13GB
     ```

2. Create and start the container. Use bind mount
   because we assume you will be using the same
   `robotdev` repository inside and outside the container.

   ```
   # assume you are at the repository's root
   docker run -it\
       --volume $(pwd):/home/kaiyu/repo/robotdev/\
       -e "TERM=xterm-256color"\
       --privileged\
       --network=host\
       robotdev:kinetic
   ```

3. To run RVIZ, you can start a new container with the following code:
   ```
   ```
