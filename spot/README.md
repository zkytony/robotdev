# Spot

## Build

0. Run `git submodule update --init --recursive` to get the spot_ros repository.

1. Similar to MOVO, we use Docker for Spot too (but we are using
   Ubuntu 20.04 with ROS noetic). First, build the docker image:
   ```
   source docker/build.noetic.sh
   ```
   This command will run for a while. If it is successful,
   the process should end with a message "Successfully tagged robotdev:noetic".


   After you finish building the docker image, start a container
   by running the following (still from the root of the directory):
   ```
   source docker/run.noetic.sh --gui
   ```
   This will drop you into a bash shell. You should be able to run GUI applications too.

2. Now, setup the spot workspace:
   ```
   # at robotdev root directory:
   source setup_spot.bash
   ```
   This will build the ROS packages for Spot too.
   If you JUST want to build spot, don't run `catkin_make`. Instead,
   after sourcing `setup_spot.bash`, run:
   ```
   $ build_spot
   ```
   If you want to build specific package(s), run
   ```
   build_spot -DCATKIN_WHITELIST_PACKAGES="rbd_spot_robot"
   ```

   The build should be successful. If you encounter an error about
   a header file in `spot_ros` not found, just rebuild. If you encounter
   an error about "Killed signal terminated program cc1plus," that
   is [a sign that you ran out of memory](https://github.com/introlab/rtabmap_ros/issues/95#issuecomment-230366461).
   Just rebuild.


## Choreographer

Note that the `choreographer` is a symbolic link
to the executable of the Choreographer Linux
downloaded from [Boston Dynamics Support Center](https://support.bostondynamics.com/s/downloads).
The Choreographer version tested is 3.0.

Read [this documentation](https://support.bostondynamics.com/s/article/How-to-Install-Choreographer) for more info
about how to install the Choreographer.


## Troubleshooting

### Weird issue: rostopic echo doesn't work but rostopic hz works
If source ROS setup.bash, then rostopic echo doesn't work!
No idea why.

After investigation, the problem is:
```
source $repo_root/${SPOT_PATH}/devel/setup.bash
export PYTHONPATH="$repo_root/${SPOT_PATH}/venv/spot/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:${PYTHONPATH}"
```
In `PYTHONPATH`, `/usr/lib/python3/dist-packages` CANNOT appear before `${PYTHONPATH}`
which after sourcing the `devel/setup.bash` contains workspace-level python configurations;
That is supposed to overwrite the system's default which is in `/usr/lib/python3/dist-packages`.
Note that `/usr/lib/python3/dist-packages` is added only so that `PyKDL` can be imported (in order to resolve [this issue](https://answers.ros.org/question/380142/how-to-install-tf2_geometry_msgs-dependency-pykdl/?answer=395887#post-id-395887).)


### XX camera communication error
You may see this on the controller as one of the fault messages (from the top).
For example, "right camera communication error."
If that happens, then you will not be able to run GetImage gRPC for that camera.

Try to restart the robot.
