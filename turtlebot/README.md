This is the ROS workspace for turtlebot. It is
created as I go through the Turtlebot tutorial:
https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/

Note that there are many outdated tutorials (dating back to 2015).
The above is referenced by [this blog](https://automaticaddison.com/how-to-launch-the-turtlebot3-simulation-with-ros/)
written in 2020.

See [troubleshooting](troubleshooting.md) for overcoming
issues encountered while setting up this workspace.


## Build
1. We are using docker with Ubuntu 20.04 with ROS noetic. To
   use this environment, run
   ```
   source docker/build.noetic.sh
   ```
   to build the docker image. Then
   ```
   source docker/run.noetic.sh --gui
   ```
   to start a container of that image.

2. Run the following at the root directory of `robotdev`
   ```
   source setup_turtlebot.bash
   ```
   See output [here](build_output)


## Run
```
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_empty_world.launch
```
![image](https://user-images.githubusercontent.com/7720184/148667706-bfb70da8-eda3-4e48-861f-1a1e677da11f.png)

```
export TURTLEBOT3_MODEL=waffle_pi
roslaunch turtlebot3_gazebo turtlebot3_house.launch
```
![image](https://user-images.githubusercontent.com/7720184/148667849-e1ef07cd-986b-4127-9019-279c66eb7ff5.png)
