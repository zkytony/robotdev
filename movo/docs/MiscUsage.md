
# Misc Usage

## Networking
1. Connect to RLAB wifi. Then `ping movo2` should show:
```
$ ping movo2
PING MOVO2 (138.16.161.17) 56(84) bytes of data.
64 bytes from MOVO2 (138.16.161.17): icmp_seq=1 ttl=64 time=4.99 ms
64 bytes from MOVO2 (138.16.161.17): icmp_seq=2 ttl=64 time=61.6 ms
```
And `ping movo1` should show (assuming that movo1 is connected to RLAB itself as well):
```
$ ping movo1
PING movo1.rlab.cs.brown.edu (138.16.161.160) 56(84) bytes of data.
64 bytes from movo1.rlab.cs.brown.edu (138.16.161.160): icmp_seq=1 ttl=64 time=5.60 ms
64 bytes from movo1.rlab.cs.brown.edu (138.16.161.160): icmp_seq=2 ttl=64 time=4.78 ms
```

2. Make sure that on the local machine, `ROS_MASTER_URI=http://movo2:11311` and `ROS_IP=<local computer's 138.16.161.X address>`.

3. Make sure to run the following command to redirect traffic (i.e. redirect packets from your computer to the local network in MOVO):
```
sudo route add -net 10.66.171.0 netmask 255.255.255.0 gw 138.16.161.17 dev <network interface>
```
where `network interface` is the name of the network that has an 138.16.161.X IP address (RLAB).

## Bring up MOVO
1. First, on your PC, check if you can ping both movo1 and movo2 through the `10.66.171.X` IP addresses. If you cannot ping movo1, then SSH into movo2 and then ssh into movo1, and then run `sudo ifconfig wlan0 down` on movo1. This disables the WiFi network interface on movo1 such that it can only talk to movo2.

2. ssh into MOVO2. Run
```
roslaunch movo_bringup movo_system.launch
```
After the MOVO performs the dance, run the following command on your local computer
```
$ roswtf
```
This will make sure network works (i.e. topics can be received).

### Convenience
Copy and paste the following into your `~/.basrhc` for convenience
```bash
usemovo()
{
    sudo route add -net 10.66.171.0 netmask 255.255.255.0 gw 138.16.161.17 dev wlo1
    useros
    export ROS_HOSTNAME=$(hostname)
    export ROS_MASTER_URI="http://movo2:11311"
    export ROS_IP="138.16.161.191"
    echo -e "You computer has been configured. Now do:"
    echo -e "- ssh movo@movo2"
    echo -e "- ssh into movo1 from movo2 (run `ssh movo1`)"
    echo -e "- sudo ifconfig wlan0 down in movo1"
    echo -e "- go back to movo2, run:"
    echo -e "    roslaunch movo_bringup movo_system.launch"
}
```


## Using Joy stick:
1. `roslaunch movo_demos robot_assisted_teleop.launch`
2. `roslaunch movo_remote_teleop movo_remote_teleop.launch`
Check out [this wiki](https://sites.google.com/a/brown.edu/brown-robotics-wiki/robots/movo/movo-joystick) for button mapping.


### Note when using docker container

When you plug the joystick into your computer,
you should see `/dev/input/js0`. If you do not see this file on the file
system of your docker container, then you need to stop the container, commit it (saves the modified container state into a new image), and then run the new image (i.e. run from the new entry point). Now the `--privleged` tag should allow your new container to see the joystick. Note that you could use the same image name so that you could run the same bash script under `robotdev/docker`.

For example, let's say you were running a container for the image `robotdev:kinetic`. Now you just plugged in the joystick into your host computer.
First, stop the container:
```
docker container ls   # list active containers
docker stop <container>
```
Then, commit the container you just stopped
```
docker commit <container> robotdev:kinetic
```
Note that this essentially _overrides_ the previous `robotdev:kinetic` image.
You can immediately list the images by `docker images` and you should see this image being created a few seconds ago.

Now, you can just run the new image (with a new entry point) as usual:
```
source docekr/run.kinetic.sh --gui
```

Side tip: If you encounter an error message `Couldn't open joystick force feedback!`, you can ignore it, and the joystick should still function as expected ([reference](https://github.com/ros-drivers/joystick_drivers/issues/134#issuecomment-507338762))

## Mapping
1. Simply run
```
roslaunch movo_object_search mapping.launch
```
2. Then, in the RViz of `viz_object_search.launch` you will see the mapping happening.
3. When mapping is done, do
```
rosrun map_server map_saver -f <map_name>
```
in the `movo_demos/map` directory. Copy this map to `movo_object_search/maps` as well for convenience of creating topological maps later.

## Navigation
Just Localization
1. Run `roslaunch movo_object_search localization.launch map_file:=cit122` where "cit122" could be replaced by another map file name.

Move_base + localization

**Note** because of this [issue](https://answers.ros.org/question/244060/roslaunch-ssh-known_host-errors-cannot-launch-remote-nodes/), I cannot actually start the navigation on my computer. The `amcl`, `move_base` packages etc must be run on the MOVO1, which is why there is a `<machine>` tag in `movo_demos/map_nav.launch`; My computer cannot ROS-ssh into MOVO1 because of that linked issue. Hence:
1. ssh into MOVO2.
2. run `roslaunch movo_demos map_nav.launch map_file:=cit122`. The [reference](https://github.com/Kinovarobotics/kinova-movo/wiki/2.-How-Tos) on kinova repo is wrong!

## Sensors
### Point Cloud
Likely it is very slow to transmit point cloud from MOVO to local computer via WIFI. There is a file `./movo_7dof_moveit_config/config/sensors.yaml` which has some configuration about point cloud. Among these configurations, according to [this doc](http://docs.ros.org/indigo/api/pr2_moveit_tutorials/html/planning/src/doc/perception_configuration.html),
 - point_subsample:  Choose one of every point_subsample points.
 - max_range: Points further than this will not be used. On MOVO, this is set to just 2.0m!

Another file with the same name and content appears at `./movo_moveit_config/config/sensors.yaml`.

**But**, if you don't feel comfortable changing these settings, I found that Kinova offers `kinect2` bridge, according to [this doc](https://github.com/Kinovarobotics/kinova-movo/blob/kinetic-devel/movo_common/movo_third_party/iai_kinect2/kinect2_bridge/README.md). You can launch Kinect2 bridge by
```
roslaunch kinect2_bridge kinect2_bridge.launch
```
When I launched this file, there were a couple rounds of failure/retry messages, and eventually it seems to work, and can see these messages:
```
[ INFO] [1579995753.625271994]: [Kinect2Bridge::initDevice] Kinect2 devices found:
[ INFO] [1579995753.625300600]: [Kinect2Bridge::initDevice]   0: 012598365247 (selected)
[Info] [Freenect2DeviceImpl] opening...
[Info] [Freenect2DeviceImpl] transfer pool sizes rgb: 20*16384 ir: 60*8*33792
[Info] [Freenect2DeviceImpl] opened
[ INFO] [1579995753.724159270]: [Kinect2Bridge::initDevice] starting kinect2
[Info] [Freenect2DeviceImpl] starting...
[Info] [Freenect2DeviceImpl] submitting rgb transfers...
[Info] [Freenect2DeviceImpl] submitting depth transfers...
[Info] [Freenect2DeviceImpl] started
[ INFO] [1579995753.803961889]: [Kinect2Bridge::initDevice] device serial: 012598365247
[ INFO] [1579995753.804021685]: [Kinect2Bridge::initDevice] device firmware: 2.3.3913.0
```
There is a corresponding `/kinect2/sd/points` topic which will have `PointCloud2` messages at a lower resolution.
**AND THIS, SOLVED MY PROBLEM!**

**__NONE OF THE ABOVE WORKS!! ON THE REAL ROBOT!!__**

Use [depth_image_proc](http://wiki.ros.org/depth_image_proc) to convert depth image to point cloud xyz. Example launch file [here](https://gist.github.com/bhaskara/2400165). Use [image_transport](http://wiki.ros.org/image_transport), specifically the `republish` node to convert compressed image to decompressed image.
```
republish theora in:=camera/image raw out:=camera/image_decompressed
```
```
rosrun image_transport republish compressed in:=(in_base_topic) raw out:=(out_base_topic)
```


## Ros Tricks
### Save a message to a file
If you want to save a particular text message directly to a file you can do
```
rostopic echo -p /your_topic/goes/here > /path/to/your/file
```
you won't see any ouptut here, but in your file you'll see the message in a CSV format.

### Echo a single message
```
rostopic echo -n1 <other options> /topic
```
### Ros <param> is different from "args" in <node> tag.
The following two are completely different:
```
<node name="republish" type="republish" pkg="image_transport" output="screen" args="compressed in:=/axis_camera raw out:=/axis_camera/image_raw" />
```
VS
```
  <node pkg="image_transport" type="republish" name="depth_decompressor"
        output="screen" >
    <param name="in" value="$(arg camera_compressed_image_topic)"/>
    <param name="out_transport" value="raw"/>
    <param name="out" value="$(arg camera_image_topic"/>
  </node>
```
The first one works. This means `arg_name:=value` cannot be accessed in python code through `get_param(arg_name)` since it is not on the parameter server!

Use this:
```
<launch>
  <arg name="camera_info_topic" default="/movo_camera/hd/camera_info"/>
  <arg name="camera_compressed_image_topic" default="/movo_camera/qhd/image_depth_rect"/>
  <arg name="camera_image_topic" default="/image_transport/depth_image_raw"/>

  <!-- Start image_transport republisher and depth_image_proc -->
  <node pkg="image_transport" type="republish" name="depth_decompressor"
        output="screen"
        args="compressed in:=$(arg camera_compressed_image_topic) raw out:=$(arg camera_image_topic)">
  </node>

  <!-- Start the depth_image_proc -->
  <!-- <node pkg="nodelet" type="nodelet" name="depth_to_cloud"
       args="load depth_image_proc/point_cloud_xyz"
       output="screen">
       <remap from="camera_info" to="$(arg camera_info_topic)"/>
       <remap from="image_rect" to="$(arg camera_image_topic)"/>
       </node> -->
</launch>
```

## Troubleshooting
1. synchronization issue

I am facing this nasty synchronization issue that basically prevents the `depth_image_proc` node to publish anything.
![issue1](https://i.imgur.com/ix871ML.jpg)

A potential solution is described in this [ROS Ask thread](https://answers.ros.org/question/298821/tf-timeout-with-multiple-machines/). This involves configuring chrony and ntp. Indeed, when I run
```
ntpdate -q movo2
```
I get:
```
server 10.66.171.2, stratum 0, offset 0.000000, delay 0.00000
27 Jan 16:19:43 ntpdate[7006]: no server suitable for synchronization found
```
which is likely an indication of time not in sync between my machine and movo2. I keep getting `no server suitable for synchronization found` even I tried their instruction.

Anyway. I came across ntp configuration. Just some notes:
1. [You're supposed to create your own `/etc/ntp.conf` file](https://superuser.com/questions/856378/no-ntp-conf-file-after-compiling-ntp-on-linux) Yet Ifrah's computer doesn't have this file.
2. [The `stratum` configuration indicates how many hops away from a reference clock that its time is unreliable. And the default value of 10 is probably too high.](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/7/html/system_administrators_guide/sect-understanding_chrony_and-its_configuration)
3. [Comprehensive document on ntp configuration](https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/6/html/deployment_guide/s1-understanding_the_ntpd_configuration_file)
4. [A reference of using `*.ubuntu.pool.ntp.org` as server list in /etc/network`](https://help.ubuntu.com/lts/serverguide/NTP.html)
