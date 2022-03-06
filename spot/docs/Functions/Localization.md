# Localization

Because we use rtabmap for mapping, we can use it again
for localization:
```
MAP_NAME="<map_name>" roslaunch rbd_spot_perception dual_camera_localization.launch
```

### How it works
By default, rtabmap is in mapping mode.
To set in localization mode with a previously created map, you should set the memory not incremental (make sure that arguments don't contain `--delete_db_on_start` too!):
```xml
<launch>
<node name="rtabmap" pkg="rtabmap_ros" type="rtabmap" args="">
   <!-- LOCALIZATION MODE -->
   <param name="Mem/IncrementalMemory" type="string" value="false"/>
</node>
</launch>
```


## Troubleshooting

### Robot starts at old location & doesn't update
When I start rtabmap in localization mode (setting `Mem/IncrementalMemory` to false),
the robot can't update its location. Its initial location seems to be where it was
when the SLAM mapping stopped.

The following Github issue describes the same problem.
https://github.com/introlab/rtabmap_ros/issues/687


### Warning "...computeTransformationImpl()...Finding correspondences with ..."
If you see this message:
```
[ WARN] (2022-03-06 02:25:05.468) RegistrationVis.cpp:1152::computeTransformationImpl() Finding correspondences with the guess cannot be done with multiple cameras, global matching is done instead. Please set "Vis/CorGuessWinSize" to 0 to avoid this warning.
```

Ignore this - This warning actually means localization is working!
Localization works properly when you
use `dual_localization.launch` even though you will
see this warning. Do not change "Vis/CorGuessWinSize" to 0
because that breaks localization.



### Localization doesn't update; Robot float in air

It appears that sometimes (very often) when you start the driver, image streamer, and then rtabmap, the robot model floats in mid air instead of
grounded on the map. In the figure below, clearly, the reason is the transform from `odom` to `body` is not correct.
<img src="https://user-images.githubusercontent.com/7720184/156898718-7375ef2c-80a3-4c1a-9157-92f852a0bf4a.png" width="500px"/>

The TF tree shows that "odom->body" is published by `spot_ros` - it is not rtabmap's fault.
<img src="https://user-images.githubusercontent.com/7720184/156898698-62eb8994-02d4-410e-a896-c08f01b5c755.png" width="600px"/>

How to fix this? 

**This does not appear to be an actual problem; it is just odometry drift.** See [SpotROS](../SpotROS.md) section on TF for more details.
Basically, if this happens, you could:
1. restart the robot, so the odom frame will be _very close_ to body, and localization should happen rather quickly.
2. Drive the robot in a loop to force loop closure; Localization happens at loop closure!  Below is where I read this ([source](https://answers.ros.org/question/302694/rtabmap-localization/)):
    >RTAB-Map will still relocalize the robot on loop closure / global localization
