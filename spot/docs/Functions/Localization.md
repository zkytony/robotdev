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

Ignore this. Localization works properly when you
use `dual_localization.launch` even though you will
see this warning. Do not change "Vis/CorGuessWinSize" to 0
because that breaks localization.



### Localization doesn't update; Robot float in air

It appears that sometimes (very often) when you start the driver, image streamer, and then rtabmap, the robot model floats in mid air instead of
grounded on the map. In the figure below, clearly, the reason is the transform from `odom` to `body` is not correct.
![image](https://user-images.githubusercontent.com/7720184/156898718-7375ef2c-80a3-4c1a-9157-92f852a0bf4a.png)

The TF tree 
![image](https://user-images.githubusercontent.com/7720184/156898698-62eb8994-02d4-410e-a896-c08f01b5c755.png)

