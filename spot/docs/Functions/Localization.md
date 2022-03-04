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
