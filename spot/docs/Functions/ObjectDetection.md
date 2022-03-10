# Arm Gripper Camera

To stream images from the gripper camera:
```
rosrun rbd_spot_perception stream_image.py -s hand_color_image hand_depth_in_hand_color_frame --pub
```
When connected to RLAB, streaming from these two image sources together can get to up to 3Hz.

Then, you can obtain color image and corresponding depth map easily:

<img src="https://user-images.githubusercontent.com/7720184/157586588-bab589e8-dbdc-4fbb-86ce-04929618cbe8.png" width="550px"/>

Then you can easily visualize the depth cloud in RVIZ:

<img src="https://user-images.githubusercontent.com/7720184/157586653-afbd7007-01cc-4a75-8e7a-61d2d8034b6b.png" width="550px"/>
