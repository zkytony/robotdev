# Arm Gripper Camera

To stream images from the gripper camera:
```
rosrun rbd_spot_perception stream_image.py -s hand_color_image hand_depth_in_hand_color_frame --pub
```
When connected to RLAB, streaming from these two image sources together can get to up to 3Hz.

Then, you can obtain color image and corresponding depth map easily:

