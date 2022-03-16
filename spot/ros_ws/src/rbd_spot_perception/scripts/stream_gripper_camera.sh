#!/bin/bash
rosrun rbd_spot_perception stream_image.py -s\
       hand_color_image\
       hand_depth_in_hand_color_frame\
       --pub
