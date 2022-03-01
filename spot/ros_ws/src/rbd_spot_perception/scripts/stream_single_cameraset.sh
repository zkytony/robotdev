#!/bin/bash
rosrun rbd_spot_perception stream_image.py -s\
       $1_fisheye_image\
       $1_depth_in_visual_frame\
       --pub
