#!/bin/bash
rosrun rbd_spot_perception stream_image.py -s\
       frontleft_fisheye_image\
       frontleft_depth_in_visual_frame\
       frontright_fisheye_image\
       frontright_depth_in_visual_frame\
       --pub
