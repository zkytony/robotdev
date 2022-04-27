#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py


import time
import argparse

import rospy
from rbd_spot_robot.utils import ros_utils
from sensor_msgs.msg import Image

from google.protobuf.json_format import MessageToDict, ParseDict
import numpy as np

import rbd_spot
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from bosdyn.api import image_pb2

from utils.MaskRCNN.infr_api import get_mrcnn_model, visualize

def make_callback():
    model = get_mrcnn_model()
    publisher = rospy.Publisher("/spot/segment/image", Image, queue_size=1)
    def callback(msg):
        fig = Figure()
        canvas = FigureCanvas(fig)
        ax = fig.gca()

        ax.axis('off')

        canvas.draw()       # draw the canvas, cache the renderer

        img = ros_utils.convert(msg)
        
        mask_res = model.detect([img], verbose=0)

        visualize(img, mask_res[0], ax=ax)
        canvas.draw()       # draw the canvas, cache the renderer
        result_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        result_bytes = result_image.tobytes()
        msg.data = result_bytes

        publisher.publish(msg)

    return callback


def main():
    rospy.init_node('segment')
    sub = rospy.Subscriber("/spot/stream_image/hand_color_image/image", Image, make_callback(), queue_size=1, buff_size=2**23)
    rospy.spin()

if __name__ == "__main__":
    main()
