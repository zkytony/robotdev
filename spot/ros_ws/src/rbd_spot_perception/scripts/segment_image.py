#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py

import time
import argparse

import rospy

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


def main():
    parser = argparse.ArgumentParser("stream image")
    parser.add_argument("-s", "--sources", nargs="+", help="image sources; or 'list'")
    parser.add_argument("-q", "--quality", type=int,
                        help="image quality [0-100]", default=75)
    formats = ["UNKNOWN", "JPEG", "RAW", "RLE"]
    parser.add_argument("-f", "--format", type=str, default="RAW",
                        help="format", choices=formats)
    parser.add_argument("-p", "--pub", action="store_true", help="publish as ROS messages")
    parser.add_argument("-t", "--timeout", type=float, help="time to keep streaming")
    args = parser.parse_args()

    conn = rbd_spot.SpotSDKConn(sdk_name="StreamImageClient")
    image_client = rbd_spot.image.create_client(conn)

    sources_result, _used_time = rbd_spot.image.listImageSources(image_client)
    print("ListImageSources took %.3fs" % _used_time)
    sources_df = pd.DataFrame(rbd_spot.image.sources_result_to_dict(sources_result))
    print("Available image sources:")
    print(sources_df, "\n")

    if args.sources is None or len(args.sources) == 0:
        # nothing to do.
        return

    # Check if the sources are valid
    ok, bad_sources = rbd_spot.image.check_sources_valid(args.sources, sources_result)
    if not ok:
        print(f"Invalid source name(s): {bad_sources}")
        return

    # Create publishers, in case we want to publish;
    # maps from source name to a publisher.
    if args.pub:
        rospy.init_node("stream_image")
        publishers = rbd_spot.image.ros_create_publishers(args.sources, name_space="stream_image")

    # We want to stream the image sources, publish as ROS message if necessary
    # First, build requests
    image_requests = rbd_spot.image.build_image_requests(
        args.sources, quality=args.quality, fmt=args.format)

    # Stream the image through specified sources
    _start_time = time.time()
    
    model = get_mrcnn_model()
    # fig, ax = plt.figure()

    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    ax.text(0.0,0.0,"Test", fontsize=45)
    ax.axis('off')

    canvas.draw()       # draw the canvas, cache the renderer

    image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')

    while True:
        try:
            result, time_taken = rbd_spot.image.getImage(image_client, image_requests)
            print(time_taken)
            #####
            img = np.fromstring(result[0].shot.image.data, dtype=np.uint8)
            rows = result[0].shot.image.rows
            cols = result[0].shot.image.cols
            img = np.reshape(img, (rows, cols, 3))

            # Run detection
            mask_res = model.detect([img], verbose=0)

            visualize(img, mask_res[0], ax=ax)
            canvas.draw()       # draw the canvas, cache the renderer
            result_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            result_bytes = result_image.tobytes()
            result[0].shot.image.data = result_bytes
            #####
            if args.pub:
                rbd_spot.image.ros_publish_image_result(conn, result, publishers)
            _used_time = time.time() - _start_time
            if args.timeout and _used_time > args.timeout:
                break
        finally:
            if args.pub and rospy.is_shutdown():
                sys.exit(1)

if __name__ == "__main__":
    main()
