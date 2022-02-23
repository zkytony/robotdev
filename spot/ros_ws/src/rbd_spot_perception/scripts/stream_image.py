#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py

import time
import argparse

import rospy
import sensor_msgs

import rbd_spot
import pandas as pd


def main():
    parser = argparse.ArgumentParser("stream image")
    parser.add_argument("-s", "--sources", nargs="+", help="image sources; or 'list'")
    parser.add_argument("-q", "--quality", type=int,
                        help="image quality [0-100]", default=75)
    formats = ["UNKNOWN", "JPEG", "RAW", "RLE"]
    parser.add_argument("-f", "--format", type=str,
                        help="format", choices=formats)
    parser.add_argument("-p", "--pub", action="store_true", help="publish as ROS messages")
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
        publishers = {}
        for source in args.sources:
            publishers[source] = {
                "image": rospy.Publisher(
                    f"/spot/stream_image/{source}/image",
                    sensor_msgs.msg.Image, queue_size=10),
                "camera_info": rospy.Publisher(
                    f"/spot/stream_image/{source}/camera_info",
                    sensor_msgs.msg.CameraInfo, queue_size=10)
            }

    # We want to stream the image sources, publish as ROS message if necessary
    # First, build requests
    image_requests = rbd_spot.image.build_image_requests(
        args.sources, quality=args.quality, fmt=args.format)

    # Stream the image through specified sources
    try:
        for result, time_taken in rbd_spot.image.getImageStream(image_client, image_requests):
            print(time_taken)
            if args.pub:
                rbd_spot.image.ros_publish_image_result(conn, result, publishers)

    except KeyboardInterrupt:
        print("Bye.")

if __name__ == "__main__":
    main()
