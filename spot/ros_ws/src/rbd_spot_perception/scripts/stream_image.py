#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py list

import time
import argparse

from rbd_spot_robot import SpotSDKConn
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request
import spot_driver.ros_helpers

import pandas as pd
from pprint import pprint

import rospy
import sensor_msgs


def extract_source_names(sources_result):
    """
    Args:
        sources_result: response of ListImageSources service.
            it is a set of ImageSource data objects
    Returns:
        a DataFrame object with two columns: name, and type
    """
    # obtain enum name from enum value
    _name_func = image_pb2.ImageSource.ImageType.Name
    return pd.DataFrame(
        sorted([(source.name, _name_func(source.image_type))
                for source in sources_result],
               key=lambda tup: tup[1]),  # sort by image type
        columns=["name", "type"])


def check_sources_valid(sources, sources_df):
    """
    Args:
        sources (list of str): List of source names
        sources_df (DataFrame): dataframe obtained as a result of extract_source_names
    Returns:
        tuple (bool, list); True if all sources are valid. False if not, and the
            list will contain invalid sources.
    """
    bad_sources = []
    ok = True
    for source_name in sources:
        names = sources_df['name']  # pandas.Series
        if len(names[names == source_name]) == 0:
            bad_sources.append(source_name)
            ok = False
    return ok, bad_sources


def stream_get_image(image_client, requests):
    """Iterator; uses the `image_client` to send
    get_image request with `requests`."""
    try:
        while True:
            _start_time = time.time()
            response = image_client.get_image(requests)
            _used_time = time.time() - _start_time
            yield response, _used_time
    except KeyboardInterrupt:
        print("Bye.")


def publish_image_response(publishers, response, conn):
    """
    Publishes images in response as ROS messages (sensor_msgs.Image)
    Args:
        publishers (dict): maps from source name to a dictionary,
            {"image": rospy.Publisher(Image), "camera_info": rospy.Publisher(CameraInfo)}
        response (GetImageResponse): The message returned by GetImage service.
    """
    # publish the image with local timestamp
    for image_response in response:
        local_time = conn.spot_time_to_local(
            image_response.shot.acquisition_time)
        image_msg, camera_info_msg =\
            spot_driver.ros_helpers._getImageMsg(image_response, local_time)
        source_name = image_response.source.name
        publishers[source_name]['image'].publish(image_msg)
        publishers[source_name]['camera_info'].publish(camera_info_msg)
        rospy.loginfo(f"Published image response from {source_name}")


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

    conn = SpotSDKConn(sdk_name="StreamImageClient")
    image_client = conn.ensure_client(ImageClient.default_service_name)

    # First, list image sources
    _start_time = time.time()
    sources_result = image_client.list_image_sources()  # blocking call
    _used_time = time.time() - _start_time
    print("ListImageSources took %.3fs" % _used_time)
    sources_df = extract_source_names(sources_result)
    print("Available image sources:")
    print(sources_df, "\n")

    if args.sources is None or len(args.sources) == 0:
        # nothing to do.
        return

    # Check if the sources are valid
    ok, bad_sources = check_sources_valid(args.sources, sources_df)
    if not ok:
        print(f"Invalid source name(s): {bad_sources}")
        return

    # We want to stream the image sources, publish as ROS message if necessary
    # First, parse format into protobuf enum value for Image.Format.
    fmt = None
    if args.format:
        fmt = image_pb2.Image.Format.Value(formats.index(args.format))

    # Create requests. We will send these requests continuously.
    requests = []
    for source in args.sources:
        req = build_image_request(source,
                                  quality_percent=args.quality,
                                  image_format=fmt)
        requests.append(req)

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

    # Stream the image through specified sources
    for response, time_taken in stream_get_image(image_client, requests):
        print(time_taken)
        if args.pub:
            publish_image_response(publishers, response, conn)


if __name__ == "__main__":
    main()
