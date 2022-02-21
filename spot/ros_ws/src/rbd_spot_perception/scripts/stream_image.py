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

import pandas as pd
from pprint import pprint


def image_callback():
    pass

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
    return pd.DataFrame(sorted([(source.name, _name_func(source.image_type))
                                for source in sources_result],
                               key=lambda tup: tup[1]),  # sort by image type
                               columns=["name", "type"])

def stream_get_image(image_client, requests):
    try:
        while True:
            _start_time = time.time()
            response = image_client.get_image(requests)
            _used_time = time.time() - _start_time
            yield response, _used_time
    except KeyboardInterrupt:
        print("Bye.")

def main():
    parser = argparse.ArgumentParser("stream image")
    parser.add_argument("sources", nargs="+", help="image sources; or 'list'")
    parser.add_argument("-q", "--quality", type=int, help="image quality [0-100]", default=75)
    formats = ["UNKNOWN", "JPEG", "RAW", "RLE"]
    parser.add_argument("-f", "--format", type=str, help="format", choices=formats)
    args = parser.parse_args()

    conn = SpotSDKConn(sdk_name="StreamImageClient")
    image_client = conn.ensure_client(ImageClient.default_service_name)

    sources = args.sources
    if len(sources) == 1 and sources[0] == "list":
        # list sources; blocking call
        _start_time = time.time()
        sources_result = image_client.list_image_sources()
        _used_time = time.time() - _start_time
        print("ListImageSources took %.3fs" % _used_time)
        df = extract_source_names(sources_result)
        print(df)
    else:
        requests = []
        for source in sources:
            fmt = None
            if args.format:
                fmt = image_pb2.Image.Format.Value(formats.index(args.format))
            req = build_image_request(source, quality_percent=args.quality, image_format=fmt)
            requests.append(req)

        # Stream the image through specified sources
        for response, time_taken in stream_get_image(image_client, requests):
            print(time_taken)


if __name__ == "__main__":
    main()
