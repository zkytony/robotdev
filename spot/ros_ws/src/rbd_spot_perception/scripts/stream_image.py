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
from bosdyn.client.image import ImageClient

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
    return pd.DataFrame([(source.name, _name_func(source.image_type))
                         for source in sources_result],
                        columns=["name", "type"])

def main():
    parser = argparse.ArgumentParser("stream image")
    parser.add_argument("sources", nargs="+", help="image sources; or 'list'")
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

if __name__ == "__main__":
    main()
