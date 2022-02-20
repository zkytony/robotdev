#!/usr/bin/env python
# Stream images through Spot
#
# Usage examples:
#
# rosrun rbd_spot_perception stream_image.py list

import argparse
from rbd_spot_robot import SpotSDKConn
import bosdyn.client as bdc
from pprint import pprint


def image_callback():
    pass

def extract_source_names(sources_result):
    """
    Args:
        sources_result (list of ListImageSources)
    """


def main():
    parser = argparse.ArgumentParser("stream image")
    parser.add_argument("sources", nargs="+", help="image sources; or 'list'")
    args = parser.parse_args()

    conn = SpotSDKConn(sdk_name="StreamImageClient")
    image_client = conn.ensure_client(bdc.image.ImageClient.default_service_name)

    sources = args.sources
    if len(sources) == 1 and sources[0] == "list":
        # list sources; blocking call
        print(image_client.list_image_sources())


if __name__ == "__main__":
    main()
