#!/usr/bin/env python
# Note that this code relies on spot ros
#
# Takes pictures from specified camera sources and saves them.
#
# Usage:
#   ./take_snapshot.py <output_dir> -s [sources ... ]
#
# a source could be, e.g., hand_color_in_hand_depth_frame, frontleft_fisheye_image
# For example:
#    ./take_snapshot.py tmp/ -s hand_color_image frontleft_fisheye_image

import os
import cv2
import argparse
import time
import pytz
from datetime import datetime
import pandas as pd
from rbd_spot_robot.utils import ros_utils
import rbd_spot


def main():
    parser = argparse.ArgumentParser("Take pictures from given camera sources and save them."\
                                     "File names will of the format <spot>_<timestamp>_<source>.png")
    parser.add_argument("output_dir", type=str, help="output directory")
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

    image_requests = rbd_spot.image.build_image_requests(
        args.sources, quality=args.quality, fmt=args.format)

    # get image and save
    result, time_taken = rbd_spot.image.getImage(image_client, image_requests)
    print("Get image took %.3fs" % time_taken)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%z") + time.tzname[0]
    for image_response in result:
        img = rbd_spot.image.imgarray_from_response(image_response, conn)
        source = image_response.source.name
        print(f"Saving image from {source}")
        cv2.imwrite(os.path.join(args.output_dir, f"spot_{timestamp}_{source}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
