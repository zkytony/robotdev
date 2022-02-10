#!/usr/bin/env python
# Copyright (c) 2021 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Example demonstrating capture of both visual and depth images and then overlaying them."""

import argparse
import sys

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.async_tasks import AsyncTasks
from bosdyn.api import image_pb2

from spot_driver.spot_wrapper import AsyncImageService, SpotWrapper

import rospy
import cv2
import numpy as np
import os
import time

from rbd_spot_robot.spot_sdk_client import SpotSDKClient

ALL_SIDES = ["frontleft", "frontright", "left", "right", "back"]

class DepthVisualPublisher(SpotSDKClient):
    def __init__(self, config={}):
        super(DepthVisualPublisher, self).__init__(name="depth_visual")


        image_sources = []
        for side in ALL_SIDES:
            image_sources.append(f"{side}_depth_in_visual_frame")
            image_sources.append(f"{side}_fisheye_image")

        self._image_requests = []

        for source in image_sources:
            self._image_requests.append(
                build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        print(self._image_requests)

        self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
        self._rates = config.get("rates", {})
        self._image_task = AsyncImageService(
            self._image_client,
            self._logger,
            5.0,
            self.DepthVisualCB,
            self._image_requests)

        self._async_tasks = AsyncTasks([self._image_task])


    @property
    def images(self):
        """Return latest proto from the _front_image_task"""
        return self._image_task.proto

    def DepthVisualCB(self, results):
        print("HHHHHHHHHHHHHHHHHELLO")

    def updateTasks(self):
        """Loop through all periodic tasks and update their data if needed."""
        try:
            self._async_tasks.update()
        except Exception as e:
            print(f"Update tasks failed with error: {str(e)}")


if __name__ == "__main__":
    rospy.init_node("HELLO")
    p = DepthVisualPublisher()
    while not rospy.is_shutdown():
        p.updateTasks()








# def main(argv):
#     # Parse args
#     parser = argparse.ArgumentParser()
#     bosdyn.client.util.add_common_arguments(parser)
#     parser.add_argument('--to-depth', help='Convert to the depth frame. Default is convert to visual.',
#                         action='store_true')
#     parser.add_argument('--camera', help='Camera to acquire image from.', default='frontleft',\
#                         choices=['frontleft', 'frontright', 'left', 'right', 'back',
#                         ])
#     parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
#                         action='store_true')
#     options = parser.parse_args(argv)

#     if options.to_depth:
#         sources = [ options.camera + '_depth', options.camera + '_visual_in_depth_frame' ]
#     else:
#         sources = [ options.camera + '_depth_in_visual_frame', options.camera + '_fisheye_image' ]


#     # Create robot object with an image client.
#     sdk = bosdyn.client.create_standard_sdk('image_depth_plus_visual')
#     robot = sdk.create_robot(options.hostname)
#     robot.authenticate(options.username, options.password)
#     image_client = robot.ensure_client(ImageClient.default_service_name)

#     # Capture and save images to disk
#     image_responses = image_client.get_image_from_sources(sources)

#     # Image responses are in the same order as the requests.
#     # Convert to opencv images.

#     if len(image_responses) < 2:
#         print('Error: failed to get images.')
#         return False

#     # Depth is a raw bytestream
#     cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
#     cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows,
#                                 image_responses[0].shot.image.cols)

#     # Visual is a JPEG
#     cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

#     # Convert the visual image from a single channel to RGB so we can add color
#     visual_rgb = cv2.cvtColor(cv2.cvtColor(cv_visual, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB) if len(cv_visual.shape) == 3 else cv2.cvtColor(cv_visual, cv2.COLOR_GRAY2RGB)


#     # Map depth ranges to color

#     # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
#     min_val = np.min(cv_depth)
#     max_val = np.max(cv_depth)
#     depth_range = max_val - min_val
#     depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
#     depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
#     depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)

#     # Add the two images together.
#     out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)

#     if options.auto_rotate:
#         if image_responses[0].source.name[0:5] == "front":
#             out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)

#         elif image_responses[0].source.name[0:5] == "right":
#             out = cv2.rotate(out, cv2.ROTATE_180)

#     # Write the image out.
#     filename = options.camera + ".jpg"
#     cv2.imwrite(filename, out)

#     return True


# if __name__ == "__main__":
#     if not main(sys.argv[1:]):
#         sys.exit(1)
