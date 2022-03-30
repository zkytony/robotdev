#####################################
# NO LONGER WORKS! BECAUSE
# I HAVE CHANGED THE INTERFACE
# OF SpotSDKClient. The logic
# of projection still works of course
#####################################
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
from spot_driver.ros_helpers import getImageMsg, populateTransformStamped

import rospy
import tf2_ros
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header

import cv2
import numpy as np
import os
import time
import struct
from tqdm import tqdm


def get_intrinsics(P):
    return dict(fx=P[0],
                fy=P[5],
                cx=P[2],
                cy=P[6])


def make_cloud(depth_visual, fisheye, caminfo):
    """
    Args:
        depth_visual (sensor_msgs.Image)
        fisheye (sensor_msgs.Image)
        caminfo (CameraInfo)

    Note:
        depth_visual should have the same camera info as fisheye;
        They also have the same dimension
    """
    points = []
    h, w = depth_visual.height, depth_visual.width
    for i in range(len(depth_visual.data)):
        if i % 2 == 1:
            continue

        u = (i // 2) % w
        v = (i // 2) / w

        # depth is represented as 16-bit unsigned integer.
        # because sensor_msg.Image's data field is 8-bits,
        # we need to get two bytes to create a 16-bit integer.
        # Credit: Jasmine
        b1 = depth_visual.data[i]
        b2 = depth_visual.data[i+1]

        if depth_visual.is_bigendian:
            depth = b1*256 + b2
        else:
            depth = b1 + b2*256

        grayscale = fisheye.data[i//2]
        I = get_intrinsics(caminfo.P)
        z = depth / 1000.0
        x = (u - I['cx']) * z / I['fx']
        y = (v - I['cy']) * z / I['fy']

        mono_rgb = struct.unpack('I', struct.pack('BBBB', grayscale, grayscale, grayscale, 255))[0]
        pt = [x, y, z, mono_rgb]
        points.append(pt)

    return points


ALL_SIDES = {"frontleft", "frontright", "left", "right", "back"}

class DepthVisualPublisher():
    def __init__(self, camera, conn,
                 max_pub_rate=5, max_query_rate=10):
        """
        Args:
            camera (str): which camera you want to receive point cloud.
                 Possible values: "frontleft", "frontright", "left", "right", "back"
            max_pub_rate (float): Maximum rate (Hz) of publishing point clouds.
                 Most likely unable to achieve this rate if it is bigger than 2.
            max_query_rate (float): Maximum rate (Hz) to query Spot SDK service fo
                 the images.
        """
        self.conn = conn

        # NOTE: THE FOLLOWING is BORROWED from spot_driver/spot_ros.py
        self.camera_static_transform_broadcaster = tf2_ros.StaticTransformBroadcaster()
        # Static transform broadcaster is super simple and just a latched publisher. Every time we add a new static
        # transform we must republish all static transforms from this source, otherwise the tree will be incomplete.
        # We keep a list of all the static transforms we already have so they can be republished, and so we can check
        # which ones we already have
        self.camera_static_transforms = []

        # Spot has 2 types of odometries: 'odom' and 'vision'
        # The former one is kinematic odometry and the second one is a combined odometry of vision and kinematics
        # These params enables to change which odometry frame is a parent of body frame and to change tf names of each odometry frames.
        self.mode_parent_odom_tf = rospy.get_param('~mode_parent_odom_tf', 'odom') # 'vision' or 'odom'
        self.tf_name_kinematic_odom = rospy.get_param('~tf_name_kinematic_odom', 'odom')
        self.tf_name_raw_kinematic = 'odom'
        self.tf_name_vision_odom = rospy.get_param('~tf_name_vision_odom', 'vision')
        self.tf_name_raw_vision = 'vision'
        if self.mode_parent_odom_tf != self.tf_name_raw_kinematic and self.mode_parent_odom_tf != self.tf_name_raw_vision:
            rospy.logerr('rosparam \'~mode_parent_odom_tf\' should be \'odom\' or \'vision\'.')
            return

        if camera not in ALL_SIDES:
            raise ValueError(f"Unrecognized camera {camera}")
        self.camera = camera

        image_sources = [f"{camera}_depth_in_visual_frame",
                         f"{camera}_fisheye_image"]

        self._image_requests = []
        for source in image_sources:
            self._image_requests.append(
                build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        self._image_client = self.conn.robot.ensure_client(ImageClient.default_service_name)
        self._max_query_rate = max_query_rate
        self._max_pub_rate = max_pub_rate
        self._image_task = AsyncImageService(
            self._image_client,
            self.conn.logger,
            self._max_query_rate,  # maximum rate
            self.DepthVisualCB,
            self._image_requests)

        self._async_tasks = AsyncTasks([self._image_task])
        self._pcpub = rospy.Publisher(f"/spot/depthvisual/{camera}/point_cloud2",
                                      PointCloud2, queue_size=5)

        # REQUIRED BY getImageMsg (bad design due to Spot ROS utility function)
        self.spot_wrapper = SpotWrapper(self.conn.username,
                                        self.conn.password,
                                        self.conn.hostname,
                                        self.conn.logger,
                                        estop_timeout=rospy.get_param("~estop_timeout", 9.0),
                                        rates=rospy.get_param("~rates", {}),
                                        callbacks={},
                                        cameras_used=[])

    @property
    def images(self):
        """Return latest proto from the _front_image_task"""
        return self._image_task.proto

    def DepthVisualCB(self, results):
        data = self.images
        if data:
            try:
                all_points = []
                depth_visual_msg, caminfo = getImageMsg(data[0], self.spot_wrapper)
                fisheye_msg, caminfo = getImageMsg(data[1], self.spot_wrapper)
                points = make_cloud(depth_visual_msg, fisheye_msg, caminfo)
                fields = [PointField('x', 0, PointField.FLOAT32, 1),
                          PointField('y', 4, PointField.FLOAT32, 1),
                          PointField('z', 8, PointField.FLOAT32, 1),
                          PointField('rgb', 12, PointField.UINT32, 1)]
                header = Header()
                header.stamp = caminfo.header.stamp
                header.frame_id = caminfo.header.frame_id
                pc2 = point_cloud2.create_cloud(header, fields, points)
                self.populate_camera_static_transforms(data[0], self.spot_wrapper)
                self._pcpub.publish(pc2)
                print(f"publish {self.camera}")
            except Exception as e:
                self.conn.logger.error("Error during callback: {}".format(e))

    def updateTasks(self):
        """Loop through all periodic tasks and update their data if needed."""
        try:
            self._async_tasks.update()
        except Exception as e:
            print(f"Update tasks failed with error: {str(e)}")


    def populate_camera_static_transforms(self, image_data, spot_wrapper):
        """Check data received from one of the image tasks and use the transform snapshot to extract the camera frame
        transforms. This is the transforms from body->frontleft->frontleft_fisheye, for example. These transforms
        never change, but they may be calibrated slightly differently for each robot so we need to generate the
        transforms at runtime.

        Args:
        image_data: Image protobuf data from the wrapper
        """
        # We exclude the odometry frames from static transforms since they are not static. We can ignore the body
        # frame because it is a child of odom or vision depending on the mode_parent_odom_tf, and will be published
        # by the non-static transform publishing that is done by the state callback
        excluded_frames = [self.tf_name_vision_odom, self.tf_name_kinematic_odom, "body"]
        for frame_name in image_data.shot.transforms_snapshot.child_to_parent_edge_map:
            if frame_name in excluded_frames:
                continue
            parent_frame = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name).parent_frame_name
            existing_transforms = [(transform.header.frame_id, transform.child_frame_id) for transform in self.camera_static_transforms]
            if (parent_frame, frame_name) in existing_transforms:
                # We already extracted this transform
                continue

            transform = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name)
            local_time = spot_wrapper.robotToLocalTime(image_data.shot.acquisition_time)
            tf_time = rospy.Time(local_time.seconds, local_time.nanos)
            static_tf = populateTransformStamped(tf_time, transform.parent_frame_name, frame_name,
                                                 transform.parent_tform_child)
            self.camera_static_transforms.append(static_tf)
            self.camera_static_transform_broadcaster.sendTransform(self.camera_static_transforms)
