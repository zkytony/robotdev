#!/usr/bin/env python

import cv2
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from rbd_spot_robot.utils import ros_utils

ALL_SIDES = ["frontleft", "frontright", "left", "right", "back"]

class SnapShotter:
    """
    Saves the first message received from all cameras.
    Timestamp will be in filename. Synchronize the image
    and camera info.
    """
    def __init__(self):
        self._saved = {}
        for side in ALL_SIDES:
            fisheye_sub = message_filters.Subscriber(
                f"/spot/camera/{side}/image", Image)
            fisheye_caminfo_sub = message_filters.Subscriber(
                f"/spot/camera/{side}/camera_info", CameraInfo)
            fisheye_ts = message_filters.TimeSynchronizer(
                [fisheye_sub, fisheye_caminfo_sub], 10)
            fisheye_ts.registerCallback(self._fisheye_callback)

            depth_sub = message_filters.Subscriber(
                f"/spot/depth/{side}/image", Image)
            depth_caminfo_sub = message_filters.Subscriber(
                f"/spot/depth/{side}/camera_info", CameraInfo)
            depth_ts = message_filters.TimeSynchronizer(
                [depth_sub, depth_caminfo_sub], 10)
            depth_ts.registerCallback(self._depth_callback)

            self._saved[side] = {
                "fisheye": False,
                "depth": False
            }

    @property
    def done(self):
        return all(self._saved[s]['fisheye'] and self._saved[s]['depth']
                   for s in ALL_SIDES)

    def _fisheye_callback(self, fisheye_img, caminfo):
        side = caminfo.header.frame_id.split("_")[0]
        if side not in self._saved:
            raise ValueError(f"Unexpected side: {side}")
        if not self._saved[side]['fisheye']:
            self._saved[side]['fisheye'] = True
            secs = caminfo.header.stamp.secs
            nsecs = caminfo.header.stamp.nsecs
            timestamp = f"{secs}.{nsecs}"
            img = ros_utils.convert(fisheye_img)
            print(f"Saving fisheye image from {side}")
            cv2.imwrite(f"fisheye_{timestamp}_{side}.png", img)

    def _depth_callback(self, depth_img, caminfo):
        side = caminfo.header.frame_id.split("_")[0]
        if side not in self._saved:
            raise ValueError(f"Unexpected side: {side}")
        if not self._saved[side]['depth']:
            self._saved[side]['depth'] = True
            secs = caminfo.header.stamp.secs
            nsecs = caminfo.header.stamp.nsecs
            timestamp = f"{secs}.{nsecs}"
            img = ros_utils.convert(depth_img)
            print(f"Saving depth image from {side}")
            cv2.imwrite(f"depth_{timestamp}_{side}.png", img)

if __name__ == "__main__":
    rospy.init_node("test_node")
    ss = SnapShotter()
    rate = rospy.Rate(10)
    while not ss.done:
        rate.sleep()
