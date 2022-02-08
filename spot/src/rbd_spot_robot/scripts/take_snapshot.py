#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo

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
            self._fisheye_ts = message_filters.TimeSynchronizer(
                [fisheye_sub, fisheye_caminfo_sub], 10)
            self._fisheye_ts.registerCallback(self._fisheye_callback)

            depth_sub = message_filters.Subscriber(
                f"/spot/depth/{side}/image", Image)
            depth_caminfo_sub = message_filters.Subscriber(
                f"/spot/depth/{side}/camera_info", CameraInfo)
            self._depth_ts = message_filters.TimeSynchronizer(
                [depth_sub, depth_caminfo_sub], 10)
            self._depth_ts.registerCallback(self._depth_callback)

            self._saved[side] = {
                "fisheye": False,
                "depth": False
            }

    @property
    def done(self):
        rself._saved[s]['fisheye'] and self._saved[s]['depth']

    def _fisheye_callback(self, fisheye_img, caminfo):
        print("YOU RECEIVED FISHEYE IMAGE")
        import pdb; pdb.set_trace()

    def _depth_callback(self, depth_img, caminfo):
        print("YOU RECEIVED DEPTH IMAGE")
        import pdb; pdb.set_trace()


if __name__ == "__main__":
    rospy.init_node("test_node")
    ss = SnapShotter()
    rate = rospy.Rate(10)
    while not ss.done:
        rate.sleep()
