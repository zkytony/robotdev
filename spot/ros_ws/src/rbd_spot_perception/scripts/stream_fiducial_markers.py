#!/usr/bin/env python
#
# Stream fiducial markers as TF transforms
#
# Note that in order to obtain necessary TF frames, you
# need to stream images from body cameras.

import sys
import time
import roslib
import rospy

from tf2_ros import TransformBroadcaster
import geometry_msgs

import rbd_spot

def main():
    """An example using the API to list and get specific objects."""
    rospy.init_node('spot_fiducial_marker_broadcaster')
    conn = rbd_spot.SpotSDKConn(sdk_name="StreamFiducialClient")
    world_object_client = rbd_spot.fiducial.create_client(conn)
    br = TransformBroadcaster()
    while not rospy.is_shutdown():
        fiducials_result, used_time = rbd_spot.fiducial.detectFiducials(world_object_client)
        print("detectFiducials took {:.3f}s".format(used_time))
        print("Currently detected fiducials: ",
              [fiducial.name for fiducial in fiducials_result])
        rbd_spot.fiducial.ros_broadcast_fiducials_tf(br, conn, fiducials_result)

if __name__ == '__main__':
    main()
