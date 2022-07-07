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


def _fiducial_pose_to_tf(position, rotation, parent_frame, fiducial_name):
    # publish body pose transform
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    # We are publishing transform of the body with respect to the map frame
    #    map_frame->T->base_frame
    t.header.frame_id = parent_frame
    t.child_frame_id = fiducial_name
    t.transform.translation = geometry_msgs.msg.Vector3(
        x=position.x, y=position.y, z=position.z)
    t.transform.rotation = geometry_msgs.msg.Quaternion(
        x=rotation.x, y=rotation.y, z=rotation.z, w=rotation.w)
    return t


def main():
    """An example using the API to list and get specific objects."""
    rospy.init_node('spot_fiducial_marker_broadcaster')
    conn = rbd_spot.SpotSDKConn(sdk_name="StreamFiducialClient")
    world_object_client = rbd_spot.fiducial.create_client(conn)
    br = TransformBroadcaster()
    while not rospy.is_shutdown():
        fiducials_result, used_time = rbd_spot.fiducial.detectFiducials(world_object_client)
        print("detectFiducials took {:.3f}s".format(used_time))
        fiducials = [fiducial.name for fiducial in fiducials_result]
        rbd_spot.fiducial.ros_broadcast_fiducials_tf
        for fiducial in fiducials_result:
            fiducial_number = fiducial.name.split("_")[-1]
            fiducial_name = "fiducial_"+str(fiducial_number)
            fiducials.append(fiducial_name)

            fiducial_transform = fiducial.transforms_snapshot.child_to_parent_edge_map[fiducial_name]
            parent_frame_name = fiducial_transform.parent_frame_name
            fiducial_position = fiducial_transform.parent_tform_child.position
            fiducial_rotation = fiducial_transform.parent_tform_child.rotation

            t = _fiducial_pose_to_tf(fiducial_position, fiducial_rotation, parent_frame_name, fiducial_name)
            br.sendTransform(t)
        print("Currently detected fiducials: ", fiducials)


if __name__ == '__main__':
    main()
