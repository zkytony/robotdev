import time
import rospy
import sys

import tf2_ros
import geometry_msgs

from bosdyn.client.world_object import WorldObjectClient
from bosdyn.api import world_object_pb2
import spot_driver.ros_helpers

from .common import populate_camera_static_transforms

def create_client(conn):
    return conn.ensure_client(WorldObjectClient.default_service_name)

def detectFiducials(world_object_client):
    """Returns currently detected fiducial markers
    (called 'world objects' in Spot SDK's language)

    Returns:
        a list of WorldObject proto objects
    """
    request_fiducials = [world_object_pb2.WORLD_OBJECT_APRILTAG]
    _start_time = time.time()
    fiducials_result = world_object_client.list_world_objects(
        object_type=request_fiducials).world_objects
    _used_time = time.time() - _start_time
    return fiducials_result, _used_time

def ros_broadcast_fiducials_tf(br, conn, fiducials_result):
    """
    br (tf2_ros.TransformBroadcaster)
    fiducial_result (list of WorldObject proto objects)
    """
    def _fiducial_pose_to_tf(fiducial):
        fiducial_number = fiducial.name.split("_")[-1]
        fiducial_name = "fiducial_"+str(fiducial_number)
        fiducial_transform = fiducial.transforms_snapshot.child_to_parent_edge_map[fiducial_name]
        parent_frame_name = fiducial_transform.parent_frame_name
        position = fiducial_transform.parent_tform_child.position
        rotation = fiducial_transform.parent_tform_child.rotation

        # publish body pose transform
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        # We are publishing transform of the body with respect to the map frame
        #    map_frame->T->base_frame
        t.header.frame_id = parent_frame_name
        t.child_frame_id = fiducial_name
        t.transform.translation = geometry_msgs.msg.Vector3(
            x=position.x, y=position.y, z=position.z)
        t.transform.rotation = geometry_msgs.msg.Quaternion(
            x=rotation.x, y=rotation.y, z=rotation.z, w=rotation.w)
        return t

    for fiducial in fiducials_result:
        t = _fiducial_pose_to_tf(fiducial)
        br.sendTransform(t)
        populate_camera_static_transforms(conn, fiducial)
