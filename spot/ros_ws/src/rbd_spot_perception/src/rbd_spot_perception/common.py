import sys
import rospy
import tf2_ros
import spot_driver.ros_helpers

# camera static transform broadcaster
CAMERA_STATIC_TF_BROADCASTER = None
CAMERA_STATIC_TRANSFORMS = []

def populate_camera_static_transforms(conn, object_w_tfs):
    """Check data received from one of the image tasks and use the transform
    snapshot to extract the camera frame transforms. This is the transforms from
    body->frontleft->frontleft_fisheye, for example. These transforms never
    change, but they may be calibrated slightly differently for each robot so we
    need to generate the transforms at runtime.

    Args:
        object_w_tfs: a proto with fields "transform_snapshot", "acquisition_time"

    note: adapted from spot_ros codebase
    """
    global CAMERA_STATIC_TF_BROADCASTER
    global CAMERA_STATIC_TRANSFORMS
    if CAMERA_STATIC_TF_BROADCASTER is None:
        CAMERA_STATIC_TF_BROADCASTER = tf2_ros.StaticTransformBroadcaster()
    tf_frames = _get_odom_tf_frames()
    excluded_frames = [tf_frames['tf_name_vision_odom'], tf_frames['tf_name_kinematic_odom'], "body"]
    for frame_name in object_w_tfs.transforms_snapshot.child_to_parent_edge_map:
        if frame_name in excluded_frames:
            continue
        parent_frame = object_w_tfs.transforms_snapshot.child_to_parent_edge_map.get(frame_name).parent_frame_name
        existing_transforms = [(transform.header.frame_id, transform.child_frame_id)
                               for transform in CAMERA_STATIC_TRANSFORMS]
        if (parent_frame, frame_name) in existing_transforms:
            # We already extracted this transform
            continue

        transform = object_w_tfs.transforms_snapshot.child_to_parent_edge_map.get(frame_name)
        local_time = conn.spot_time_to_local(object_w_tfs.acquisition_time)
        tf_time = rospy.Time(local_time.seconds, local_time.nanos)
        static_tf = spot_driver.ros_helpers.populateTransformStamped(
            tf_time, transform.parent_frame_name, frame_name,
            transform.parent_tform_child)
        CAMERA_STATIC_TRANSFORMS.append(static_tf)
        CAMERA_STATIC_TF_BROADCASTER.sendTransform(CAMERA_STATIC_TRANSFORMS)


def _get_odom_tf_frames():
    """
    note: adapted from spot_ros codebase
    """
    # get tf frames; Spot has 2 types of odometries: 'odom' and 'vision'
    mode_parent_odom_tf = rospy.get_param('~mode_parent_odom_tf', 'odom') # 'vision' or 'odom'
    tf_name_kinematic_odom = rospy.get_param('~tf_name_kinematic_odom', 'odom')
    tf_name_raw_kinematic = 'odom'
    tf_name_vision_odom = rospy.get_param('~tf_name_vision_odom', 'vision')
    tf_name_raw_vision = 'vision'
    if mode_parent_odom_tf != tf_name_raw_kinematic and mode_parent_odom_tf != tf_name_raw_vision:
        rospy.logerr('rosparam \'~mode_parent_odom_tf\' should be \'odom\' or \'vision\'.')
        sys.exit(1)
    return dict(tf_name_kinematic_odom=tf_name_kinematic_odom,
                tf_name_raw_kinematic=tf_name_raw_kinematic,
                tf_name_vision_odom=tf_name_vision_odom,
                tf_name_raw_vision=tf_name_raw_vision)
