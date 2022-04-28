# Provides functions to interact with image services Spot SDK
# and connect that to ROS.

import time
import rospy
import sensor_msgs
import tf2_ros
import sys
import numpy as np

from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

import spot_driver.ros_helpers
from cv_bridge import CvBridge

# Note that if you don't specify image format (None) when
# sending GetImageRequest, the response will be in JPEG format.
IMAGE_FORMATS = ["UNKNOWN", "JPEG", "RAW", "RLE"]

# camera static transform broadcaster
CAMERA_STATIC_TF_BROADCASTER = None
CAMERA_STATIC_TRANSFORMS = []


def listImageSources(image_client):
    """
    Calls the ListImageSources service.
    Args:
        image_client (ImageClient)

    Returns:
        either the protobuf output of the ListImageSources service,
        or a dictionary
    """
    _start_time = time.time()
    sources_result = image_client.list_image_sources()  # blocking call
    _used_time = time.time() - _start_time
    return sources_result, _used_time


def getImage(image_client, requests):
    """Iterator; uses the `image_client` to send
    get_image request with `requests`."""
    _start_time = time.time()
    result = image_client.get_image(requests)
    _used_time = time.time() - _start_time
    return result, _used_time


def ros_create_publishers(sources, name_space="stream_image"):
    """
    Returns a dictionary of publishers for given sources.

    Args:
        sources (list): List of source names
    Returns:
        dict
    """
    publishers = {}
    for source in sources:
        publishers[source] = {
            "image": rospy.Publisher(
                f"/spot/{name_space}/{source}/image",
                sensor_msgs.msg.Image, queue_size=10),
            "camera_info": rospy.Publisher(
                f"/spot/{name_space}/{source}/camera_info",
                sensor_msgs.msg.CameraInfo, queue_size=10)
        }
    return publishers


def ros_publish_image_result(conn, get_image_result, publishers, broadcast_tf=True):
    """
    Publishes images in response as ROS messages (sensor_msgs.Image)
    Args:
        publishers (dict): maps from source name to a dictionary,
            {"image": rospy.Publisher(Image), "camera_info": rospy.Publisher(CameraInfo)}
        response (GetImageResponse): The message returned by GetImage service.
        broadcast_tf (bool): If true, will publish tf transforms for the camera optical frames.
    """
    tf_frames = _get_odom_tf_frames()

    # publish the image with local timestamp
    for image_response in get_image_result:
        source_name = image_response.source.name
        image_msg, camera_info_msg = ros_image_response_to_message(conn, image_response)
        publishers[source_name]['image'].publish(image_msg)
        publishers[source_name]['camera_info'].publish(camera_info_msg)
        rospy.loginfo(f"Published image response from {source_name}")

        if broadcast_tf:
            populate_camera_static_transforms(conn, image_response, tf_frames)

def ros_image_response_to_message(conn, result):
    """
    Given a result (ImageResponse), return a tuple
    (sensor_msgs/Image, sensor_msgs/CameraInfo)
    """
    local_time = conn.spot_time_to_local(
        result.shot.acquisition_time)
    image_msg, camera_info_msg =\
        spot_driver.ros_helpers._getImageMsg(result, local_time)
    return image_msg, camera_info_msg

def image_response_to_array(conn, result):
    """
    Given a result (ImageResponse), return a numpy array
    representation of the image.

    Args:
        conn (SpotSDKConn): establishes connection to the robot.
        result (ImageResponse): a single image response.
    """
    image_msg, _ = ros_image_response_to_message(conn, result)
    bridge = CvBridge()
    return bridge.imgmsg_to_cv2(image_msg)

def extract_pinhole_intrinsic(result):
    """
    Given a result (ImageResponse), return a tuple (width, length, fx, fy, cx,
    cy), which are the parameters of a pinhole camera model.
    """
    return (result.shot.image.cols,
            result.shot.image.rows,
            result.source.pinhole.intrinsics.focal_length.x,
            result.source.pinhole.intrinsics.focal_length.y,
            result.source.pinhole.intrinsics.principal_point.x,
            result.source.pinhole.intrinsics.principal_point.y)

def _get_odom_tf_frames():
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


def populate_camera_static_transforms(conn, image_data, tf_frames):
    global CAMERA_STATIC_TF_BROADCASTER
    global CAMERA_STATIC_TRANSFORMS
    if CAMERA_STATIC_TF_BROADCASTER is None:
        CAMERA_STATIC_TF_BROADCASTER = tf2_ros.StaticTransformBroadcaster()
    excluded_frames = [tf_frames['tf_name_vision_odom'], tf_frames['tf_name_kinematic_odom'], "body"]
    for frame_name in image_data.shot.transforms_snapshot.child_to_parent_edge_map:
        if frame_name in excluded_frames:
            continue
        parent_frame = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name).parent_frame_name
        existing_transforms = [(transform.header.frame_id, transform.child_frame_id)
                               for transform in CAMERA_STATIC_TRANSFORMS]
        if (parent_frame, frame_name) in existing_transforms:
            # We already extracted this transform
            continue

        transform = image_data.shot.transforms_snapshot.child_to_parent_edge_map.get(frame_name)
        local_time = conn.spot_time_to_local(image_data.shot.acquisition_time)
        tf_time = rospy.Time(local_time.seconds, local_time.nanos)
        static_tf = spot_driver.ros_helpers.populateTransformStamped(
            tf_time, transform.parent_frame_name, frame_name,
            transform.parent_tform_child)
        CAMERA_STATIC_TRANSFORMS.append(static_tf)
        CAMERA_STATIC_TF_BROADCASTER.sendTransform(CAMERA_STATIC_TRANSFORMS)


def create_client(conn):
    """
    Given conn (SpotSDKConn) returns a ImageClient.
    """
    return conn.ensure_client(ImageClient.default_service_name)


def check_sources_valid(sources, sources_result):
    """
    Checks if the image sources are valid ones provided by Spot,
    according to the sources

    Args:
        sources (list of str): List of source names
        sources_result (proto): result of listImageSources
    Returns:
        tuple (bool, list); True if all sources are valid. False if not, and the
            list will contain invalid sources.
    """
    sources_dict = sources_result_to_dict(sources_result)
    valid_source_names = set(sources_dict['name'])
    bad_sources = []
    ok = True
    for source_name in sources:
        if source_name not in valid_source_names:
            bad_sources.append(source_name)
            ok = False
    return ok, bad_sources

def build_image_requests(sources, quality=75, fmt="RAW"):
    """Create requests.
    fmt (str): must be one of IMAGE_FORMATS."""
    if fmt is not None and type(fmt) == str:
        fmt = image_pb2.Image.Format.Value(f"FORMAT_{fmt}")

    requests = []
    for source in sources:
        req = build_image_request(source,
                                  quality_percent=quality,
                                  image_format=fmt)
        requests.append(req)
    return requests

def sources_result_to_dict(sources_result):
    """
    Given the response of ListImageSources service, extract
    source names and source types, and arrange them into a
    dictionary.

    Args:
        sources_result: response of ListImageSources service.
            it is a set of ImageSource data objects
    Returns:
        a dictionary {"name": [...], "type": [...]}
    """
    # obtain enum name from enum value
    _name_func = image_pb2.ImageSource.ImageType.Name
    return {"name": [source.name for source in sources_result],
            "type": [_name_func(source.image_type)
                     for source in sources_result]}
