# Provides functions to interact with image services Spot SDK
# and connect that to ROS.

import time
import rospy

from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient, build_image_request

import spot_driver.ros_helpers

IMAGE_FORMATS = ["UNKNOWN", "JPEG", "RAW", "RLE"]


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


def getImageStream(image_client, requests):
    """Iterator; uses the `image_client` to send
    get_image request with `requests`."""
    while True:
        _start_time = time.time()
        response = image_client.get_image(requests)
        _used_time = time.time() - _start_time
        yield response, _used_time


def ros_publish_image_result(conn, get_image_result, publishers):
    """
    Publishes images in response as ROS messages (sensor_msgs.Image)
    Args:
        publishers (dict): maps from source name to a dictionary,
            {"image": rospy.Publisher(Image), "camera_info": rospy.Publisher(CameraInfo)}
        response (GetImageResponse): The message returned by GetImage service.
    """
    # publish the image with local timestamp
    for image_response in get_image_result:
        local_time = conn.spot_time_to_local(
            image_response.shot.acquisition_time)
        image_msg, camera_info_msg =\
            spot_driver.ros_helpers._getImageMsg(image_response, local_time)
        source_name = image_response.source.name
        publishers[source_name]['image'].publish(image_msg)
        publishers[source_name]['camera_info'].publish(camera_info_msg)
        rospy.loginfo(f"Published image response from {source_name}")


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

def build_image_requests(sources, quality=75, fmt=None):
    """Create requests.
    fmt (None or str): If not None, must be one of IMAGE_FORMATS."""
    if fmt is not None and type(fmt) == str:
        fmt = image_pb2.Image.Format.Value(IMAGE_FORMATS.index(fmt))

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
