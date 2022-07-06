import time
from bosdyn.client.world_object import WorldObjectClient
from bosdyn.api import world_object_pb2

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
