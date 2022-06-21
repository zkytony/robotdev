import time
import os
from bosdyn.api.graph_nav import map_pb2
from bosdyn.api.graph_nav import nav_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.frame_helpers import get_odom_tform_body
from . import graphnav_util

def create_client(conn):
    """
    Given conn (SpotSDKConn) returns a ImageClient.
    """
    return conn.ensure_client(GraphNavClient.default_service_name)


def _get_point_cloud_data_in_seed_frame(waypoints, snapshots, anchorings, waypoint_id):
    """
    Create a 3 x N numpy array of points in the seed frame. Note that in graph_nav, "point cloud" refers to the
    feature cloud of a waypoint -- that is, a collection of visual features observed by all five cameras at a particular
    point in time. The visual features are associated with points that are rigidly attached to a waypoint.
    :param waypoints: dict of waypoint ID to waypoint.
    :param snapshots: dict of waypoint snapshot ID to waypoint snapshot.
    :param anchorings: dict of waypoint ID to the anchoring of that waypoint w.r.t the map.
    :param waypoint_id: the waypoint ID of the waypoint whose point cloud we want to render.
    :return: a 3 x N numpy array in the seed frame.
    """
    wp = waypoints[waypoint_id]
    snapshot = snapshots[wp.snapshot_id]
    cloud = snapshot.point_cloud
    odom_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, ODOM_FRAME_NAME,
                                     cloud.source.frame_name_sensor)
    waypoint_tform_odom = SE3Pose.from_obj(wp.waypoint_tform_ko)
    waypoint_tform_cloud = waypoint_tform_odom * odom_tform_cloud
    if waypoint_id not in anchorings:
        raise Exception("{} not found in anchorings. Does the map have anchoring data?".format(waypoint_id))
    seed_tform_cloud = SE3Pose.from_obj(anchorings[waypoint_id].seed_tform_waypoint) * waypoint_tform_cloud
    point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
    return seed_tform_cloud.transform_cloud(point_cloud_data)


def load_map(path):
    """
    Load a map from the given file path.
    :param path: Path to the root directory of the map.
    :return: the graph, waypoints, waypoint snapshots, edge snapshots, and anchorings.
    """
    with open(os.path.join(path, "graph"), "rb") as graph_file:
        # Load the graph file and deserialize it. The graph file is a protobuf containing only the waypoints and the
        # edges between them.
        data = graph_file.read()
        current_graph = map_pb2.Graph()
        current_graph.ParseFromString(data)

        # Set up maps from waypoint ID to waypoints, edges, snapshots, etc.
        current_waypoints = {}
        current_waypoint_snapshots = {}
        current_edge_snapshots = {}
        current_anchors = {}
        current_anchored_world_objects = {}

        # Load the anchored world objects first so we can look in each waypoint snapshot as we load it.
        for anchored_world_object in current_graph.anchoring.objects:
            current_anchored_world_objects[anchored_world_object.id] = (anchored_world_object,)
        # For each waypoint, load any snapshot associated with it.
        for waypoint in current_graph.waypoints:
            current_waypoints[waypoint.id] = waypoint

            if len(waypoint.snapshot_id) == 0:
                continue
            # Load the snapshot. Note that snapshots contain all of the raw data in a waypoint and may be large.
            file_name = os.path.join(path, "waypoint_snapshots", waypoint.snapshot_id)
            if not os.path.exists(file_name):
                continue
            with open(file_name, "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot

                for fiducial in waypoint_snapshot.objects:
                    if not fiducial.HasField("apriltag_properties"):
                        continue

                    str_id = str(fiducial.apriltag_properties.tag_id)
                    if (str_id in current_anchored_world_objects and
                            len(current_anchored_world_objects[str_id]) == 1):

                        # Replace the placeholder tuple with a tuple of (wo, waypoint, fiducial).
                        anchored_wo = current_anchored_world_objects[str_id][0]
                        current_anchored_world_objects[str_id] = (anchored_wo, waypoint, fiducial)

        # Similarly, edges have snapshot data.
        for edge in current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            file_name = os.path.join(path, "edge_snapshots", edge.snapshot_id)
            if not os.path.exists(file_name):
                continue
            with open(file_name, "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        for anchor in current_graph.anchoring.anchors:
            current_anchors[anchor.id] = anchor
        print("Loaded graph with {} waypoints, {} edges, {} anchors, and {} anchored world objects".
              format(len(current_graph.waypoints), len(current_graph.edges),
                     len(current_graph.anchoring.anchors), len(current_graph.anchoring.objects)))
        return (current_graph, current_waypoints, current_waypoint_snapshots,
                current_edge_snapshots, current_anchors, current_anchored_world_objects)


def load_map_as_points(path):
    """
    Given a string path to a folder that contains a downloaded GraphNav map,
    loads the map as point cloud and returns a (N,3) numpy array.

    If asfloat is True, the output numpy array will have type np.float32.
    """
    (current_graph, current_waypoints, current_waypoint_snapshots, current_edge_snapshots,
     current_anchors, current_anchored_world_objects) = load_map(path)

    # Concatenate the data from all waypoints.
    data = None  # will be a 3XN numpy array
    for wp in current_graph.waypoints:
        cloud_data = _get_point_cloud_data_in_seed_frame(
            current_waypoints, current_waypoint_snapshots, current_anchors, wp.id)
        if data is None:
            data = cloud_data
        else:
            data = np.concatenate((data, cloud_data))
    print(data[:10])
    print(data.shape)
    print(data.dtype)
    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(data)
    # viz = o3d.visualization.Visualizer()
    # viz.create_window()
    # viz.add_geometry(pcd)
    # opt = viz.get_render_option()
    # opt.show_coordinate_frame = True
    # viz.run()
    # viz.destroy_window()
    return data



def uploadGraph(graphnav_client, graph, waypoint_snapshots, edge_snapshots, lease=None):
    """given path to graphnav map directory, upload this
    map to the robot. Modifed based on _upload_graph_and_snapshots in
    Spot SDK graphnav examples.

    Args:
        map_path (str): path to graphnav map directory
        lease (Lease): The Lease to show ownership of graph-nav service. """
    print("Uploading the graph and snapshots to the robot...")
    if lease is None:
        lease_proto = None
    else:
        lease_proto = lease.lease_proto

    true_if_empty = not len(graph.anchoring.anchors)
    _start_time = time.time()
    upload_result = graphnav_client.upload_graph(lease=lease_proto,
                                            graph=graph,
                                            generate_new_anchoring=true_if_empty)
    # Upload the snapshots to the robot.
    for snapshot_id in upload_result.unknown_waypoint_snapshot_ids:
        waypoint_snapshot = waypoint_snapshots[snapshot_id]
        graphnav_client.upload_waypoint_snapshot(waypoint_snapshot)
        print("Uploaded {}".format(waypoint_snapshot.id))
    for snapshot_id in upload_result.unknown_edge_snapshot_ids:
        edge_snapshot = edge_snapshots[snapshot_id]
        graphnav_client.upload_edge_snapshot(edge_snapshot)
        print("Uploaded {}".format(edge_snapshot.id))
    _used_time = time.time() - _start_time
    return upload_result, _used_time


def clearGraph(graphnav_client, lease=None):
    """Clear the state of the map on the robot, removing all waypoints and edges."""
    if lease is None:
        lease_proto = None
    else:
        lease_proto = lease.lease_proto
    _start_time = time.time()
    result = graphnav_client.clear_graph(lease=lease_proto)
    _used_time = time.time() - _start_time
    return result, _used_time


def downloadGraph(graphnav_client):
    _start_time = time.time()
    # note that this gets the graph directly instead of a Response object
    graph = graphnav_client.download_graph()
    _used_time = time.time() - _start_time
    return graph, _used_time

def getLocalizationState(graphnav_client):
    _start_time = time.time()
    state_result = graphnav_client.get_localization_state()
    _used_time = time.time() - _start_time
    return state_result, _used_time

def get_pose(state_result, frame='waypoint', stamped=False):
    """
    frame: the frame this pose is with respect to.
        Either 'waypoint' or 'seed'

    Returns:
        if frame == 'waypoint', then
            (waypoint id, pose)
        if frame == 'seed', then
            pose
        if stamped is True, append timestamp to the return.
    """
    if frame != "waypoint" and frame != "seed":
        raise ValueError("frame must be 'waypoint' or 'seed'")

    if not state_result.localization.waypoint_id:
        # The robot is not localized to the newly uploaded graph.
        print("\n")
        print("The robot is currently not localized to the map; please localize")
        return

    if frame == "waypoint":
        if stamped:
            return state_result.localization.waypoint_id,\
                state_result.localization.waypoint_tform_body,\
                state_result.localization.timestamp
        else:
            return state_result.localization.waypoint_id,\
                state_result.localization.waypoint_tform_body
    else:
        if stamped:
            return state_result.localization.seed_tform_body,\
                state_result.localization.timestamp
        else:
            return state_result.localization.seed_tform_body


def setLocalizationFiducial(graphnav_client, robot_state_client):
    """Trigger localization when near a fiducial. Code taken from SDK example"""
    robot_state = robot_state_client.get_robot_state()
    current_odom_tform_body = get_odom_tform_body(
        robot_state.kinematic_state.transforms_snapshot).to_proto()
    # Create an empty instance for initial localization since we are asking it to localize
    # based on the nearest fiducial.
    localization = nav_pb2.Localization()
    _start_time = time.time()
    result = graphnav_client.set_localization(initial_guess_localization=localization,
                                              ko_tform_body=current_odom_tform_body)
    _used_time = time.time() - _start_time
    return result, _used_time



def setLocalizationWaypoint(graphnav_client, robot_state_client,
                            waypoint_id=None, graph=None):
    """Trigger localization to a waypoint. Code taken from SDK example"""
    assert waypoint_id is not None, "waypoint_id required"
    assert graph is not None, "graph required"
    annotation_name_to_wp_id, edges =\
        graphnav_util.update_waypoints_and_edges(current_graph, waypoint_id)

    destination_waypoint = graph_navutil.find_unique_waypoint_id(
        waypoint_id, graph, annotation_name_to_wp_id)
    if not destination_waypoint:
        # Failed to find the unique waypoint id.
        return

    robot_state = robot_state_client.get_robot_state()
    current_odom_tform_body = get_odom_tform_body(
        robot_state.kinematic_state.transforms_snapshot).to_proto()
    # Create an initial localization to the specified waypoint as the identity.
    localization = nav_pb2.Localization()
    localization.waypoint_id = destination_waypoint
    localization.waypoint_tform_body.rotation.w = 1.0
    _start_time = time.time()
    result = graphnav_client.set_localization(
        initial_guess_localization=localization,
        # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
        max_distance=0.2,
        max_yaw=20.0 * math.pi / 180.0,
        fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
        ko_tform_body=current_odom_tform_body)
    _used_time = time.time() - _start_time
    return result, _used_time
