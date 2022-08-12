# robot state related
import rospy
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import JointState
from spot_driver.ros_helpers import (populateTransformStamped,
                                     friendly_joint_names)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.math_helpers import SE3Pose

def getRobotState(robot_state_client):
    return robot_state_client.get_robot_state()

def create_client(conn):
    return conn.ensure_client(RobotStateClient.default_service_name)

def get_tf_from_state(state, conn, root_frame):
    """
    Args:
        state: robot state proto
        root_frame: the frame that will act as the root in the tf tree
    Returns:
        tf2_msg.TFMessage
    """
    return _GetTFFromState(state, conn, root_frame)

def get_joint_state_from_state(state, conn):
    return _GetJointStatesFromState(state, conn)

def _GetTFFromState(state, conn, inverse_target_frame):
    """
    Note: modified based on spot_ros
    Maps robot link state data from robot state proto to ROS TFMessage message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
        inverse_target_frame: A frame name to be inversed to a parent frame.
    Returns:
        TFMessage message
    """
    tf_msg = TFMessage()

    for frame_name in state.kinematic_state.transforms_snapshot.child_to_parent_edge_map:
        if state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get(frame_name).parent_frame_name:
            try:
                transform = state.kinematic_state.transforms_snapshot.child_to_parent_edge_map.get(frame_name)
                local_time = conn.spot_time_to_local(state.kinematic_state.acquisition_timestamp)
                tf_time = rospy.Time(local_time.seconds, local_time.nanos)
                if inverse_target_frame == frame_name:
                    geo_tform_inversed = SE3Pose.from_obj(transform.parent_tform_child).inverse()
                    new_tf = populateTransformStamped(tf_time, frame_name, transform.parent_frame_name, geo_tform_inversed)
                else:
                    new_tf = populateTransformStamped(tf_time, transform.parent_frame_name, frame_name, transform.parent_tform_child)
                tf_msg.transforms.append(new_tf)
            except Exception as e:
                print('Error: {}, {}'.format(type(e), e))

    return tf_msg

def _GetJointStatesFromState(state, conn):
    """
    Note: modified based on spot_ros
    Maps joint state data from robot state proto to ROS JointState message

    Args:
        data: Robot State proto
        spot_wrapper: A SpotWrapper object
    Returns:
        JointState message
    """
    joint_state = JointState()
    try:
        local_time = conn.spot_time_to_local(state.kinematic_state.acquisition_timestamp)
        joint_state.header.stamp = rospy.Time(local_time.seconds, local_time.nanos)
        for joint in state.kinematic_state.joint_states:
            joint_state.name.append(friendly_joint_names.get(joint.name, "ERROR"))
            joint_state.position.append(joint.position.value)
            joint_state.velocity.append(joint.velocity.value)
            joint_state.effort.append(joint.load.value)
    except Exception as e:
        print('Error: {}, {}'.format(type(e), e))
    return joint_state
