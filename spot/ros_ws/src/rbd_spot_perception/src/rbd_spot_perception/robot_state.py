from bosdyn.client.robot_state import RobotStateClient


def create_client(conn):
    """
    Given conn (SpotSDKConn) returns a ImageClient.
    """
    return conn.ensure_client(RobotStateClient.default_service_name)
