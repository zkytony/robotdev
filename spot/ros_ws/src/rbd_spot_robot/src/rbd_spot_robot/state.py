# robot state related
from bosdyn.client.robot_state import RobotStateClient

def getRobotState(robot_state_client):
    return robot_state_client.get_robot_state()

def create_client(conn):
    return conn.ensure_client(RobotStateClient.default_service_name)
