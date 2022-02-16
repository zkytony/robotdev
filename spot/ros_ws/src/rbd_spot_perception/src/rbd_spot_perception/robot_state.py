# robot state

import bosdyn.client
import bosdyn.client.util

import rbd_spot_robot.spot_sdk_service as srv  # access Spot SDK services through here
from rbd_spot_robot.spot_sdk_client import SpotSDKClient

class RobotStateClient(SpotSDKClient):
    def __init__(self):
        super(RobotStateClient, self).__init__("RobotStateClient", name="robot_state_client")
        self._robot_state_client = self._robot.ensure_client(
            srv.RobotStateClient.default_service_name)

    def get(self):
        return self._robot_state_client.get_robot_state()
