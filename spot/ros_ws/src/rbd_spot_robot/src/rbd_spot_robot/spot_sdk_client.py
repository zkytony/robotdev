from bosdyn.client import create_standard_sdk, ResponseError, RpcError

import os
import rospy
import logging

class SpotSDKClient:
    """Assume source setup_spot.bash is successful."""
    def __init__(self, name='client'):
        self.name = name
        self._logger = logging.getLogger('rosout')
        rospy.loginfo("Starting Spot SDK Client for {}".format(self.name))

        try:
            self._sdk = create_standard_sdk('DepthVisualPublisher')
        except Exception as e:
            self._logger.error("Error creating SDK object: %s", e)
            self._valid = False
            return

        self._hostname = os.environ['SPOT_IP']
        self._username = "user"
        self._password = os.environ['SPOT_USER_PASSWORD']

        self._robot = self._sdk.create_robot(self._hostname)

        try:
            self._robot.authenticate(self._username, self._password)
            self._robot.start_time_sync()
        except RpcError as err:
            self._logger.error("Failed to communicate with robot: %s", err)
            self._valid = False
            return
