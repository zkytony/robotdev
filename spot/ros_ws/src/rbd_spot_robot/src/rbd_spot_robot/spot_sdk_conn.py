import os
from dataclasses import dataclass, field

from bosdyn.client import create_standard_sdk, ResponseError, RpcError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, LeaseWallet, ResourceAlreadyClaimedError
from google.protobuf.timestamp_pb2 import Timestamp

import logging


@dataclass(init=True)
class SpotSDKConn:
    """Establishes connection with Spot SDK and
    creates standard objects every program that
    uses Spot SDK needs such as 'robot'"""
    # Assumes you have successfully run source setup_spot.sh
    sdk_name: str
    hostname: str = os.environ['SPOT_IP']
    username: str = "user"
    password: str = os.environ['SPOT_USER_PASSWORD']
    conn_type: str = os.environ['SPOT_CONN'].replace(" ", "_")
    logto: str = "rosout"
    acquire_lease: bool = False
    take_lease: bool = False

    def __post_init__(self):
        self.logger = logging.getLogger(self.logto)
        try:
            sdk = create_standard_sdk(self.sdk_name)
        except Exception as e:
            self.logger.error("Error creating SDK object: %s", e)
            return

        self.robot = sdk.create_robot(self.hostname)
        try:
            self.robot.authenticate(self.username, self.password)
            self.robot.start_time_sync()
        except RpcError as err:
            self.logger.error("Failed to communicate with robot: %s", err)
            return

        self._lease = None
        if self.acquire_lease or self.take_lease:
            self._lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
            self.getLease()

    def ensure_client(self, service_name):
        return self.robot.ensure_client(service_name)

    def getLease(self):
        try:
            if self.take_lease:
                self.logger.info("Take lease!")
                self._lease = self._lease_client.take()
            else:
                self._lease = self._lease_client.acquire()
        except (ResponseError, RpcError) as err:
            self.logger.error("Failed to obtain lease: ", err)
            raise ValueError("Failed to obtain lease: ", err)
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

    def __del__(self):
        """Give up the lease"""
        if self._lease is not None:
            self._lease_keepalive.shutdown()
            self._lease_client.return_lease(self._lease)

    @property
    def lease(self):
        return self._lease

    @property
    def clock_skew(self):
        return self.robot.time_sync.endpoint.clock_skew

    def spot_time_to_local(self, spot_timestamp):
        """
        Args:
            spot_timestamp (google.protobuf.Timestamp): timestamp in Spot clock
        Returns:
            google.protobuf.Timestamp
        """
        seconds = spot_timestamp.seconds - self.clock_skew.seconds
        nanos = spot_timestamp.nanos - self.clock_skew.nanos
        if nanos < 0:
           nanos = nanos + 1000000000  # 10 digits
           seconds = seconds - 1
        if seconds < 0:
            seconds = 0
            nanos = 0
        return Timestamp(seconds=seconds, nanos=nanos)
