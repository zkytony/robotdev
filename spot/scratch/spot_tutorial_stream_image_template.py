# Imports
import os
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util

def main():
    # Establish connection
    # 1. create SDK
    bosdyn.client.util.setup_logging()
    sdk = bosdyn.client.create_standard_sdk('MyStreamImageClient')

    # 2. create robot
    hostname = os.environ.get('SPOT_IP', None)
    robot = sdk.create_robot(hostname)

    # 3. authenticate
    username = "user"
    password = os.environ.get('SPOT_USER_PASSWORD', None)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()

    # Lease is unnecessary! (because we are not controlling the robot)

    # Create Client & Call Service
    pass

if __name__ == "__main__":
    main()
