# Install the Spot Python SDK

[Main reference](https://dev.bostondynamics.com/docs/python/quickstart#system-setup).

1. `source setup_spot.bash`. Just to be in the right mindset.
2. Run
  ```
  pip install --upgrade bosdyn-client bosdyn-mission bosdyn-choreography-client
  ```
  (These should already been installed if you had setup Spot with ROS).

3. Verify. `pip list | grep bosdyn`. Everything is so far version `3.0.3`.

# Install the Full Spot SDK

1. Clone the repo (or add it as submodule). This is about 160MB.
  ```
  git clone https://github.com/boston-dynamics/spot-sdk.git
  ```
2.  [Run the Hello Spot example](https://dev.bostondynamics.com/docs/python/quickstart#run-hello-spot-let-s-see-the-robot-move)
    You need to do `pip install -r requirements.txt` specifically for that example. Note that you should be inside your spot virtualenv (created by `setup_spot.bash`).

    Note that with our setup, you can run the hello_spot.py script by:
    ```
    python3 hello_spot.py --username user --password $SPOT_USER_PASSWORD $SPOT_IP
    ```
    Note that you need to release control from your controller otherwise you get a `ResourceAlreadyClaimedError`.
    If successful, you get output:
    ```
    $ python3 hello_spot.py --username user --password $SPOT_USER_PASSWORD $SP
    OT_IP
    2022-02-09 14:59:54,893 - INFO - Starting lease check-in
    2022-02-09 14:59:54,904 - INFO - Powering on robot... This may take several
     seconds.
    2022-02-09 15:00:03,090 - INFO - Robot powered on.
    2022-02-09 15:00:03,091 - INFO - Commanding robot to stand...
    2022-02-09 15:00:04,641 - INFO - Robot standing.
    2022-02-09 15:00:07,650 - INFO - Robot standing twisted.
    2022-02-09 15:00:10,665 - INFO - Robot standing tall.
    2022-02-09 15:00:16,765 - INFO - Added comment "HelloSpot tutorial user com
    ment." to robot log.
    2022-02-09 15:00:25,808 - INFO - Robot safely powered off.
    2022-02-09 15:00:25,808 - INFO - Lease check-in stopped
    ```
    and you see the robot move accordingly, and you see a picture taken from the front-left camera.


3. Run estop in a separate shell. Press Estop any time during the hello_spot script. "EStop is your friend." --- BD Documentation.

    Steps to run E-Stop:
    ```
    cd ~/spot-sdk/python/examples/estop
    python3 -m pip install -r requirements.txt
    $ python3 estop_gui.py --username user --password $SPOT_USER_PASSWORD $SPOT_IP
    ```

    Here is a scene after Spot "glide stopped" after EStop is pressed:
    <img src="https://i.imgur.com/9myKjho.jpg" width="550px"/>
