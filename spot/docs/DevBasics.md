# Software Development with Spot

Official [Spot SDK docs](https://dev.bostondynamics.com/).

The Spot API follows a client-server model. Client applications
communicate with on-board services over a network connection.

The Spot services can be categorized into "core", "robot" and "autonomy" as follows:

![spot sdk viz](https://d33wubrfki0l68.cloudfront.net/f34bcc5ff400782c699351096b289ed9f943164a/32716/_images/api_top_level.png)


## Install the Spot Python SDK

[Main reference](https://dev.bostondynamics.com/docs/python/quickstart#system-setup).

1. `source setup_spot.bash`. Just to be in the right mindset.
2. Run
  ```
  pip install --upgrade bosdyn-client bosdyn-mission bosdyn-choreography-client
  ```
  (These should already been installed if you had setup Spot with ROS).
  
3. Verify. `pip list | grep bosdyn`. Everything is so far version `3.0.3`.

## Install the Full Spot SDK

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
    

## Obtain Spot ID
```
$ python3 -m bosdyn.client 192.168.80.3 id
spot-BD-12070012     000060189461B2000097            spot (V3)
 Software: 2.3.8 (b821228e2997 2021-08-09 17:35:08)
  Installed: 2021-08-09 19:27:31
```

[reference](https://dev.bostondynamics.com/docs/python/quickstart#request-spot-robot-s-id)

## Troubleshooting

### Resource AlreadyClaimedError()

When running the hello_spot.py script, I get:
```
$ python3 hello_spot.py --username user --password $SPOT_USER_PASSWORD $SP
OT_IP
2022-02-09 14:55:20,001 - ERROR - Hello, Spot! threw an exception: Resource
AlreadyClaimedError()
```

[This BD support thread](https://support.bostondynamics.com/s/question/0D54X00006UNIReSAP/get-resourcealreadyclaimederror-when-trying-to-run-the-hellospotpy-example-code) talks about the exact same issue.
The problem seems to be the controller is "Taking Control" of the robot. So you need to either "RELEASE CONTROL" from the controller,
or force take the control from the script, do `lease_client.take()` instead of `lease_client.acquire()` as the support suggests.

Note that when you RELEASE CONTROL, if the robot is standing, it will sit down. And the LED lights will be rainbow colors.
