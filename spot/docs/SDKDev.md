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


3. Run estop in a separate shell. Press Estop any time during the hello_spot script. "EStop is your friend." --- BD Documentation.

    Steps to run E-Stop:
    ```
    cd ~/spot-sdk/python/examples/estop
    python3 -m pip install -r requirements.txt
    $ python3 estop_gui.py --username user --password $SPOT_USER_PASSWORD $SPOT_IP
    ```

    Here is a scene after Spot "glide stopped" after EStop is pressed:
    <img src="https://i.imgur.com/9myKjho.jpg" width="550px"/>


## Getting Started with Spot SDK

Useful resources:

* [Python Examples for Spot SDK](https://dev.bostondynamics.com/python/examples/readme)

   * [Arm Examples](https://dev.bostondynamics.com/python/examples/docs/arm_examples)


**Note on requirements.txt for each example:**
In the remainder of this document, when we mention examples provided by Boston
Dynamics for Spot SDK, "`[N]`" means no additional packages were installed when
running `pip install -r requirements.txt` (assuming you had successfully set up this
repository by running `setup_spot.sh`).

If there is additional packages to be installed, they will be noted as `[pkg1, pkg2, ...]`.


## Obtain Spot ID
```
$ python3 -m bosdyn.client 192.168.80.3 id
spot-BD-12070012     000060189461B2000097            spot (V3)
 Software: 2.3.8 (b821228e2997 2021-08-09 17:35:08)
  Installed: 2021-08-09 19:27:31
```

[reference](https://dev.bostondynamics.com/docs/python/quickstart#request-spot-robot-s-id)


## Arm Control

Look at [these examples](https://dev.bostondynamics.com/python/examples/docs/arm_examples)

Experience with different examples:

- Arm Simple `[N]`: the robot stands up, and the arm starts moving. The end effector rotates. The arm did a simple extension. That's it.

- Arm Stow and Unstow `[N]`: the robot stands up. The arm extends (stow). Then returns (unstow).


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
Once you take back control with the controller (or when you are running the hello_spot script that needs control), the LED lights will turn green again. (When the hello_spot script finishes running, control is released again and the lights turn rainbow again).

### Robot is estopped

This may happen after you E-Stop the robot and then try to take back control.

You are able to take control, but the controller will show a red "(ERROR)" on the top right.

Also, when you run hello_spot.py, you get the error message:
```
$ python3 hello_spot.py --username user --password $SPOT_USER_PASSWORD $SPOT_IP
2022-02-09 15:16:23,047 - ERROR - Hello, Spot! threw an exception: Assertio
nError('Robot is estopped. Please use an external E-Stop client, such as th
e estop SDK example, to configure E-Stop.')
```
Solution: Press the green "Release" on the EStop GUI. If the "(ERROR)" on the top-right of controller doesn't go away, try to run hello_spot.py again. You will get pass the error above, but you may get a new error (see below).

Here is a photo of the controller when the ERROR happened. The "WIFI" sign is flashing yellow.

<img src="https://i.imgur.com/VWq2LVk.jpg" width="400px"/>


### Stand (ID 5689) no longer processing

This happened after Spot was first E-Stopped, then I released the E-Stop, then I try to run the hello_spot.py again. The Spot won't stand up. Controller keeps showing `(ERROR)` on top right.

```
$ python3 hello_spot.py --username user --password $SPOT_USER_PASSWORD $SPOT_IP
2022-02-09 15:19:26,187 - INFO - Starting lease check-in
2022-02-09 15:19:26,192 - INFO - Powering on robot... This may take several
 seconds.
2022-02-09 15:19:34,369 - INFO - Robot powered on.
2022-02-09 15:19:34,369 - INFO - Commanding robot to stand...
2022-02-09 15:19:34,400 - INFO - Lease check-in stopped
2022-02-09 15:19:34,405 - ERROR - Hello, Spot! threw an exception: CommandFailedError('Stand (ID 5689) no longer processing (now STATUS_COMMAND_OVERRIDDEN)')
```

This may in fact be a sign that the RPCs running on Spot are terminated after E-Stop. Something similar happens on MOVO.
One way to resolve this is to just reboot Spot. I have also reached out to BD support.


### No Exception Printed
When using `AsyncImageService` and `AsyncTasks` from Spot ROS and Spot SDK,
exceptions won't terminate the program and exception messages are not printed.


**Investigation:**
The callback function passed into `AsyncImageService`, when called, will not throw the
error message when there is an exception.


**SOLUTION:** The `SpotWrapper` (and the `SpotSDKClient`) has a logger.
Use that logger to log the error message in the callback function. That
is the only way. For example:
```python
def DepthVisualCB(self, results):
    try:
        raise ValueError("HELLO")
    except Exception as e:
        self._logger.error("Error during callback: {}".format(e)
```
This will show up as:
```
...
[ERROR] [1644629373.765228]: Error during callback: HELLO
```

### Could not find version for vtk

I keep getting this error when running `python3 -m pip install -r requirements.txt`
in Spot SDK examples (e.g. `visualizer/basic_streaming_visualizer.py`)
```
$ python3 -m pip install -r requirements.txt
...
ERROR: Could not find a version that satisfies the requirement
 vtk==8.1.2 (from -r requirements.txt (line 5)) (from versions
: 9.0.1, 9.0.2, 9.0.3, 9.1.0rc1, 9.1.0rc2, 9.1.0rc4, 9.1.0)
ERROR: No matching distribution found for vtk==8.1.2 (from -r
requirements.txt (line 5))
```

VTK is a C++-based visualization toolkit for 3D. It's very old but popular still.
[VTK 8.1.2](https://pypi.org/project/vtk/8.1.2/) was released in 2018.
As of 02/16/2022, the most recent VTK version is [9.1.0](https://pypi.org/project/vtk/).
The version 8.1.2 was released for python 3.5, 3.6 and 3.7. The virtualenv
for Spot I am using uses Python 3.8.

Turns out that 9.1.0 seems to just work. You can ignore the error message,
if the virtualenv has already installed VTK 9.1.0. Below is a screenshot when
it is working:


## Appendix

### Spot SDK Services

```
$ python -m bosdyn.client --user user --password $SPOT_USER_PASSWORD $SPOT_IP dir list
name                             type                                                      authority                                   tokens
--------------------------------------------------------------------------------------------------------------------------------------------
arm-surface-contact              bosdyn.api.ArmSurfaceContactService                       arm-surface-contact.spot.robot              user
auth                             bosdyn.api.AuthService                                    auth.spot.robot
auto-return                      bosdyn.api.auto_return.AutoReturnService                  auto-return.spot.robot                      user
choreography                     bosdyn.api.spot.ChoreographyService                       choreographyservice.spot.robot              user
data                             bosdyn.api.DataService                                    data.spot.robot                             user
data-acquisition                 bosdyn.api.DataAcquisitionService                         data-acquisition.spot.robot                 user
data-acquisition-store           bosdyn.api.DataAcquisitionStoreService                    data-acquisition-store.spot.robot           user
data-buffer                      bosdyn.api.DataBufferService                              buffer.spot.robot                           user
data-buffer-private              bosdyn.api.DataBufferService                              bufferprivate.spot.robot                    user
deprecated-auth                  bdRobotApi.AuthService                                    auth.spot.robot
directory                        bosdyn.api.DirectoryService                               api.spot.robot                              user
directory-registration           bosdyn.api.DirectoryRegistrationService                   api.spot.robot                              user
docking                          bosdyn.api.docking.DockingService                         docking.spot.robot                          user
door                             bosdyn.api.spot.DoorService                               door.spot.robot                             user
echo                             bosdyn.api.EchoService                                    echo.spot.robot                             user
estop                            bosdyn.api.EstopService                                   estop.spot.robot                            user
fault                            bosdyn.api.FaultService                                   fault.spot.robot                            user
graph-nav-service                bosdyn.api.graph_nav.GraphNavService                      graph-nav.spot.robot                        user
gripper-camera-param-service     bosdyn.api.internal.GripperCameraParamService             gripper-camera-param-service.spot.robot     user
image                            bosdyn.api.ImageService                                   api.spot.robot                              user
internal.localnav                bosdyn.api.internal.localnav.LocalNavService              localnav.spot.robot                         user
lease                            bosdyn.api.LeaseService                                   api.spot.robot                              user
license                          bosdyn.api.LicenseService                                 api.spot.robot                              user
local-grid-service               bosdyn.api.LocalGridService                               localgrid.spot.robot                        user
log-annotation                   bosdyn.api.LogAnnotationService                           log.spot.robot                              user
manipulation                     bosdyn.api.ManipulationApiService                         manipulation.spot.robot                     user
metrics-logging                  bosdyn.api.metrics_logging.MetricsLoggingRobotService     metricslogging.spot.robot
network-compute-bridge           bosdyn.api.NetworkComputeBridge                           network-compute-bridge.spot.robot           user
old_directory                    bdRobotApi.DirectoryService                               api.spot.robot                              user
payload                          bosdyn.api.PayloadService                                 payload.spot.robot                          user
payload-registration             bosdyn.api.PayloadRegistrationService                     payload-registration.spot.robot
power                            bosdyn.api.PowerService                                   power.spot.robot                            user
public_api                       bdRobotApi.RobotRpc                                       api.spot.robot                              user
public_estop_api                 bdRobotApi.estop.RobotEstopRpc                            estop-api.spot.robot                        user
ray-cast                         bosdyn.api.internal.RayCastService                        ray-cast.spot.robot                         user
recording-service                bosdyn.api.graph_nav.GraphNavRecordingService             recordingv2.spot.robot                      user
robot-command                    bosdyn.api.RobotCommandService                            command.spot.robot                          user
robot-id                         bosdyn.api.RobotIdService                                 id.spot.robot
robot-mission                    bosdyn.api.mission.MissionService                         robot-mission.spot.robot                    user
robot-state                      bosdyn.api.RobotStateService                              state.spot.robot                            user
robot_images                     bdRobotApi.ImageService                                   api.spot.robot                              user
spot-check                       bosdyn.api.spot.SpotCheckService                          check.spot.robot                            user
time-sync                        bosdyn.api.TimeSyncService                                api.spot.robot                              user
world-objects                    bosdyn.api.WorldObjectService                             world-objects.spot.robot                    user
```
