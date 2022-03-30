# Troubleshooting

## Resource AlreadyClaimedError()

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

## Robot is estopped

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


## Stand (ID 5689) no longer processing

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


## No Exception Printed
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

## Could not find version for vtk

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
