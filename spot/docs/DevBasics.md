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


## Obtain Spot ID
```
$ python3 -m bosdyn.client 192.168.80.3 id
spot-BD-12070012     000060189461B2000097            spot (V3)
 Software: 2.3.8 (b821228e2997 2021-08-09 17:35:08)
  Installed: 2021-08-09 19:27:31
```
[reference](https://dev.bostondynamics.com/docs/python/quickstart#request-spot-robot-s-id)
