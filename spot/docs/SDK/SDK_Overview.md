Official [Spot SDK docs](https://dev.bostondynamics.com/).

The Spot API follows a client-server model. Client applications
communicate with on-board services over a network connection.

The Spot services can be categorized into "core", "robot" and "autonomy" as follows:

![spot sdk viz](https://d33wubrfki0l68.cloudfront.net/f34bcc5ff400782c699351096b289ed9f943164a/32716/_images/api_top_level.png)


# Getting Started with Spot SDK

Useful resources:

* [Python Examples for Spot SDK](https://dev.bostondynamics.com/python/examples/readme)

   * [Arm Examples](https://dev.bostondynamics.com/python/examples/docs/arm_examples)


**Note on requirements.txt for each example:**
In the remainder of this document, when we mention examples provided by Boston
Dynamics for Spot SDK, "`[N]`" means no additional packages were installed when
running `pip install -r requirements.txt` (assuming you had successfully set up this
repository by running `setup_spot.sh`).

If there is additional packages to be installed, they will be noted as `[pkg1, pkg2, ...]`.


# Obtain Spot ID
```
$ python3 -m bosdyn.client 192.168.80.3 id
spot-BD-12070012     000060189461B2000097            spot (V3)
 Software: 2.3.8 (b821228e2997 2021-08-09 17:35:08)
  Installed: 2021-08-09 19:27:31
```

[reference](https://dev.bostondynamics.com/docs/python/quickstart#request-spot-robot-s-id)
