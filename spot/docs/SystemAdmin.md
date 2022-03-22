# System Admin

Reference: https://www.bostondynamics.com/sites/default/files/inline-files/spot-system-administration.pdf

Log in with 'admin' and its password.

The primary tools for administration:
- web-based console
- diagnostic services and calibration tools through the tablet controller

(so there is no terminal-level administration)



## Software Update

Boston dynamics manages Spot software "as a whole."

To update software, you need to download the ".BDE" file "for the latest release by following the link provided by Boston
Dynamics."

https://support.bostondynamics.com/s/downloads

Then, in the admin console, go to "Software Update" and then upload and install the downloaded file.

**The robot's key software systems must be running the same version of software.** When updating
the robot's software the following components must also be updated:
* The robot's software
* The tablet controller app
* The Spot SDK
* Spot CAM (if present)
* Spot CORE (if present)


If you see message in orange box:
>The robot is taking longer than normal to reboot. This page will time out after
>5 minutes. The reboot may have already succeeded. You may try refreshing this
>page or power cycling the robot again.
Then you can refresh.

If the upgrade is successful, both "CURRENTLY STAGED FILE" and "SOFTWARE VERSION"
should show the correct new version.


#### Software Logs
* (012) As of 02/16/2022, the "currently staged file" is "spot_release_v2.3.8.bde" installed on 08-09-2021
* (012) As of 03/09/2022, the "currently staged file" is "spot_release_v3.1.0.bde" installed on 03-09-2022
* (002) As of 2022-03-22 15:33:35, the "currently staged file" is "spot_release_v3.1.0.bde" installed on 03-22-2022


### Update Tablet App

Follow the steps [here](https://support.bostondynamics.com/s/article/Updating-the-Spot-system)

1. On the tablet, go to support.bostondynamics.com
2. Sign in
3. Go to "Downloads."
4. Download the desired .apk file (that is the android app; e.g. version 3.1.0)
5. Exit chrome
6. Go to Explorer (you can swipe up from the tablet to see all apps; Explorer is one of them)
7. Click "Home" -> "internal memory" -> "Download"
8. Select ".apk" file (for your particular version)
9. Click it, it will ask you to INSTALL
10. If you install, it will override the previous app.
11. After installation is successful, click on the same Spot app and you should see it in a new version.

**Note**: After software upgrade on the tablet (to v3.1.0), "AutoLogin" doesn't work. You can still, however, enter the username and password credentials manually and the tablet can STILL CONTROL the robot (which is running an older software version (v2.3.8))


### Important points 3.1.0

- The `–username` and `–password` command line options are deprecated in the Python
  SDK. (**affects, at least, the examples**)

  The solution BD came up with is to ask you to use
  `bosdyn.client.util.add_base_arguments(parser)` when
  creating CLI and not `add_common_arguments()`.

  The username and password are in fact provided through
  `BOSDYN_CLIENT_USERNAME` and `BOSDYN_CLIENT_PASSWORD`
  environment variables (this is how we are doing it in robotdev anyways!!
  SPOT_IP, SPOT_USER_PASSWORD ...)

  _I am not sure when these environment variables are set though._
  It wasn't clear from their [release notes](https://dev.bostondynamics.com/docs/release_notes)
