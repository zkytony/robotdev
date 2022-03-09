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
Dynamics." Then, in the admin console, go to "Software Update" and then upload and install the downloaded file.

**The robot's key software systems must be running the same version of software.** When updating
the robot's software the following components must also be updated:
* The robot's software
* The tablet controller app
* The Spot SDK
* Spot CAM (if present)
* Spot CORE (if present)


As of 02/16/2022, the "currently staged file" is "spot_release_v2.3.8.bde" installed on 08-09-2021


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
