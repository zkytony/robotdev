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
