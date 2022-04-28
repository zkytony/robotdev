# Testing Viam Python SDK

**Note** that due to Viam's bug in backwards compatibility,
code within this folder requires Python 3.9+.

**Note** that when working with Viam, we DO NOT assume
access to any ROS stuff. It is just pure python plus
Viam's Python SDK.

## Setup
1. Get Python 3.9
   ```
   cd spot/viam
   source get_python3.9.sh
   ```
   If successful, then you should be able to type `python3.9` in the command line and see
   ```
   Python 3.9.12 (main, Apr 28 2022, 02:39:45)
   [GCC 9.3.0] on linux
   Type "help", "copyright", "credits" or "license" for more information.
   >>>
   ```

## Run test

- test 1:

    ```
    cd robotdev/spot/viam
    python -m camera.spot_camera
    ```
