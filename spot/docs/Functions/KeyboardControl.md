# Keyboard Control

You can run the following script:
```
rosrun rbd_spot_action keyboard_control.py
```
You can press `h` for help.

**Note** that this script uses the `/spot/cmd_vel` topic subscribed by spot ros.
It requires that you have driver running in control mode, which will acquire
the control authority.
```
roslaunch rbd_spot_robot driver.launch control:=true
```
