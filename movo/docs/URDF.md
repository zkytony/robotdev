# MOVO URDF

* MOVO's URDF is stored at `movo_ws/src/kinova-movo/movo_common/movo_description/urdf/movo.urdf.xacro`.
* The URDF is loaded during `roslaunch movo_bringup movo_system.launch` through `movo2.launch` through `movo_upload.launch`
* The `movo.urdf.xacro` file contains many checks for environment variables. The command `optenv ENV_VARIABVLE default_value` returrns the `ENV_VAIABLE`'s value (or default).
* The environment variables used in `movo.urrdf.xacro` are defined under `movo_common/movo_config/movo_config.bash`.
* By default, if you set `MOVO_HAS_TWO_KINOVA_ARMS` to be `false`, then it assumes you have the right arm attached (which many not be always desirable). So, I suggest even if you have only the left arm attached, you should leave this variable to be true.
   * In fact, somebody (as of 01/12/2022) had added a line in `movo_config.bash` to hack this:
     ```
     export KINOVA_RIGHT_ARM_IP_ADDRESS=10.66.171.16  # NOTE: hack to make movo boot while right arm nonresponsive
     ```
