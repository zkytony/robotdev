# MOVO URDF

* MOVO's URDF is stored at `movo_ws/src/kinova-movo/movo_common/movo_description/urdf/movo.urdf.xacro`.
* The URDF is loaded during `roslaunch movo_bringup movo_system.launch` through `movo2.launch` through `movo_upload.launch`
* The `movo.urdf.xacro` file contains many checks for environment variables. The command `optenv ENV_VARIABVLE default_value` returrns the `ENV_VAIABLE`'s value (or default).
* The environment vaiables used in `movo.urrdf.xacro` is defined under `movo_common/movo_config/movo_config.bash`.
