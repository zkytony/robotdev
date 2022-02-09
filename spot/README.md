# Spot

Activate spot workspace:
```
# at robotdev root directory:
$ source setup_spot.bash
```

To build spot, don't run `catkin_make`. Instead,
after sourcing `setup_spot.bash`, run:

```
$ build_spot
```
If you want to build specific package(s), run
```
build_spot -DCATKIN_WHITELIST_PACKAGES="rbd_spot_robot"
```


## Troubleshooting

### Weird issue: rostopic echo doesn't work but rostopic hz works
If source ROS setup.bash, then rostopic echo doesn't work!
No idea why.

After investigation, the problem is:
```
source $repo_root/${SPOT_PATH}/devel/setup.bash
export PYTHONPATH="$repo_root/${SPOT_PATH}/venv/spot/lib/python3.8/site-packages:/usr/lib/python3/dist-packages:${PYTHONPATH}"
```
In `PYTHONPATH`, `/usr/lib/python3/dist-packages` CANNOT appear before `${PYTHONPATH}`
which after sourcing the `devel/setup.bash` contains workspace-level python configurations;
That is supposed to overwrite the system's default which is in `/usr/lib/python3/dist-packages`.
Note that `/usr/lib/python3/dist-packages` is added only so that `PyKDL` can be imported (in order to resolve [this issue](https://answers.ros.org/question/380142/how-to-install-tf2_geometry_msgs-dependency-pykdl/?answer=395887#post-id-395887).)
