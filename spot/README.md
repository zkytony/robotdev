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
