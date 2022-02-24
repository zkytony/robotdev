# Mapping

 I would like to try out [ORB_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3),
 a very recent SLAM library. The alternative is [rtabmap](https://github.com/introlab/rtabmap)
 and [cartographer](https://google-cartographer.readthedocs.io/en/latest/).

## Which package to use?

[ORB_SLAM's original paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7219438) has been probably the most popular
SLAM-package paper. ORB_SLAM3, despite released only in 2020, has
gained wild popularity on github (3.4k stars vs. 1.7k for [rtabmap](https://github.com/introlab/rtabmap)).
So, if you use any other package, you will be left wondering,
"what if I use ORB_SLAM?"

ORB_SLAM3's github says: "ORB-SLAM3 is the **first real-time SLAM library** able to
perform Visual, Visual-Inertial and Multi-Map SLAM with monocular, stereo and
RGB-D cameras, using pin-hole and fisheye lens models. In all sensor
configurations, **ORB-SLAM3 is as robust as the best systems available in the**
**literature, and significantly more accurate.**

There is no other choice than ORB_SLAM3.

Cartographer is not considered because it does not provide RGBD SLAM.
