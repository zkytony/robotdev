If you would like to stream the fiducial marker positions of the robot as transforms,
```
python stream_fiducial_markers.py
```

There will be two TF frames related to fiducial marker detection:
"filtered_fiducial_xxx" will persist and shakes (if the robot's body moves),
while "fiducial_xxx" will only appear when the fiducial is detected, and is more stable.
Although, random jumps or changes do occur.
