If you would like to stream the fiducial marker positions of the robot as transforms, first stream all the camera images and poses from the Spot body:

rosrun rbd_spot_perception stream_body_camerasets.sh

Next, you can now run the rosnode that gets the pose of the fiducial marker relative to the spot, and publishes as a transform to tf:
roscd rbd_spot_perception
python stream_fiducial_markers.py [ROBOT_IP] --user user --password [SPOT_USER_PASSWORD]
