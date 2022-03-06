# Spot Navigation

Because Spot can handle placement of its feet, achieving navigation can be done in possibly the following ways:

1. Use move_base and pretend the robot is a wheeled robot - although move_base is well-supported, the output velocity commands may not be suitable for Spot (it might be though because technically you just need to control the linear velocities).
2. Use the Autowalk feature (somehow) programmatically
3. Do it yourself - because we have access to the grid map, then given a global pose, it should be straightforward (but not trivial) to implement a global 
   path planner, and then code up a program that sends velocity commands intermittedly to make the robot follow the global path. This is hacky and risky.
4. Use some library that supports quadruped navigation? ([towr](http://wiki.ros.org/towr) seemed like a promising choice, but it assumes your robot can't do
   gaits itself and it is not available for ROS noetic).


## Spot SDK GraphNav

Spot SDK provides [the GraphNav service](https://dev.bostondynamics.com/docs/concepts/autonomy/graphnav_service).

You can record a map "using the GraphNavRecording service".


Spot SDK's GraphNav is intended for autonomous site inspection;
basically, the robot would autonomously navigate from waypoint
to waypoint, designed for the specific site.
