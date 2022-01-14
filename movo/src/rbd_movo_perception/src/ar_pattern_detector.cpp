/**
 * Given a specification of (multi)AR-tag, publish:
 * - Whether the robot has detected the specified pattern
 * - Properties of interest (also specified), such as average
 *   3D pose, surface norm, co-linear, etc.
 *
 * The specification is passed in as a yaml file.
 *
 * The AR tag detector assumed to be used is ar_track_alvar.
 * It is straightforward to change this, as long as the detected
 * 3D pose, frame, and AR code ID can be obtained.
 */

#include <ros/ros.h>
#include <std_msgs/String.h>
// requires installation of ros-kinetic-ar-track-alvar
#include <ar_track_alvar_msgs/AlvarMarkers.h>

#include <sstream>
#include <string>
#include <vector>

using namespace ar_track_alvar_msgs;

void ARMarkerCallback(const AlvarMarkers::ConstPtr& msg) {
  std::vector<AlvarMarker> markers = msg->markers;
  for (int i = 0; i < markers.size(); i++) {
    ROS_INFO("ID: [%d]", markers[i].id);
    ROS_INFO_STREAM(markers[i]);
  }
}

int main(int argc, char** argv) {
  // TODO: expand this.
  // For now, we will code up a basic publisher that publishes
  // 'yes' if the node detects a single AR-tag that is specified
  // by a parameter.
  ros::init(argc, argv, "artag_detector");
  ros::NodeHandle node;

  std::string ar_marker_topic_;
  node.param("ar_marker_topic", ar_marker_topic_, std::string("/ar_pose_marker"));

  // ros::Publisher detector_pub = node.advertise<std_msgs::String>("artag_detections", 100);
  ros::Subscriber sub = node.subscribe(ar_marker_topic_, 100, ARMarkerCallback);
  // ros::Rate loop_rate(5);  //5hz

  // while (ros::ok()) {
  //   std_msgs::String msg;

  //   detector_pub.publish(msg);
  //   ros::spinOnce();
  //   loop_rate.sleep();
  // }
  ros::spin();
  return 0;
}
