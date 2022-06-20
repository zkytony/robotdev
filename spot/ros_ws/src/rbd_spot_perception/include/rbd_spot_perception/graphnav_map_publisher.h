#ifndef GRAPHNAV_MAP_PUBLISHER_H
#define GRAPHNAV_MAP_PUBLISHER_H

#include <string>
#include <vector>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pcl_ros/point_cloud.h>

using std::string;
using std::vector;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

// Loads a GraphNav map as point cloud and
// publishes the point cloud as a ROS message
class GraphNavMapPublisher {
public:
    GraphNavMapPublisher(string map_path);
    void run();
private:
    ros::NodeHandle nh_;
    ros::Publisher pcl_pub_;
    string pcl_frame_id_;
    double pcl_rate_;
    string map_path_;
    string pub_topic_;
    PointCloud cloud_;

    void loadMap_();
    void parsePointsArray_(PyArrayObject *dataArray);
};

#endif
