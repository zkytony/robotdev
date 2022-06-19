#ifndef GRAPHNAV_MAP_PUBLISHER_H
#define GRAPHNAV_MAP_PUBLISHER_H

#include <string>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pcl_ros/point_cloud.h>

using std::string;

// Loads a GraphNav map as point cloud and
// publishes the point cloud as a ROS message
class GraphNavMapPublisher {
public:
    GraphNavMapPublisher(string map_path, string pub_topic);

private:
    string map_path_;
    string pub_topic_;
    pcl::PointCloud<pcl::PointXYZ> *point_cloud_;

    void loadMap_();
    void parse_points_array_(PyArrayObject *dataArray);
};

#endif
