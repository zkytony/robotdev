#define PY_SSIZE_T_CLEAN

// standard libs
#include <iostream>
#include <vector>

// python and numpy C APIs
#include <Python.h>
#include <numpy/arrayobject.h>

// ros and PCL
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

// our headers
#include "graphnav_map_publisher.h"
#include "utils/c_api_utils.h"

using std::string;
using std::vector;

GraphNavMapPublisher::GraphNavMapPublisher(string map_path)
    : map_path_(map_path), nh_(ros::NodeHandle("~")) {
    loadMap_();

    pub_topic_ = "graphnav_map_points";
    if (nh_.hasParam("topic")) {
        nh_.getParam("topic", pub_topic_);
    }

    pcl_frame_id_ = "graphnav_map";
    if (nh_.hasParam("frame_id")) {
        nh_.getParam("frame_id", pcl_frame_id_);
    }

    pcl_rate_ = 4.0;
    if (nh_.hasParam("rate")) {
        nh_.getParam("rate", pcl_rate_);
    }

    pcl_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(pub_topic_, 10);
}


/* Loads the map and gets the numpy array object */
void GraphNavMapPublisher::loadMap_() {
    // run the python code to load the map as a point cloud represented
    // as a numpy array
    PyObject *gnmModule = PyImport_ImportModule("rbd_spot_perception.graphnav_map");
    PyObject *loadMapAsPointsFunc = PyObject_GetAttrString(gnmModule, "load_map_as_points");
    PyObject *mapPath = PyUnicode_FromFormat(this->map_path_.c_str());
    PyObject *pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, mapPath);
    std::cout << "Loading GraphNav map from: " << this->map_path_ << std::endl;
    PyObject *dataObj = PyObject_CallObject(loadMapAsPointsFunc, pArgs);

    if (dataObj != NULL) {
        PyArrayObject *dataArray = obj_to_array_no_conversion(dataObj, NPY_FLOAT);
        if (dataArray != NULL) {
            this->parsePointsArray_(dataArray);
        } else {
            goto fail;
        }
    } else {
        goto fail;
    }
    return;

    fail:
        Py_DECREF(loadMapAsPointsFunc);
        Py_DECREF(gnmModule);
        PyErr_Print();
        fprintf(stderr,"Call failed\n");
}

/* given the numpy data array returned by the load_map_as_points function,
 * converts it into a vector<pcl::PointXYZ> object to be easier to work with. */
void GraphNavMapPublisher::parsePointsArray_(PyArrayObject *dataArray) {
    auto size = PyArray_DIM(dataArray, 0);
    pcl::PointXYZ* points_arr = (pcl::PointXYZ*) PyArray_DATA(dataArray);
    for (int i=0; i<size; i++) {
        this->cloud_.push_back(points_arr[i]);
    }
    std::cout << "Loaded " << this->cloud_.points.size() << " points." << std::endl;
}

void GraphNavMapPublisher::run() {

    sensor_msgs::PointCloud2 pcl_msg;
    pcl::toROSMsg(this->cloud_, pcl_msg);
    pcl_msg.header.frame_id = this->pcl_frame_id_;
    ros::Rate loop_rate(this->pcl_rate_);
    while (this->nh_.ok()) {
        pcl_msg.header.stamp = ros::Time::now();
        this->pcl_pub_.publish(pcl_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_map_directory>" << std::endl;
        return 1;
    }
    Py_Initialize();
    init_numpy();
    ros::init(argc, argv, "graphnav_map_publisher");
    GraphNavMapPublisher mapPub = GraphNavMapPublisher(argv[1]);
    mapPub.run();
    Py_Finalize();
}
