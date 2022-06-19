#define PY_SSIZE_T_CLEAN

#include <iostream>
#include <vector>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <pcl_ros/point_cloud.h>
#include "graphnav_map_publisher.h"
#include "utils/c_api_utils.h"

using std::string;
using std::vector;

GraphNavMapPublisher::GraphNavMapPublisher(string map_path, string pub_topic)
    : map_path_(map_path), pub_topic_(pub_topic) {

    this->loadMap_();
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
        PyArrayObject *dataArray = obj_to_array_no_conversion(dataObj, NPY_DOUBLE);
        if (dataArray != NULL) {
            this->parse_points_array_(dataArray);
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

void GraphNavMapPublisher::parse_points_array_(PyArrayObject *dataArray) {
    auto size = PyArray_DIM(dataArray, 0);
    std::cout << size << std::endl;

    pcl::PointXYZ* points_arr = (pcl::PointXYZ*) PyArray_DATA(dataArray);
    vector<pcl::PointXYZ> point_cloud(points_arr, points_arr + size);
    this->point_cloud_ = point_cloud;
    for (pcl::PointXYZ point: this->point_cloud_) {
        std::cout << "[" << point.x << ", " << point.y << ", " << point.z << "]" << std::endl;
    }
}


int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_map_directory>" << std::endl;
        return 1;
    }
    Py_Initialize();
    init_numpy();
    GraphNavMapPublisher mapPub = GraphNavMapPublisher(argv[1], "map");
    Py_Finalize();
}
