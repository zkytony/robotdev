#define PY_SSIZE_T_CLEAN
#include "graphnav_map_publisher.h"
#include <Python.h>
#include <iostream>

using std::string;

GraphNavMapPublisher::GraphNavMapPublisher(string map_path, string pub_topic)
    : map_path_(map_path), pub_topic_(pub_topic) {

    // run the python code to load the map as a point cloud represented
    // as a numpy array
    Py_Initialize();
    // PyRun_SimpleString(("from rbd_spot_perception.graphnav_map import load_map_as_points\n"
    //                     "load_map_as_points(\"" + map_path_ + "\")").c_str());

    PyObject *gnmModule = PyImport_ImportModule("rbd_spot_perception.graphnav_map");
    PyObject *loadMapAsPointsFunc = PyObject_GetAttrString(gnmModule, "load_map_as_points");
    if (loadMapAsPointsFunc && PyCallable_Check(loadMapAsPointsFunc)) {
        PyObject *mapPath = PyUnicode_FromFormat(map_path.c_str());
        PyObject *pArgs = PyTuple_New(1);
        PyTuple_SetItem(pArgs, 0, mapPath);
        PyObject *dataNumpy = PyObject_CallObject(loadMapAsPointsFunc, pArgs);
    }
    std::cout << "Loading GraphNav map from: " << this->map_path_ << std::endl;
}


int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_map_directory>" << std::endl;
        return 1;
    }
    GraphNavMapPublisher mapPub = GraphNavMapPublisher(argv[1], "map");
}
