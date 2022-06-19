#define PY_SSIZE_T_CLEAN
#include "graphnav_map_publisher.h"
#include <Python.h>

using std::string;

GraphNavMapPublisher::GraphNavMapPublisher(string map_path, string pub_topic)
    : map_path_(map_path), pub_topic_(pub_topic) {

}
