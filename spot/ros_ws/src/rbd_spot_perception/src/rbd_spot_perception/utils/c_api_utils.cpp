#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "utils/c_api_utils.h"


// converts PyObject to PyArrayObject (a numpy type)
PyArrayObject* obj_to_array_no_conversion(PyObject* input, int typecode) {
    PyArrayObject* ary = NULL;
    if (is_array(input) && (typecode == PyArray_NOTYPE ||
                            PyArray_EquivTypenums(array_type(input),
                                                  typecode))) {
        ary = (PyArrayObject*) input;
    } else if (is_array(input)) {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type = typecode_string(array_type(input));
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required.  Array of type '%s' given",
                     desired_type, actual_type);
        ary = NULL;
    } else {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type = typecode_string(input);
        PyErr_Format(PyExc_TypeError,
                     "Array of type '%s' required.  Array of type '%s' given",
                     desired_type, actual_type);
        ary = NULL;
    }
    return ary;
}
