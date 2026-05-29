#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "checks.h"

#include <math.h>
#include <limits.h>
#include <string.h>

// implementation of the functions for checking and parsing parameters


impurity_func_t get_impurity_func(const char* name) {
    if (strcmp(name, "gini") == 0)
        return gini_from_counts;

    if (strcmp(name, "entropy") == 0)
        return entropy_from_counts;

    PyErr_Format(PyExc_ValueError,
                 "Unknown impurity function: %s", name);
    return NULL;
}

int parse_impurity(PyObject* impurity_obj, const char** out_name) {
    if (!PyUnicode_Check(impurity_obj)) {
        PyErr_SetString(PyExc_TypeError, "impurity must be a string");
        return -1;
    }

    const char* name = PyUnicode_AsUTF8(impurity_obj);
    if (!name) {
        return -1;  // PyUnicode_AsUTF8 already sets error
    }

    *out_name = name;
    return 0;
}

int parse_max_height(PyObject* obj, size_t* out) {
    // None -> infinity
    if (obj == Py_None) {
        *out = SIZE_MAX;
        return 0;
    }

    // must be int
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "max_height must be int or None");
        return -1;
    }

    // convert to long long
    long long value = PyLong_AsLongLong(obj);

    // PyLong_AsLongLong already sets overflow error if needed
    if (PyErr_Occurred()) {
        return -1;
    }

    // non-negative check
    if (value < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "max_height must be >= 0 or None");
        return -1;
    }

    // Cast to size_t
    *out = (size_t)value;
    return 0;
}

int parse_min_samples_split(PyObject* obj, size_t* out) {
    // reject None
    if (obj == NULL || obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_split must be int or float >= 2");
        return -1;
    }

    // reject bool
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_split must be int or float, not bool");
        return -1;
    }

    double value;

    // int case
    if (PyLong_Check(obj)) {
        value = (double)PyLong_AsLongLong(obj);
        if (PyErr_Occurred())
            return -1;
    }
    // float case
    else if (PyFloat_Check(obj)) {
        value = PyFloat_AsDouble(obj);
        if (PyErr_Occurred())
            return -1;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_split must be int or float");
        return -1;
    }

    // domain check before conversion
    if (value < 2.0) {
        PyErr_SetString(PyExc_ValueError,
                        "min_samples_split must be >= 2");
        return -1;
    }

    // floor for floats
    double floored = floor(value);

    // 2.0 edge case
    if (floored < 2.0) {
        PyErr_SetString(PyExc_ValueError,
                        "min_samples_split must be >= 2 after flooring");
        return -1;
    }

    if (floored > (double)SIZE_MAX) {
        PyErr_SetString(PyExc_OverflowError,
                        "min_samples_split too large");
        return -1;
    }

    *out = (size_t)floored;
    return 0;
}

int parse_min_samples_leaf(PyObject* obj, size_t* out) {
    // reject None
    if (obj == NULL || obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_leaf must be int or float >= 1");
        return -1;
    }

    // reject bool
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_leaf must be int or float, not bool");
        return -1;
    }

    double value;

    // int case
    if (PyLong_Check(obj)) {
        value = (double)PyLong_AsLongLong(obj);
        if (PyErr_Occurred())
            return -1;
    }
    // float case
    else if (PyFloat_Check(obj)) {
        value = PyFloat_AsDouble(obj);
        if (PyErr_Occurred())
            return -1;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_leaf must be int or float");
        return -1;
    }

    // domain check before conversion
    if (value < 1.0) {
        PyErr_SetString(PyExc_ValueError,
                        "min_samples_leaf must be >= 1");
        return -1;
    }

    // floor for floats
    double floored = floor(value);

    // handles 1.0 edge case
    if (floored < 1.0) {
        PyErr_SetString(PyExc_ValueError,
                        "min_samples_leaf must be >= 1 after flooring");
        return -1;
    }

    if (floored > (double)SIZE_MAX) {
        PyErr_SetString(PyExc_OverflowError,
                        "min_samples_leaf too large");
        return -1;
    }

    *out = (size_t)floored;
    return 0;
}

int parse_max_features(PyObject* obj, size_t* out) {
    // None -> infinity
    if (obj == Py_None) {
        *out = SIZE_MAX;
        return 0;
    }

    // reject bool 
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "max_features must be int, float, or None");
        return -1;
    }

    double value;

    // int case
    if (PyLong_Check(obj)) {
        value = (double)PyLong_AsLongLong(obj);

        if (PyErr_Occurred())
            return -1;
    }
    // float case
    else if (PyFloat_Check(obj)) {
        value = PyFloat_AsDouble(obj);

        if (PyErr_Occurred())
            return -1;
    }
    else {
        PyErr_SetString(PyExc_TypeError,
                        "max_features must be int, float, or None");
        return -1;
    }

    // must be >= 1 before flooring/casting
    if (value < 1.0) {
        PyErr_SetString(PyExc_ValueError,
                        "max_features must be >= 1");
        return -1;
    }

    // floor semantics
    double floored = floor(value);

    // overflow protection
    if (floored > (double)SIZE_MAX) {
        PyErr_SetString(PyExc_OverflowError,
                        "max_features too large");
        return -1;
    }

    *out = (size_t)floored;

    return 0;
}

int parse_random_state(PyObject* obj, int* out) {
    // None -> -1
    if (obj == Py_None) {
        *out = -1;
        return 0;
    }

    // reject bool
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "random_state must be int or None");
        return -1;
    }

    // must be int
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "random_state must be int or None");
        return -1;
    }

    long value = PyLong_AsLong(obj);

    // handles overflow automatically
    if (PyErr_Occurred()) {
        return -1;
    }

    // must be >= 0
    if (value < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "random_state must be >= 0 or None");
        return -1;
    }

    // safety check for platforms where long > int
    if (value > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError,
                        "random_state too large");
        return -1;
    }

    *out = (int)value;

    return 0;
}

int check_same_n(PyArrayObject* X, PyArrayObject* y) {
    // check if X and y have the same number of samples
    npy_intp nX = PyArray_DIM(X, 0);
    npy_intp ny = PyArray_DIM(y, 0);

    if (nX != ny) {
        PyErr_SetString(PyExc_ValueError,
                        "X and y have inconsistent lengths");
        return 0;
    }

    return 1;
}
