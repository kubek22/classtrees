// helper module, it is not required to run import_array()
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MyModule

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

int parse_n_estimators(PyObject* obj, size_t* out) {
    // reject None
    if (obj == NULL || obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "n_estimators must be int >= 1");
        return -1;
    }

    // IMPORTANT: reject bool (bool is subclass of int)
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "n_estimators must be int, not bool");
        return -1;
    }

    // must be int
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "n_estimators must be int");
        return -1;
    }

    long long value = PyLong_AsLongLong(obj);

    if (PyErr_Occurred()) {
        return -1;
    }

    if (value < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "n_estimators must be >= 1");
        return -1;
    }

    *out = (size_t)value;
    return 0;
}

int parse_max_height(PyObject* obj, size_t* out) {
    // None -> infinity
    if (obj == Py_None) {
        *out = SIZE_MAX;
        return 0;
    }

    // reject bool explicitly (critical)
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "max_height must be int or None, not bool");
        return -1;
    }

    // must be exact Python int
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "max_height must be int or None");
        return -1;
    }

    PyObject* tmp = PyNumber_Long(obj);
    if (tmp == NULL)
        return -1;

    long long value = PyLong_AsLongLong(tmp);
    Py_DECREF(tmp);

    if (PyErr_Occurred())
        return -1;

    if (value < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "max_height must be >= 0 or None");
        return -1;
    }

    *out = (size_t)value;
    return 0;
}

int parse_min_samples_split(PyObject* obj, size_t* out) {
    if (obj == NULL || obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_split must be int");
        return -1;
    }

    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_split must be int");
        return -1;
    }

    // direct conversion to size_t (correct API)
    size_t value = PyLong_AsSize_t(obj);

    if (PyErr_Occurred())
        return -1;

    if (value < 2) {
        PyErr_SetString(PyExc_ValueError,
                        "min_samples_split must be >= 2");
        return -1;
    }

    *out = value;
    return 0;
}

int parse_min_samples_leaf(PyObject* obj, size_t* out) {
    // reject None
    if (obj == NULL || obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_leaf must be int");
        return -1;
    }

    // reject bool (bool is subclass of int)
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_leaf must be int, not bool");
        return -1;
    }

    // only accept integers
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_leaf must be int");
        return -1;
    }

    size_t value = PyLong_AsSize_t(obj);

    if (PyErr_Occurred())
        return -1;

    if (value < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "min_samples_leaf must be >= 1");
        return -1;
    }

    *out = value;
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
                        "max_features must be int or None");
        return -1;
    }

    // only accept ints
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "max_features must be int or None");
        return -1;
    }

    size_t value = PyLong_AsSize_t(obj);

    if (PyErr_Occurred())
        return -1;

    if (value < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "max_features must be >= 1");
        return -1;
    }

    *out = value;
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

int parse_n_jobs(PyObject* obj, int* out) {
    // reject None
    if (obj == NULL || obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "n_jobs must be int (-1 or >= 1)");
        return -1;
    }

    // reject bool explicitly
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "n_jobs must be int (-1 or >= 1), not bool");
        return -1;
    }

    // must be int
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "n_jobs must be int");
        return -1;
    }

    long long value = PyLong_AsLongLong(obj);

    if (PyErr_Occurred()) {
        return -1;
    }

    // valid: -1 or >= 1
    if (value != -1 && value < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "n_jobs must be -1 or >= 1");
        return -1;
    }

    // safe cast (int is correct type here)
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

PyArrayObject* parse_X(PyObject* X_obj) {

    // must be NumPy array already
    if (!PyArray_Check(X_obj)) {
        PyErr_SetString(
            PyExc_TypeError,
            "X must be a numpy.ndarray"
        );
        return NULL;
    }

    PyArrayObject* X =
        (PyArrayObject*)PyArray_FROM_OTF(
            X_obj,
            NPY_DOUBLE,
            NPY_ARRAY_C_CONTIGUOUS |
            NPY_ARRAY_ALIGNED
        );

    if (!X) {
        PyErr_SetString(
            PyExc_TypeError,
            "X must be convertible to float64 ndarray"
        );
        return NULL;
    }

    if (PyArray_NDIM(X) != 2) {
        Py_DECREF(X);
        PyErr_SetString(
            PyExc_ValueError,
            "X must be 2D"
        );
        return NULL;
    }

    npy_intp n = PyArray_DIM(X, 0);
    npy_intp p = PyArray_DIM(X, 1);

    if (n <= 0 || p <= 0) {
        Py_DECREF(X);
        PyErr_SetString(
            PyExc_ValueError,
            "X cannot be empty"
        );
        return NULL;
    }

    double* data = (double*)PyArray_DATA(X);
    npy_intp total = PyArray_SIZE(X);

    for (npy_intp i = 0; i < total; i++) {
        if (isnan(data[i]) || isinf(data[i])) {
            Py_DECREF(X);
            PyErr_SetString(
                PyExc_ValueError,
                "X contains NaN or Inf"
            );
            return NULL;
        }
    }

    return X;
}

PyArrayObject* parse_y(PyObject* y_obj) {
    // unpack and check y - it should be 1D array of non-negative integers
    PyArrayObject* y = (PyArrayObject*)PyArray_FROM_OTF(
        y_obj,
        NPY_INT64,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
    );

    if (!y)
        return NULL;

    if (PyArray_NDIM(y) != 1) {
        Py_DECREF(y);
        PyErr_SetString(PyExc_ValueError,
                        "y must be 1D");
        return NULL;
    }

    if (PyArray_SIZE(y) == 0) {
        Py_DECREF(y);
        PyErr_SetString(PyExc_ValueError,
                        "y cannot be empty");
        return NULL;
    }

    int64_t* data = (int64_t*)PyArray_DATA(y);

    npy_intp n = PyArray_SIZE(y);

    for (npy_intp i = 0; i < n; i++) {
        if (data[i] < 0) {
            Py_DECREF(y);
            PyErr_SetString(PyExc_ValueError,
                            "y cannot contain negative classes");
            return NULL;
        }
    }

    return y;
}

// for tree
int require_fitted(PyTree* self) {
    // check if the tree is fitted (i.e. root is not NULL)
    if (!self->root) {
        PyErr_SetString(PyExc_RuntimeError,
                        "The Tree is not fitted");
        return 0;
    }
    return 1;
}

int check_n_features(PyTree* self, PyArrayObject* X) {
    // check if the number of features in X matches the tree's n_features
    size_t p = (size_t)PyArray_DIM(X, 1);
    if (p != self->n_features) {
        PyErr_Format(PyExc_ValueError,
             "Number of features in X does not match expected %zu",
             self->n_features);
        return 0;
    }
    return 1;
}

int require_fitted_rf(PyForest* self) {
    if (!self || !self->roots) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Random Forest is not fitted");
        return 0;
    }

    // optional stronger check: at least one tree exists
    for (size_t i = 0; i < self->n_estimators; i++) {
        if (self->roots[i] == NULL) {
            PyErr_SetString(PyExc_RuntimeError,
                        "One of estimators in Random Forest is not fitted");
            return 0;
        }
    }
    return 1;
}

int check_n_features_rf(PyForest* self, PyArrayObject* X) {
    size_t p = (size_t)PyArray_DIM(X, 1);

    if (p != self->n_features) {
        PyErr_Format(PyExc_ValueError,
            "Number of features in X does not match expected %zu",
            self->n_features);
        return 0;
    }

    return 1;
}

