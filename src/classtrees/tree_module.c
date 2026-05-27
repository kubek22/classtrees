#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <math.h>
#include <limits.h>

#include "tree.h"
#include "random.h"

#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>


// file for binding C with Python
// it should call only tree.c - algorithm implementation and do checks

typedef struct {
    PyObject_HEAD
    Node* root;
    size_t n_classes;  // MUST be set during fit
    size_t n_features; // optional, can be set during fit
    impurity_func_t impurity_func;
    size_t max_height;
    size_t min_samples_split;
    size_t min_samples_leaf;
    size_t max_features;
    int random_state; // negative value works like None
    pcg32_random_t rng;
} PyTree;

impurity_func_t get_impurity_func(const char* name) {
    if (strcmp(name, "gini") == 0)
        return gini_from_counts;

    if (strcmp(name, "entropy") == 0)
        return entropy_from_counts;

    PyErr_Format(PyExc_ValueError,
                 "Unknown impurity function: %s", name);
    return NULL;
}

static PyObject* PyTree_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    // allocate memory for the object and initialize fields
    PyTree *self = (PyTree*)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    self->root = NULL;
    self->impurity_func = NULL;
    self->n_classes = 0;
    self->n_features = 0;
    self->max_height = 0;
    self->min_samples_split = 0;
    self->min_samples_leaf = 0;
    self->max_features = 0;
    self->random_state = 0;
    self->rng.state = 0;
    self->rng.inc = 0;

    return (PyObject*)self;
}

static int parse_impurity(PyObject* impurity_obj, const char** out_name) {
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

static int parse_max_height(PyObject* obj, size_t* out)
{
    // Case 1: None → unbounded
    if (obj == Py_None) {
        *out = SIZE_MAX;
        return 0;
    }

    // Case 2: must be int
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "max_height must be int or None");
        return -1;
    }

    // Convert to long long safely
    long long value = PyLong_AsLongLong(obj);

    // PyLong_AsLongLong already sets overflow error if needed
    if (PyErr_Occurred()) {
        return -1;
    }

    // Validate domain
    if (value < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "max_height must be >= 0 or None");
        return -1;
    }

    // Cast safely to size_t
    *out = (size_t)value;
    return 0;
}

static int parse_min_samples_split(PyObject* obj, size_t* out)
{
    // reject None (required parameter)
    if (obj == NULL || obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_split must be int or float >= 2");
        return -1;
    }

    // reject bool (subclass of int in Python C API)
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

    // final safety check (handles 2.0 edge case safely)
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

static int parse_min_samples_leaf(PyObject* obj, size_t* out)
{
    // reject None (required parameter)
    if (obj == NULL || obj == Py_None) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_leaf must be int or float >= 1");
        return -1;
    }

    // reject bool (subclass of int in Python C API)
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

    // final safety check (handles 1.0 edge case safely)
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

static int parse_max_features(PyObject* obj, size_t* out)
{
    // None -> unlimited
    if (obj == Py_None) {
        *out = SIZE_MAX;
        return 0;
    }

    // Reject bool explicitly
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

    // Must be >= 1 BEFORE flooring/casting
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

static int parse_random_state(PyObject* obj, int* out)
{
    // None -> sentinel value
    if (obj == Py_None) {
        *out = -1;
        return 0;
    }

    // Reject bool explicitly
    if (PyBool_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "random_state must be int or None");
        return -1;
    }

    // Must be int
    if (!PyLong_Check(obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "random_state must be int or None");
        return -1;
    }

    long value = PyLong_AsLong(obj);

    // Handles overflow automatically
    if (PyErr_Occurred()) {
        return -1;
    }

    // Must be >= 0
    if (value < 0) {
        PyErr_SetString(PyExc_ValueError,
                        "random_state must be >= 0 or None");
        return -1;
    }

    // Extra safety for platforms where long > int
    if (value > INT_MAX) {
        PyErr_SetString(PyExc_OverflowError,
                        "random_state too large");
        return -1;
    }

    *out = (int)value;

    return 0;
}

static int PyTree_init(PyTree* self, PyObject* args, PyObject* kwds)
{
    // NULL for required parameters, Py_None for optional parameters with None default
    PyObject* impurity_obj = NULL;
    PyObject* max_height_obj = NULL;
    PyObject* min_samples_split_obj = NULL;
    PyObject* min_samples_leaf_obj = NULL;
    PyObject* max_features_obj = NULL;
    PyObject* random_state_obj = NULL;

    static char* kwlist[] = {
        "impurity",
        "max_height",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "random_state",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "OOOOOO",
            kwlist,
            &impurity_obj,
            &max_height_obj,
            &min_samples_split_obj,
            &min_samples_leaf_obj,
            &max_features_obj,
            &random_state_obj))
    {
        return -1;
    }

    // impurity
    const char *impurity_name = NULL;
    if (parse_impurity(impurity_obj, &impurity_name) < 0)
        return -1;
    self->impurity_func = get_impurity_func(impurity_name);
    if (!self->impurity_func) {
        PyErr_SetString(PyExc_ValueError, "unknown impurity function");
        return -1;
    }

    // max_height (None supported)
    parse_max_height(max_height_obj, &self->max_height);

    // min_samples_split
    parse_min_samples_split(min_samples_split_obj, &self->min_samples_split);

    // min_samples_leaf
    parse_min_samples_leaf(min_samples_leaf_obj, &self->min_samples_leaf);

    // max_features
    parse_max_features(max_features_obj, &self->max_features);

    // random_state
    parse_random_state(random_state_obj, &self->random_state);

    // set remaining fields to default values
    self->n_classes = 0;
    self->n_features = 0;
    self->root = NULL;

    if (self->random_state < 0) {
        // Seed with current time if no seed provided
        self->rng.state = (uint64_t)time(NULL);
    } else {
        // Seed with provided random_state
        self->rng.state = (uint64_t)self->random_state;
    }
    self->rng.inc = INC_DEFAULT; // fixed odd constant

    pcg32_random_r(&self->rng); // advance state to avoid initial low-quality output

    return 0;
}

static PyArrayObject* parse_X(PyObject* X_obj) {

    // HARD REJECTION: must be NumPy array already
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

static PyArrayObject* parse_y(PyObject* y_obj) {
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

static int check_same_n(PyArrayObject* X, PyArrayObject* y) {
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

static PyObject* fit(PyTree* self, PyObject* args) {
    if (self->root) {
        PyErr_SetString(
            PyExc_ValueError,
            "Tree is already fitted"
        );
        return NULL;
    }

    // parse and check arguments, call tree_fit
    PyObject *X_obj = NULL;
    PyObject *y_obj = NULL;

    if (!PyArg_ParseTuple(args, "OO", &X_obj, &y_obj))
        return NULL;

    PyArrayObject* X = parse_X(X_obj);

    if (!X)
        return NULL;

    PyArrayObject* y = parse_y(y_obj);

    if (!y) {
        Py_DECREF(X);
        return NULL;
    }

    if (!check_same_n(X, y)) {
        Py_DECREF(X);
        Py_DECREF(y);
        return NULL;
    }

    size_t n = (size_t)PyArray_DIM(X, 0);
    size_t p = (size_t)PyArray_DIM(X, 1);

    double* X_data = (double*)PyArray_DATA(X);

    int64_t* y_data = (int64_t*)PyArray_DATA(y);

    size_t* y_conv = malloc(n * sizeof(size_t));

    if (!y_conv) {
        Py_DECREF(X);
        Py_DECREF(y);
        PyErr_NoMemory();
        return NULL;
    }

    for (size_t i = 0; i < n; ++i)
        y_conv[i] = (size_t)y_data[i];

    self->n_classes = get_classes(y_conv, n);
    self->n_features = p;

    tree_fit(
        &self->root,
        X_data,
        y_conv,
        n,
        p,
        self->n_classes,
        self->impurity_func,
        self->max_height,
        self->min_samples_split,
        self->min_samples_leaf,
        self->max_features,
        &self->rng
    );

    free(y_conv);

    Py_DECREF(X);
    Py_DECREF(y);

    Py_RETURN_NONE;
}

static int require_fitted(PyTree* self) {
    // check if the tree is fitted (i.e. root is not NULL)
    if (!self->root) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Tree is not fitted");
        return 0;
    }
    return 1;
}

static int check_n_features(PyTree* self, PyArrayObject* X) {
    // check if the number of features in X matches the tree's n_features
    size_t p = (size_t)PyArray_DIM(X, 1);
    if (p != self->n_features) {
        PyErr_SetString(PyExc_ValueError,
                        "Number of features in X does not match the tree");
        return 0;
    }
    return 1;
}

static PyObject* predict(PyTree* self, PyObject* args) {
    // parse and check arguments, call tree_predict
    if (!require_fitted(self))
        return NULL;

    PyObject* X_obj;

    if (!PyArg_ParseTuple(args, "O", &X_obj))
        return NULL;

    PyArrayObject* X = parse_X(X_obj);

    if (!X)
        return NULL;

    if (!check_n_features(self, X)) {
        Py_DECREF(X);
        return NULL;
    }

    size_t n = (size_t)PyArray_DIM(X, 0);
    size_t p = (size_t)PyArray_DIM(X, 1);

    double* X_data = (double*)PyArray_DATA(X);

    int64_t* preds =
        tree_predict(self->root, X_data, n, p);

    Py_DECREF(X);

    if (!preds) {
        PyErr_SetString(PyExc_RuntimeError,
                        "Prediction failed");
        return NULL;
    }

    npy_intp dims[1] = {(npy_intp)n};

    PyObject* result =
        PyArray_SimpleNewFromData(
            1,
            dims,
            NPY_INT64,
            preds
        );

    if (!result) {
        free(preds);
        return NULL;
    }

    PyArray_ENABLEFLAGS(
        (PyArrayObject*)result,
        NPY_ARRAY_OWNDATA
    );

    return result;
}

static PyObject* predict_proba(PyTree* self, PyObject* args) {
    // parse and check arguments, call tree_predict_proba
    if (!require_fitted(self))
        return NULL;

    PyObject* X_obj;

    if (!PyArg_ParseTuple(args, "O", &X_obj))
        return NULL;

    PyArrayObject* X = parse_X(X_obj);

    if (!X) return NULL;

    if (!check_n_features(self, X)) {
        Py_DECREF(X);
        return NULL;
    }

    size_t n = (size_t)PyArray_DIM(X, 0);
    size_t p = (size_t)PyArray_DIM(X, 1);

    double* X_data = (double*)PyArray_DATA(X);

    size_t c = self->n_classes;

    double* probs = tree_predict_proba(
        self->root,
        X_data,
        n,
        p
    );

    Py_DECREF(X);

    if (!probs) {
        PyErr_SetString(PyExc_RuntimeError,
                        "tree_predict_proba failed");
        return NULL;
    }

    npy_intp dims[2] = {
        (npy_intp)n,
        (npy_intp)c
    };

    PyObject* result =
        PyArray_SimpleNewFromData(
            2,
            dims,
            NPY_DOUBLE,
            probs
        );

    if (!result) {
        free(probs);
        return NULL;
    }

    PyArray_ENABLEFLAGS(
        (PyArrayObject*)result,
        NPY_ARRAY_OWNDATA
    );

    return result;
}

// define methods
static PyMethodDef PyTree_methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, "Fit decision tree"},
    {"predict", (PyCFunction)predict, METH_VARARGS, "Predict labels"},
    {"predict_proba", (PyCFunction)predict_proba, METH_VARARGS, "Predict class probabilities"},
    {NULL, NULL, 0, NULL}
};

// define python-C class
static PyTypeObject PyTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "classtrees.PyTree",
    .tp_basicsize = sizeof(PyTree),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyTree_new,
    .tp_init = (initproc)PyTree_init,
    .tp_methods = PyTree_methods
};


// Python module definition

static PyModuleDef tree_module_def = {
    PyModuleDef_HEAD_INIT,
    "tree_module",
    "C tree module",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_tree_module(void) {
    import_array();  // NumPy init MUST be here

    if (PyType_Ready(&PyTreeType) < 0)
        return NULL;

    PyObject* m = PyModule_Create(&tree_module_def);
    if (!m)
        return NULL;

    Py_INCREF(&PyTreeType);
    PyModule_AddObject(m, "PyTree", (PyObject*)&PyTreeType);

    return m;
}
