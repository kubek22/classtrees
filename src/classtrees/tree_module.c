// Main module file
// import_array() is executed only here
#define PY_ARRAY_UNIQUE_SYMBOL MyModule

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <math.h>
#include <limits.h>

#include "tree.h"
#include "random.h"
#include "checks.h"

#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>


// file for binding C with Python
// it should call only tree.c - algorithm implementation and do checks


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

static PyObject* PyForest_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    // allocate memory for the object and initialize fields
    PyForest *self = (PyForest*)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    self->roots = NULL;
    self->n_estimators = 0;
    self->impurity_func = NULL;
    self->n_classes = 0;
    self->n_features = 0;
    self->max_height = 0;
    self->min_samples_split = 0;
    self->min_samples_leaf = 0;
    self->max_features = 0;
    self->random_state = 0;
    self->rngs = NULL;
    self->n_jobs = 0;

    return (PyObject*)self;
}

static int PyTree_init(PyTree* self, PyObject* args, PyObject* kwds) {
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
    if (parse_max_height(max_height_obj, &self->max_height) < 0)
        return -1;

    // min_samples_split
    if (parse_min_samples_split(min_samples_split_obj, &self->min_samples_split) < 0)
        return -1;

    // min_samples_leaf
    if (parse_min_samples_leaf(min_samples_leaf_obj, &self->min_samples_leaf) < 0)
        return -1;

    // max_features
    if (parse_max_features(max_features_obj, &self->max_features) < 0)
        return -1;

    // random_state
    if (parse_random_state(random_state_obj, &self->random_state) < 0)
        return -1;

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

// rf
static int PyForest_init(PyForest* self, PyObject* args, PyObject* kwds) {
    // NULL for required parameters, Py_None for optional parameters with None default
    PyObject* n_estimators_obj = NULL;
    PyObject* impurity_obj = NULL;
    PyObject* max_height_obj = NULL;
    PyObject* min_samples_split_obj = NULL;
    PyObject* min_samples_leaf_obj = NULL;
    PyObject* max_features_obj = NULL;
    PyObject* random_state_obj = NULL;
    PyObject* n_jobs_obj = NULL;

    static char* kwlist[] = {
        "n_estimators",
        "impurity",
        "max_height",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "random_state",
        "n_jobs",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "OOOOOOOO",
            kwlist,
            &n_estimators_obj,
            &impurity_obj,
            &max_height_obj,
            &min_samples_split_obj,
            &min_samples_leaf_obj,
            &max_features_obj,
            &random_state_obj,
            &n_jobs_obj
        ))
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

    // n_estimators_obj (should be int > 0)
    if (parse_n_estimators(n_estimators_obj, &self->n_estimators) < 0)
        return -1;

    // max_height (None supported)
    if (parse_max_height(max_height_obj, &self->max_height) < 0)
        return -1;

    // min_samples_split
    if (parse_min_samples_split(min_samples_split_obj, &self->min_samples_split) < 0)
        return -1;

    // min_samples_leaf
    if (parse_min_samples_leaf(min_samples_leaf_obj, &self->min_samples_leaf) < 0)
        return -1;

    // max_features
    if (parse_max_features(max_features_obj, &self->max_features) < 0)
        return -1;

    // random_state
    if (parse_random_state(random_state_obj, &self->random_state) < 0)
        return -1;

    // n_jobs
    if (parse_n_jobs(n_jobs_obj, &self->n_jobs) < 0)
        return -1;

    // set remaining fields to default values
    self->n_classes = 0;
    self->n_features = 0;
    self->roots = NULL;

    uint64_t initial_random_state;
    if (self->random_state < 0) {
        // Seed with current time if no seed provided
        initial_random_state = (uint64_t)time(NULL);
    } else {
        // Seed with provided random_state
        initial_random_state = (uint64_t)self->random_state;
    }

    // creating rng objects
    self->rngs = (pcg32_random_t*)malloc(self->n_estimators * sizeof(pcg32_random_t));
    for (size_t i = 0; i < self->n_estimators; i++) {
        // we assume each tree has different initial seed, incremented by 1
        pcg32_random_t rng;
        rng.state = initial_random_state + i;
        rng.inc = INC_DEFAULT; // fixed odd constant
        pcg32_random_r(&rng); // advance state to avoid initial low-quality output
        self->rngs[i] = rng;
    }

    return 0;
}

// tree
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

static PyObject* rffit(PyForest* self, PyObject* args) {
    if (self->roots) {
        PyErr_SetString(
            PyExc_ValueError,
            "Random Forest is already fitted"
        );
        return NULL;
    }

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

    // TODO allocating memory for roots (maybe it should be somewhere else?), additional flag is_fitted may be safer
    // we store a pointer to a list of pointers to Nodes
    self->roots = (Node**)malloc(self->n_estimators * sizeof(Node*));
    for (size_t i = 0; i < self->n_estimators; i++) self->roots[i] = NULL;

    rf_fit(self->roots, self->n_estimators, X_data, y_conv,
        n, self->n_features, self->n_classes, self->impurity_func, self->max_height,
        self->min_samples_split, self->min_samples_leaf, self->max_features,
        self->rngs, self->n_jobs
    );

    free(y_conv);

    Py_DECREF(X);
    Py_DECREF(y);

    Py_RETURN_NONE;
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

static PyObject* rfpredict(PyForest* self, PyObject* args) {
    // parse and check arguments, call tree_predict
    if (!require_fitted_rf(self))
        return NULL;

    PyObject* X_obj;

    if (!PyArg_ParseTuple(args, "O", &X_obj))
        return NULL;

    PyArrayObject* X = parse_X(X_obj);

    if (!X)
        return NULL;

    if (!check_n_features_rf(self, X)) {
        Py_DECREF(X);
        return NULL;
    }

    size_t n = (size_t)PyArray_DIM(X, 0);
    size_t p = (size_t)PyArray_DIM(X, 1);

    double* X_data = (double*)PyArray_DATA(X);

    int64_t* preds = rf_predict(self->roots, self->n_estimators, X_data, n, p, self->n_classes, self->n_jobs);

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

static PyObject* rfpredict_proba(PyForest* self, PyObject* args) {
    // parse and check arguments, call tree_predict_proba
    if (!require_fitted_rf(self))
        return NULL;

    PyObject* X_obj;

    if (!PyArg_ParseTuple(args, "O", &X_obj))
        return NULL;

    PyArrayObject* X = parse_X(X_obj);

    if (!X) return NULL;

    if (!check_n_features_rf(self, X)) {
        Py_DECREF(X);
        return NULL;
    }

    size_t n = (size_t)PyArray_DIM(X, 0);
    size_t p = (size_t)PyArray_DIM(X, 1);

    double* X_data = (double*)PyArray_DATA(X);

    size_t c = self->n_classes;

    double* probs = rf_predict_proba(self->roots, self->n_estimators, X_data, n, p, self->n_classes, self->n_jobs);

    Py_DECREF(X);

    if (!probs) {
        PyErr_SetString(PyExc_RuntimeError,
                        "rf_predict_proba failed");
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


// define methods for tree
static PyMethodDef PyTree_methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, "Fit decision tree"},
    {"predict", (PyCFunction)predict, METH_VARARGS, "Predict labels"},
    {"predict_proba", (PyCFunction)predict_proba, METH_VARARGS, "Predict class probabilities"},
    {NULL, NULL, 0, NULL}
};

// define methods for rf
static PyMethodDef PyForest_methods[] = {
    {"rffit", (PyCFunction)rffit, METH_VARARGS, "Fit Random Forest"},
    {"rfpredict", (PyCFunction)rfpredict, METH_VARARGS, "Predict labels"},
    {"rfpredict_proba", (PyCFunction)rfpredict_proba, METH_VARARGS, "Predict class probabilities"},
    {NULL, NULL, 0, NULL}
};

// define python-C class for tree
static PyTypeObject PyTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "classtrees.PyTree",
    .tp_basicsize = sizeof(PyTree),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyTree_new,
    .tp_init = (initproc)PyTree_init,
    .tp_methods = PyTree_methods
};

// define python-C class for rf
static PyTypeObject PyForestType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "classtrees.PyForest",
    .tp_basicsize = sizeof(PyForest),
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyForest_new,
    .tp_init = (initproc)PyForest_init,
    .tp_methods = PyForest_methods
};


// Python module definition (just once)
static PyModuleDef tree_module_def = {
    PyModuleDef_HEAD_INIT,
    "tree_module",
    "C tree module",
    -1,
    NULL
};

PyMODINIT_FUNC PyInit_tree_module(void) {
    import_array();  // NumPy init MUST be here

    // tree
    if (PyType_Ready(&PyTreeType) < 0)
        return NULL;

    // rf
    if (PyType_Ready(&PyForestType) < 0)
        return NULL;

    PyObject* m = PyModule_Create(&tree_module_def);
    if (!m)
        return NULL;

    Py_INCREF(&PyTreeType);
    Py_INCREF(&PyForestType);

    PyModule_AddObject(m, "PyTree", (PyObject*)&PyTreeType);
    PyModule_AddObject(m, "PyForest", (PyObject*)&PyForestType);

    return m;
}
