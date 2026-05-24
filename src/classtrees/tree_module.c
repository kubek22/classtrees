#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <math.h>

#include "tree.h"

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

    return (PyObject*)self;
}

static int PyTree_init(PyTree* self, PyObject* args, PyObject* kwds)
{
    PyObject* impurity_obj = NULL;
    PyObject* max_height_obj = Py_None;
    PyObject* min_samples_split_obj = NULL;

    static char* kwlist[] = {
        "impurity",
        "max_height",
        "min_samples_split",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "OOO",
            kwlist,
            &impurity_obj,
            &max_height_obj,
            &min_samples_split_obj))
    {
        return -1;
    }

    /* impurity */
    const char* impurity_name = PyUnicode_AsUTF8(impurity_obj);
    if (!impurity_name) {
        PyErr_SetString(PyExc_TypeError,
                        "impurity must be a string");
        return -1;
    }

    /* max_height (None supported) */
    size_t max_height;

    if (max_height_obj == Py_None) {
        max_height = SIZE_MAX;
    } else if (PyLong_Check(max_height_obj)) {
        long tmp = PyLong_AsLong(max_height_obj);
        if (PyErr_Occurred())
            return -1;

        if (tmp < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "max_height must be >= 0 or None");
            return -1;
        }

        max_height = (size_t)tmp;
    } else {
        PyErr_SetString(PyExc_TypeError,
                        "max_height must be int or None");
        return -1;
    }

    /* min_samples_split */
    if (!PyLong_Check(min_samples_split_obj)) {
        PyErr_SetString(PyExc_TypeError,
                        "min_samples_split must be int");
        return -1;
    }

    long mss = PyLong_AsLong(min_samples_split_obj);
    if (PyErr_Occurred())
        return -1;

    if (mss < 1) {
        PyErr_SetString(PyExc_ValueError,
                        "min_samples_split must be >= 1");
        return -1;
    }

    /* init impurity */
    self->impurity_func = get_impurity_func(impurity_name);
    if (!self->impurity_func)
        return -1;

    self->max_height = max_height;
    self->min_samples_split = (size_t)mss;

    self->n_classes = 0;
    self->n_features = 0;
    self->root = NULL;

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
        self->min_samples_split
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
