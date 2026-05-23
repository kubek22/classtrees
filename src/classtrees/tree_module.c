#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include "tree.h"

#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h>


// file for binding C with Python
// it should call only tree.c - algorithm implementation and do checks

typedef struct {
    PyObject_HEAD
    Node* root;
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
    PyTree *self = (PyTree*)type->tp_alloc(type, 0);
    if (!self)
        return NULL;

    self->root = NULL;
    self->impurity_func = NULL;
    self->max_height = 0;
    self->min_samples_split = 0;

    return (PyObject*)self;
}

static int PyTree_init(PyTree* self, PyObject* args, PyObject* kwds) {
    const char* impurity_name = NULL;
    size_t max_height = 0;
    size_t min_samples_split = 0;

    static char* kwlist[] = {
        "impurity",
        "max_height",
        "min_samples_split",
        NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args, kwds,
            "snn",
            kwlist,
            &impurity_name,
            &max_height,
            &min_samples_split))
    {
        return -1;
    }

    self->impurity_func = get_impurity_func(impurity_name);

    if (!self->impurity_func) {
        return -1;  // PyErr already set in get_impurity_func
    }

    self->max_height = max_height;
    self->min_samples_split = min_samples_split;

    return 0;
}

static PyObject* fit(PyTree* self, PyObject* args) {
    PyObject *X_obj = NULL;
    PyObject *y_obj = NULL;

    if (!PyArg_ParseTuple(args, "OO", &X_obj, &y_obj)) {
        return NULL;
    }

    // Convert X to C contiguous NumPy array (double)
    PyArrayObject *X = (PyArrayObject*) PyArray_FROM_OTF(
        X_obj,
        NPY_DOUBLE,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
    );

    if (!X) return NULL;

    // Convert y to C contiguous NumPy array (size_t-compatible)
    PyArrayObject *y = (PyArrayObject*) PyArray_FROM_OTF(
        y_obj,
        NPY_UINT64,   // safest proxy for size_t
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
    );

    if (!y) {
        Py_DECREF(X);
        return NULL;
    }

    // Extract dimensions
    npy_intp *dims = PyArray_DIMS(X);
    size_t n = (size_t)dims[0];
    size_t p = (size_t)dims[1];

    // Raw pointers
    double *X_data = (double*)PyArray_DATA(X);
    size_t *y_data = (size_t*)PyArray_DATA(y);

    // TODO optionally check if y has the same number of samples as X
    // TODO check if y is 1D and X is 2D
    // TODO check if classes are positive integers and if there are at least 2 classes

    // Getting the number of classes
    size_t c = get_classes(y_data, n);

    // Call C implementation
    tree_fit(
        &self->root,
        X_data,
        y_data,
        n,
        p,
        c,
        self->impurity_func,
        self->max_height,
        self->min_samples_split
    );

    // Cleanup
    Py_DECREF(X);
    Py_DECREF(y);

    Py_RETURN_NONE;
}

// TODO implement predict method
static PyObject* predict(PyTree* self, PyObject* args) {
    PyObject* X_obj;

    if (!PyArg_ParseTuple(args, "O", &X_obj)) {
        return NULL;
    }

    // Ensure NumPy array (double, C-contiguous)
    PyArrayObject* X = (PyArrayObject*)PyArray_FROM_OTF(
        X_obj,
        NPY_DOUBLE,
        NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
    );

    if (!X) {
        return NULL;
    }

    // shape extraction
    npy_intp* dims = PyArray_DIMS(X);
    size_t n = (size_t)dims[0];
    size_t p = (size_t)dims[1];

    double* X_data = (double*)PyArray_DATA(X);

    // call C function
    size_t* preds = tree_predict(
        self->root,
        X_data,
        n,
        p
    );

    if (!preds) {
        Py_DECREF(X);
        PyErr_SetString(PyExc_RuntimeError, "tree_predict failed");
        return NULL;
    }

    // convert result to NumPy array
    npy_intp out_dims[1] = { (npy_intp)n };

    PyObject* result = PyArray_SimpleNewFromData(
        1,
        out_dims,
        NPY_UINT64,
        preds
    );

    if (!result) {
        free(preds);
        Py_DECREF(X);
        return NULL;
    }

    // IMPORTANT: make NumPy own memory
    PyArrayObject* arr = (PyArrayObject*)result;
    PyArray_ENABLEFLAGS(arr, NPY_ARRAY_OWNDATA);

    Py_DECREF(X);

    return result;
}

// TODO implement predict_proba later

// define methods
static PyMethodDef PyTree_methods[] = {
    {"fit", (PyCFunction)fit, METH_VARARGS, "Fit decision tree"},
    {"predict", (PyCFunction)predict, METH_VARARGS, "Predict labels"},
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
