#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <numpy/arrayobject.h>
#include <numpy/ndarrayobject.h>
#include <numpy/arrayscalars.h> 


static PyObject* hello(PyObject* self, PyObject* args) {
    return PyUnicode_FromString("hello from C");
}

static PyMethodDef methods[] = {
    {"hello", hello, METH_NOARGS, "hello function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "tree",
    NULL,
    -1,
    methods
};

PyMODINIT_FUNC PyInit_tree(void) {
    return PyModule_Create(&module);
}
