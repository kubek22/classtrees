#ifndef CHECKS_H
#define CHECKS_H

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include "tree.h"
#include "random.h"
#include "pyobjects.h"


// parameter parsing

impurity_func_t get_impurity_func(const char* name);

int parse_impurity(PyObject* obj, const char** out);

int parse_n_estimators(PyObject* obj, size_t* out);

int parse_max_height(PyObject* obj, size_t* out);

int parse_min_samples_split(PyObject* obj, size_t* out);

int parse_min_samples_leaf(PyObject* obj, size_t* out);

int parse_max_features(PyObject* obj, size_t* out);

int parse_random_state(PyObject* obj, int* out);

int parse_n_jobs(PyObject* obj, int* out);

// data conversion (NumPy)

PyArrayObject* parse_X(PyObject* obj);

PyArrayObject* parse_y(PyObject* obj);

// validation

int check_same_n(PyArrayObject* X, PyArrayObject* y);

// tree
int require_fitted(PyTree* self);
int check_n_features(PyTree* self, PyArrayObject* X);

// rf
int require_fitted_rf(PyForest* self);
int check_n_features_rf(PyForest* self, PyArrayObject* X);


#endif
