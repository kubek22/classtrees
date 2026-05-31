#ifndef PYOBJECTS_H

#define PYOBJECTS_H

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "tree.h"
#include "random.h"


// Python objects

// tree
typedef struct {
    PyObject_HEAD
    Node* root;
    size_t n_classes;
    size_t n_features;
    impurity_func_t impurity_func;
    size_t max_height;
    size_t min_samples_split;
    size_t min_samples_leaf;
    size_t max_features;
    int random_state;
    pcg32_random_t rng;
} PyTree;


// random forest
typedef struct {
    PyObject_HEAD
    Node** roots;
    size_t n_estimators;
    size_t n_classes;
    size_t n_features;
    impurity_func_t impurity_func;
    size_t max_height;
    size_t min_samples_split;
    size_t min_samples_leaf;
    size_t max_features;
    int random_state;
    pcg32_random_t* rngs;
    int n_jobs;
} PyForest;

#endif
