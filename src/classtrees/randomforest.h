#ifndef RF_H

#define RF_H

#include "tree.h"

void rf_fit(Node** roots, size_t n_estimators, const double* X, const int64_t* y,
    size_t n, size_t p, size_t c, impurity_func_t impurity_func, size_t max_height,
    size_t min_samples_split, size_t min_samples_leaf, size_t max_features,
    pcg32_random_t* rngs, int n_jobs);

double* rf_predict_proba(Node** roots, size_t n_estimators, const double* X, size_t n, size_t p, size_t c, int n_jobs);

int64_t* rf_predict(Node** roots, size_t n_estimators, const double* X, size_t n, size_t p, size_t c, int n_jobs);

#endif
