#ifndef TREE_BOOTSTRAP_H

#define TREE_BOOTSTRAP_H

#include <stddef.h>
#include "tree.h"
#include "random.h"

void tree_fit_bootstrap(Node** root, const double* X, const size_t* y, const size_t* bootstrap_counts,
    size_t n, size_t p, size_t c, impurity_func_t impurity_func, size_t max_height,
    size_t min_samples_split, size_t min_samples_leaf, size_t max_features, pcg32_random_t* rng);

#endif
