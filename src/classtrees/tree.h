#ifndef TREE_H

#define TREE_H

#include <stddef.h>
#include <stdint.h>
#include "random.h"


// defining types
typedef struct _idx_array {
    size_t size;
    size_t* data;
} idx_array;

typedef struct _array {
    size_t size;
    double* data;
} double_array;

typedef struct _Node {
    struct _Node* left;
    struct _Node* right;
    size_t feature; // optionally an ordering of features to deal with NaNs
    double threshold;
    size_t h; // height
    double_array probs; // node probabilities
    double impurity;
} Node;

typedef double (*impurity_func_t)(idx_array, size_t);

// impurity_func_t get_impurity_func(const char* str);
double gini_from_counts(idx_array counts, size_t n);
double entropy_from_counts(idx_array counts, size_t n);
size_t get_classes(size_t* y, size_t n);
void tree_fit(Node** root, const double* X, const size_t* y, size_t n, size_t p, size_t c,
    impurity_func_t impurity_func, size_t max_height, size_t min_samples_split, size_t min_samples_leaf,
    size_t max_features, pcg32_random_t* rng);
int64_t * tree_predict(Node* root, const double* X, size_t n, size_t p);
double* tree_predict_proba(Node* root, const double* X, size_t n, size_t p);
double* predict_proba_one(Node* root, const double* x);


#endif
