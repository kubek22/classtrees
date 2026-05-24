#include "tree.h"
#include "assert.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>

// Convention
// LEFT : <
// RIGHT : >=

// make some functions static

#define QUICKSORT_NMIN 40

#define MAX(A, B) ((A)>(B))?(A):(B)
#define MIN(A, B) ((A)<(B))?(A):(B)

// TODO for testing
static void print_double_array(double_array x) {
    ASSERT(x.data);

    printf("[");
    for (size_t i = 0; i < x.size; i++) {
        printf("%f, ", x.data[i]);
    }
    printf("]\n");
}

static idx_array init_indexes(size_t size) {
    idx_array ret;
    ret.size = size;
    ret.data = (size_t*)malloc(size * sizeof(size_t));
    for (size_t i=0; i<size; i++) {
        ret.data[i] = i;
    }
    return ret;
}

size_t get_classes(size_t* y, size_t n) {
    // we assume, that the classes are represented as consecutive natural numbers from 0 to c-1
    // we should maybe check first by an assert if there are at least 2 classes
    ASSERT(y);
    size_t ret = 0;
    for (size_t i = 0; i < n; i++)
        ret = MAX(ret, y[i]);
    // we return the number of classes so max_class_idx + 1
    return ret + 1;
}

static double_array zeros_array(size_t size) {
    double_array ret;
    ret.size = size;
    ret.data = (double*)malloc(size * sizeof(double));
    for (size_t i=0; i<size; i++) {
        ret.data[i] = 0.0;
    }
    return ret;
}

static idx_array zeros_idx_array(size_t size) {
    idx_array ret;
    ret.size = size;
    // TODO consider calloc - may be faster
    ret.data = (size_t*)malloc(size * sizeof(size_t));
    for (size_t i=0; i<size; i++) {
        ret.data[i] = 0;
    }
    return ret;
}

static idx_array get_counts(size_t c, const size_t* y, size_t* indexes, size_t start_idx, size_t end_idx) {
    ASSERT(y);
    ASSERT(indexes);
    ASSERT(end_idx - start_idx >= 1);

    // initializing probs array with zeros
    idx_array ret = zeros_idx_array(c);
    // counting classes
    for (size_t i=start_idx; i<end_idx; i++) {
        size_t class = y[indexes[i]];
        ASSERT(class < c);
        ret.data[class]++;
    }
    return ret;
}

static double_array get_probs_from_counts(idx_array counts, size_t n) {
    ASSERT(counts.size > 0);
    ASSERT(counts.data);
    ASSERT(n > 0);

    double_array ret;
    ret.size = counts.size;
    ret.data = (double*)malloc(ret.size * sizeof(double));
    double sum = 0.0;
    for (size_t i=0; i<ret.size - 1; i++) {
        ret.data[i] = (double)counts.data[i] / n;
        sum += ret.data[i];
    }
    // ensure normalization
    ret.data[ret.size - 1] = 1.0 - sum;
    return ret;
}

static double_array get_probs(size_t c, const size_t* y, size_t* indexes, size_t start_idx, size_t end_idx) {
    ASSERT(c >= 1);
    ASSERT(y);
    ASSERT(indexes);
    ASSERT(end_idx - start_idx >= 1);

    // initializing probs array with zeros
    double_array ret = zeros_array(c);
    // counting classes
    size_t sum = end_idx - start_idx;
    // size_t i = start_idx;
    while (start_idx < end_idx) {
        size_t class = y[indexes[start_idx]];
        ret.data[class] += 1.0;
        start_idx++;
    }
    // normalizing
    for (size_t i=0; i<ret.size; i++) {
        ret.data[i] /= (double)sum;
    }
    return ret;
}

// TODO check if it is used
static double gini(double_array probs) {
    ASSERT(probs.data);
    double ret = 1.0;
    for (size_t i = 0; i < probs.size; i++) {
        ret -= probs.data[i] * probs.data[i];
    }
    return ret;
}

double gini_from_counts(idx_array counts, size_t n) {
    ASSERT(n > 0);
    ASSERT(counts.data);
    double ret = 1.0;
    for (size_t i = 0; i < counts.size; i++) {
        double p = (double)counts.data[i] / n;
        ret -= p * p;
    }
    return ret;
}

static double entropy(double_array probs) {
    ASSERT(probs.data);
    double ret = 0.0;
    for (size_t i = 0; i < probs.size; i++) {
        double p = probs.data[i];
        // we assume 0log(0) is 0
        if (p > 0.0) {
            ret -= p * log(p);
        }
    }
    return ret;
}

double entropy_from_counts(idx_array counts, size_t n) {
    ASSERT(n > 0);
    ASSERT(counts.data);
    double ret = 0.0;
    for (size_t i = 0; i < counts.size; i++) {
        double p = (double)counts.data[i] / n;
        if (p > 0.0) {
            ret -= p * log(p);
        }
    }
    return ret;
}

// for testing
// consider C contiguous arrays
static double** generate_X(size_t n, size_t p) {
    ASSERT(n > 0);
    ASSERT(p > 0);
    double** X = (double**)malloc(n * sizeof(double*));
    for (size_t i = 0; i < n; i++) {
        X[i] = (double*)malloc(p * sizeof(double));
        for (size_t j = 0; j < p; j++) {
            X[i][j] = (double)rand() / RAND_MAX; // random values between 0 and 1
        }
    }
    return X;
}

// for testing
// should accept also C contiguous arrays
static size_t* generate_y(double * const * X, size_t n, size_t p, size_t c) {
    ASSERT(X); // maybe assert also X[i]?
    ASSERT(n > 0);
    ASSERT(p > 0);
    ASSERT(c > 0);

    size_t* y = (size_t*)malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; i++) {
        double total = 0.0;
        for (size_t j = 0; j < p; j++) {
            total += X[i][j];
        }
        y[i] = (size_t)(total / p * c) % c;
    }
    return y;
}

// for testing
static size_t* generate_binary_y(double * const *X, size_t n, size_t p) {
    ASSERT(X);
    ASSERT(n > 0);
    ASSERT(p > 0);
    size_t* y = malloc(n * sizeof(size_t));
    for (size_t i = 0; i < n; ++i) {
        double x0 = X[i][0 % p];
        double x1 = X[i][1 % p];
        double x2 = X[i][2 % p];
        if (x0 < 0.5) {
            y[i] = (x1 < 0.5) ? 0 : 1;
        } else {
            y[i] = (x2 < 0.5) ? 1 : 0;
        }
    }
    return y;
}

// for testing
static void free_X(double** X, size_t n) {
    for (size_t i = 0; i < n; i++) {
        free(X[i]);
    }
    free(X);
}

// TODO modified - check references to it
static size_t partition(const double* X, size_t* indexes, size_t start_idx, size_t end_idx, size_t feature, double t, size_t p) {
    // p is the number of features
    // modifies indexes within range [start_idx, end_idx) so that values X[:][feature] < t in the analyzed range will be located before elements >= t
    // returns the first index of an element >= t
    // t threshold does not need to be present in the data
    // the algorithm guarantees only to place elements <t first and >=t later
    // it does not guarantee shrinkage of the problem so cannot be applied for quicksort
    // it t is greater than all values, end_idx is returned

    ASSERT(X);
    ASSERT(indexes);
    ASSERT(end_idx >= start_idx);
    ASSERT(feature < p);

    // copying the initial indexes
    size_t i = start_idx;
    size_t j = end_idx;

    while (i < j) {
        // searching for missplaced elements
        while (i < j && X[indexes[i] * p + feature] < t)
            i++;
        // dangerous - it may underflow
        while (i < j && X[indexes[j - 1] * p + feature] >= t)
            j--;
        // swapping the elements
        if (i < j) {
            size_t tmp = indexes[i];
            indexes[i] = indexes[j - 1];
            indexes[j - 1] = tmp;
            // moving forward
            ++i;
            --j;
        }
    }
    return i;
}

// TODO modified - check references to it
static size_t partition_quicksort(const double* X, size_t* indexes, size_t start_idx, size_t end_idx, size_t feature, size_t p_index, size_t p) {
    // Require non-empty range
    // Require start_idx <= p_index < end_idx
    ASSERT(X);
    ASSERT(indexes);
    ASSERT(end_idx > start_idx);
    ASSERT(p_index >= start_idx && p_index < end_idx);
    ASSERT(feature < p);

    double pivot = X[indexes[p_index] * p + feature];

    // Move pivot to the end
    size_t pivot_idx = indexes[p_index];
    indexes[p_index] = indexes[end_idx - 1];
    indexes[end_idx - 1] = pivot_idx;

    // Partition excluding pivot
    size_t mid = partition(X, indexes, start_idx, end_idx - 1, feature, pivot, p);

    // Move pivot into final place
    indexes[end_idx - 1] = indexes[mid];
    indexes[mid] = pivot_idx;

    return mid;
}

// TODO not used now
static double pivot(double * const * X, size_t feature, size_t* indexes, size_t start_idx, size_t end_idx) {
    // selects 3 elements and chooses a median from them

    ASSERT(X);
    ASSERT(indexes);
    ASSERT(end_idx > start_idx);

    size_t mid = start_idx + (end_idx - start_idx) / 2;

    double a = X[indexes[start_idx]][feature];
    double b = X[indexes[mid]][feature];
    double c = X[indexes[end_idx - 1]][feature];
    
    if ((a <= b && b <= c) || (c <= b && b <= a))
        return b;
    if ((b <= a && a <= c) || (c <= a && a <= b))
        return a;
    return c;
}

// TODO new
static size_t pivot_idx(const double* X, size_t feature, size_t* indexes, size_t start_idx, size_t end_idx, size_t p) {
    ASSERT(X);
    ASSERT(indexes);
    ASSERT(start_idx < end_idx);
    ASSERT(feature < p);

    // selects 3 elements and chooses a median from them
    size_t mid = start_idx + (end_idx - start_idx) / 2;

    double a = X[indexes[start_idx] * p + feature];
    double b = X[indexes[mid] * p + feature];
    double c = X[indexes[end_idx - 1] * p + feature];

    if ((a <= b && b <= c) || (c <= b && b <= a))
        return mid;
    if ((b <= a && a <= c) || (c <= a && a <= b))
        return start_idx;
    return end_idx - 1;
}

static void insertion_argsort(const double* X, size_t feature, size_t* indexes, size_t start_idx, size_t end_idx, size_t p) {
    ASSERT(X);
    ASSERT(indexes);
    ASSERT(feature < p);

    for (size_t i = start_idx + 1; i < end_idx; i++) {
        size_t idx = indexes[i];
        double t = X[idx * p + feature];
        size_t j = i;
        while (j > start_idx && X[indexes[j - 1] * p + feature] > t) {
            indexes[j] = indexes[j - 1];
            j--;
        }
        indexes[j] = idx;
    }
}

static void argsort(const double* X, size_t feature, size_t* indexes, size_t start_idx, size_t end_idx, size_t p) {
    // gets X matrix as input, considers only given features and applies argsort on indexes array within given range
    // modifies indexes inplace
    // quicksort + insertion sort

    ASSERT(X);
    ASSERT(indexes);
    ASSERT(feature < p);

    // sort until there is only one element
    size_t n = end_idx - start_idx;
    while (n > 1) {
        // run insertion sort for small arrays
        if (n < QUICKSORT_NMIN) return insertion_argsort(X, feature, indexes, start_idx, end_idx, p);
        size_t p_index = pivot_idx(X, feature, indexes, start_idx, end_idx, p);
        size_t mid_idx = partition_quicksort(X, indexes, start_idx, end_idx, feature, p_index, p);
        if (mid_idx - start_idx < n - mid_idx - 1) {
            argsort(X, feature, indexes, start_idx, mid_idx, p);
            start_idx = mid_idx + 1;
        }
        else {
            argsort(X, feature, indexes, mid_idx + 1, end_idx, p);
            end_idx = mid_idx;
        }
        // update size
        n = end_idx - start_idx;
    }
}

// TODO used only for tests
static size_t assert_arrays_equal(double_array x, double_array y) {
    // for testing
    ASSERT(x.data);
    ASSERT(y.data);
    if (x.size != y.size) return 1;
    size_t error = 0;
    for (size_t i = 0; i < x.size; i++) {
        if (fabs(x.data[i] - y.data[i]) > 1e-12) {
            print_double_array(x);
            print_double_array(y);
            printf("\n");
            error = 1;
        }
    }
    return error;
}

// TODO maybe should be accessible from other files
Node* init_node(size_t h, double_array* probs, double impurity) {
    Node* node = (Node*)malloc(sizeof(Node));
    node->h = h;
    node->probs = *probs;
    node->impurity = impurity;
    node->feature = 0;
    node->threshold = 0.0;
    node->left = NULL;
    node->right = NULL;
    return node;
}

static void split(Node* node, const double* X, const size_t* y, size_t p, size_t c, size_t* indexes, size_t start_idx, size_t end_idx, double (*impurity_func)(idx_array, size_t), size_t max_height, size_t min_samples_split) {
    // p is the number of features
    // c is the number of classes
    // impurity_func is gini or entropy
    // for given data X and for considered observation within range [start_idx, end_idx), we want to make an optimal split

    ASSERT(node);
    ASSERT(X);
    ASSERT(y);
    ASSERT(start_idx < end_idx);

    // we need at least two samples to make a split
    if (end_idx - start_idx < 2)
        return;
    // for impurity 0, we dont split anymore - parametrize later
    if (node->impurity < 1e-12)
        return;
    // max height
    if (node->h >= max_height)
        return;
    // minimal number of samples to split a leaf
    if (end_idx - start_idx < min_samples_split)
        return;

    // store initial best impurity in the node
    // we want to minimize impurities
    double best_score = INFINITY;
    size_t n = end_idx - start_idx; // number of observations in a node

    // for each feature
    for (size_t feature = 0; feature < p; feature++) {
        // sort values for the feature
        argsort(X, feature, indexes, start_idx, end_idx, p);
        
        // initially all elements are in the right subtree
        // possibly allocate this only once and just fill with zeros
        idx_array left_counts = zeros_idx_array(c);
        idx_array right_counts = get_counts(c, y, indexes, start_idx, end_idx);

        // iterate over middle points as thresholds
        for (size_t i = 0; i < n - 1; i++) {
            // TODO: check what happens if there is only one unique value -> then impurity is 0 -> we already check it at the beginning

            // move 1 element to the left subtree
            size_t class = y[indexes[start_idx + i]];
            left_counts.data[class]++;
            right_counts.data[class]--;

            // skip if we have identical values
            if (fabs(X[indexes[start_idx + i + 1] * p + feature] - X[indexes[start_idx + i] * p + feature]) < 1e-12)
                continue;

            // compute the new threshold
            double threshold = (X[indexes[start_idx + i] * p + feature] + X[indexes[start_idx + i + 1] * p + feature]) * 0.5;

            // update with counts
            double impurity_left = impurity_func(left_counts, i+1);
            double impurity_right = impurity_func(right_counts, n-i-1);

            // counting prior probability of selecting the right and left node
            // there are (i+1) observations in the left subtree and end_idx-start_idx-i-1 in the right one
            double p_left = (double)(i + 1) / n;
            double p_right = 1.0 - p_left;

            // overwrite the best results if needed
            double score = p_left * impurity_left + p_right * impurity_right;

            if (best_score > score) {
                best_score = score;
                // set the best feature and threshold
                node->feature = feature;
                node->threshold = threshold;
            }
        }
        // we already chose the best threshold for given feature

        // freeing counts
        free(left_counts.data);
        free(right_counts.data);
    }
    // we already chose the best feature and threshold

    // recomputing probs - partition will be faster than sorting
    size_t mid_idx = partition(X, indexes, start_idx, end_idx, node->feature, node->threshold, p);

    // recompute probs
    // helper counts - for imputity
    idx_array counts_left = get_counts(c, y, indexes, start_idx, mid_idx);
    idx_array counts_right = get_counts(c, y, indexes, mid_idx, end_idx);

    double impurity_left = impurity_func(counts_left, mid_idx - start_idx);
    double impurity_right = impurity_func(counts_right, end_idx - mid_idx);

    double_array probs_left = get_probs_from_counts(counts_left, mid_idx - start_idx);
    double_array probs_right = get_probs_from_counts(counts_right, end_idx - mid_idx);

    free(counts_left.data);
    free(counts_right.data);

    // creating subtrees
    node->left = init_node(node->h+1, &probs_left, impurity_left);
    node->right = init_node(node->h+1, &probs_right, impurity_right);

    // recurentially build the tree
    // maybe start from the subtree with smaller number of observations
    split(node->left, X, y, p, c, indexes, start_idx, mid_idx, impurity_func, max_height, min_samples_split);
    split(node->right, X, y, p, c, indexes, mid_idx, end_idx, impurity_func, max_height, min_samples_split);
}

// TODO check if it will be needed
void free_tree(Node* root) {
    if (!root) return;
    if (root->left) free_tree(root->left);
    if (root->right) free_tree(root->right);
    root->left = NULL;
    root->right = NULL;
    free(root->probs.data);
    free(root);
}

// TODO change ASSESRT(root) into python error optionally
// TODO it works only for C contiguous arrays - check if it is needed
static int64_t predict_one(Node* root, const double* x) {
    // predicts a class for only one observation
    
    // sanity check
    ASSERT(root);
    ASSERT(x);
    
    // going down to a leaf
    while (root->left || root->right) {
        // TODO it may be dangerous when we try to access non-existing feature -> should be checked at the beginning
        if (x[root->feature] < root->threshold)
            root = root->left;
        else
            root = root->right;
    }
    ASSERT(root->probs.data);
    ASSERT(root->probs.size > 0);
    
    size_t c = root->probs.size;
    double* probs_array = root->probs.data;
    // select argmax
    size_t ret = 0;
    for (size_t i = 1; i < c; i++) {
        // for equal probs the first class is returned
        if (probs_array[ret] < probs_array[i])
            ret = i;
    }
    return (int64_t)ret;
}

int64_t* tree_predict(Node* root, const double* X, size_t n, size_t p) {
    // initialize y_pred
    int64_t* ret = (int64_t*)malloc(n * sizeof(int64_t));
    // iterate over samples
    for (size_t i = 0; i < n; i++) {
        // TODO check if it makes sense
        ret[i] = predict_one(root, &X[i * p]);
    }
    return ret;
}

static double* predict_proba_one(Node* root, const double* x) {
    // predicts a class for only one observation
    
    // sanity check
    ASSERT(root);
    ASSERT(x);
    
    // going down to a leaf
    while (root->left || root->right) {
        // TODO it may be dangerous when we try to access non-existing feature -> should be checked at the beginning
        if (x[root->feature] < root->threshold)
            root = root->left;
        else
            root = root->right;
    }
    return root->probs.data;
}

double* tree_predict_proba(Node* root, const double* X, size_t n, size_t p) {
    size_t c = root->probs.size;
    // initialize y_pred
    double* ret = (double*)malloc(n * c * sizeof(double));
    // iterate over samples
    for (size_t i = 0; i < n; i++) {
        // TODO check if it makes sense
        double* probs = predict_proba_one(root, &X[i * p]);
        for (size_t j = 0; j < c; j++) {
            ret[i * c + j] = probs[j];
        }
    }
    return ret;
}

// TODO for tests
static double* c_contiguous_array(double * const * X, size_t n, size_t p) {
    double* ret = (double*)malloc(n * p * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < p; j++) {
            ret[i * p + j] = X[i][j];
        }
    }
    return ret;
}

static double accuracy(const size_t* y_true, const size_t* y_pred, size_t n) {
    ASSERT(y_true);
    ASSERT(y_pred);
    ASSERT(n > 0);

    size_t correct = 0;
    for (size_t i = 0; i < n; i++) {
        if (y_true[i] == y_pred[i])
            correct++;
    }
    return ((double)correct) / n;
}


// ----- UNIT TESTS -----
// TODO move to separate file later


static void test_init_indexes(size_t n) {
    idx_array arr = init_indexes(n);

    ASSERT(arr.size == n);

    for (size_t i = 0; i < n; i++) {
        ASSERT(arr.data[i] == i);
    }

    free(arr.data);
}

static void test_get_classes_basic() {
    size_t y[] = {0, 1, 2, 1, 0};

    ASSERT(get_classes(y, 5) == 3);
}

static void test_zeros_array(size_t n) {
    double_array arr = zeros_array(n);

    ASSERT(arr.size == n);

    for (size_t i = 0; i < n; i++) {
        ASSERT(arr.data[i] == 0.0);
    }

    free(arr.data);
}

static void test_get_classes() {
    // single class
    size_t y1[] = {0, 0, 0, 0};
    ASSERT(get_classes(y1, 4) == 1);

    // unordered classes
    size_t y2[] = {2, 0, 1, 2};
    ASSERT(get_classes(y2, 4) == 3);

    // wrong labeling - jump
    size_t y3[] = {0, 5, 2, 1};
    ASSERT(get_classes(y3, 4) == 6);
}

static void test_get_probs_basic() {
    size_t y[] = {0, 1, 1, 2};
    size_t indexes[] = {0, 1, 2, 3};

    double_array probs =
        get_probs(3, y, indexes, 0, 4);

    ASSERT(probs.size == 3);

    ASSERT_DOUBLE_EQ(probs.data[0], 0.25, 1e-12);
    ASSERT_DOUBLE_EQ(probs.data[1], 0.50, 1e-12);
    ASSERT_DOUBLE_EQ(probs.data[2], 0.25, 1e-12);

    free(probs.data);
}

static void test_get_probs_subset_indexes() {
    size_t y[] = {0, 1, 2, 1, 0};

    size_t indexes[] = {4, 3, 1};

    /*
        selected labels:
        y[4] = 0
        y[3] = 1
        y[1] = 1

        counts:
        class 0 -> 1
        class 1 -> 2
        class 2 -> 0
    */

    double_array probs =
        get_probs(3, y, indexes, 0, 3);

    ASSERT_DOUBLE_EQ(probs.data[0], 1.0/3.0, 1e-12);
    ASSERT_DOUBLE_EQ(probs.data[1], 2.0/3.0, 1e-12);
    ASSERT_DOUBLE_EQ(probs.data[2], 0.0,     1e-12);

    free(probs.data);
}

static void test_get_probs_subrange() {
    size_t y[] = {0, 1, 2, 1, 0};

    size_t indexes[] = {0, 1, 2, 3, 4};

    /*
        using indexes[1:4]

        labels:
        1,2,1
    */

    double_array probs =
        get_probs(3, y, indexes, 1, 4);

    ASSERT_DOUBLE_EQ(probs.data[0], 0.0,     1e-12);
    ASSERT_DOUBLE_EQ(probs.data[1], 2.0/3.0, 1e-12);
    ASSERT_DOUBLE_EQ(probs.data[2], 1.0/3.0, 1e-12);

    free(probs.data);
}

static void test_get_probs_sum_to_one() {
    size_t y[] = {0, 1, 1, 2};
    size_t indexes[] = {0, 1, 2, 3};

    double_array probs =
        get_probs(3, y, indexes, 0, 4);

    double sum = 0.0;

    for (size_t i = 0; i < probs.size; ++i)
        sum += probs.data[i];

    ASSERT_DOUBLE_EQ(sum, 1.0, 1e-12);

    free(probs.data);
}

static void test_gini_pure() {
    double data[] = {1.0, 0.0, 0.0};

    double_array probs = {
        .size = 3,
        .data = data
    };

    ASSERT_DOUBLE_EQ(gini(probs), 0.0, 1e-12);
}

static void test_gini_binary_balanced() {
    double data[] = {0.5, 0.5};

    double_array probs = {
        .size = 2,
        .data = data
    };

    /*
        1 - (0.25 + 0.25) = 0.5
    */

    ASSERT_DOUBLE_EQ(gini(probs), 0.5, 1e-12);
}

static void test_gini_uniform_three_classes() {
    double data[] = {
        1.0/3.0,
        1.0/3.0,
        1.0/3.0
    };

    double_array probs = {
        .size = 3,
        .data = data
    };

    /*
        1 - 3*(1/9) = 2/3
    */

    ASSERT_DOUBLE_EQ(gini(probs), 2.0/3.0, 1e-12);
}

static void test_entropy_pure() {
    double data[] = {1.0, 0.0};

    double_array probs = {
        .size = 2,
        .data = data
    };

    ASSERT_DOUBLE_EQ(entropy(probs), 0.0, 1e-12);
}

static void test_entropy_binary_balanced() {
    double data[] = {0.5, 0.5};

    double_array probs = {
        .size = 2,
        .data = data
    };

    ASSERT_DOUBLE_EQ(
        entropy(probs),
        log(2.0),
        1e-12
    );
}

static void test_entropy_uniform_three_classes() {
    double data[] = {
        1.0/3.0,
        1.0/3.0,
        1.0/3.0
    };

    double_array probs = {
        .size = 3,
        .data = data
    };

    ASSERT_DOUBLE_EQ(
        entropy(probs),
        log(3.0),
        1e-12
    );
}

static void assert_partition_valid(const double* X, size_t* indexes, size_t start_idx, size_t mid, size_t end_idx, size_t feature, double t, size_t p) {
    for (size_t i = start_idx; i < mid; ++i) {
        ASSERT(X[indexes[i] * p + feature] < t);
    }
    for (size_t i = mid; i < end_idx; ++i) {
        ASSERT(X[indexes[i] * p + feature] >= t);
    }
}

static void test_partition_basic() {
    double X[] = {1.0, 5.0, 2.0, 7.0};

    size_t indexes[] = {0,1,2,3};

    size_t mid =
        partition(X, indexes, 0, 4, 0, 3.0, 1);

    assert_partition_valid(X, indexes, 0, mid, 4, 0, 3.0, 1);

    ASSERT(mid == 2);
}

static void test_partition_already_partitioned() {
    double X[] = {1.0, 2.0, 5.0, 6.0};

    size_t indexes[] = {0,1,2,3};

    size_t mid = partition(X, indexes, 0, 4, 0, 4.0, 1);

    assert_partition_valid(X, indexes, 0, mid, 4, 0, 4.0, 1);

    ASSERT(mid == 2);
}

static void test_partition_all_left() {
    double X[] = {1.0, 2.0};

    size_t indexes[] = {0,1};

    size_t mid = partition(X, indexes, 0, 2, 0, 5.0, 1);

    ASSERT(mid == 2);

    assert_partition_valid(X, indexes, 0, mid, 2, 0, 5.0, 1);
}

static void test_partition_all_right() {
    double X[] = {5.0, 6.0};

    size_t indexes[] = {0,1};

    size_t mid = partition(X, indexes, 0, 2, 0, 5.0, 1);

    ASSERT(mid == 0);

    assert_partition_valid(X, indexes, 0, mid, 2, 0, 5.0, 1);
}

static void test_partition_equal_goes_right() {
    double X[] = {3.0, 2.0};

    size_t indexes[] = {0,1};

    size_t mid = partition(X, indexes, 0, 2, 0, 3.0, 1);

    assert_partition_valid(X, indexes, 0, mid, 2, 0, 3.0, 1);

    ASSERT(mid == 1);
}

static void test_partition_subrange() {
    double X[] = {100.0, 1.0, 5.0, 2.0, 100.0};

    size_t indexes[] = {0,1,2,3,4};

    size_t mid = partition(X, indexes, 1, 4, 0, 3.0, 1);

    /*
        only indexes[1:4] may change
    */

    ASSERT(indexes[0] == 0);
    ASSERT(indexes[4] == 4);

    assert_partition_valid(X, indexes, 1, mid, 4, 0, 3.0, 1);
}

static void check_partition(const double* X, size_t *indexes, size_t start_idx, size_t end_idx, size_t feature, size_t mid, size_t p) {
    double pivot = X[indexes[mid] * p + feature];
    for (size_t i = start_idx; i < mid; ++i) {
        ASSERT(X[indexes[i] * p + feature] < pivot);
    }
    for (size_t i = mid + 1; i < end_idx; ++i) {
        ASSERT(X[indexes[i] * p + feature] >= pivot);
    }
}

static void test_sorted() {
    double X[] = {1.0, 2.0, 3.0, 4.0};

    size_t indexes[] = {0,1,2,3};

    size_t mid = partition_quicksort(X, indexes, 0, 4, 0, 3, 1); // pivot = 4

    check_partition(X, indexes, 0, 4, 0, mid, 1);
}

static void test_reverse() {
    double X[] = {4.0, 3.0, 2.0, 1.0};

    size_t indexes[] = {0,1,2,3};

    size_t mid = partition_quicksort(X, indexes, 0, 4, 0, 0, 1); // pivot = 4

    check_partition(X, indexes, 0, 4, 0, mid, 1);
}

static void test_duplicates() {
    double X[] = {5.0, 1.0, 5.0, 3.0, 5.0};

    size_t indexes[] = {0,1,2,3,4};

    size_t mid = partition_quicksort(X, indexes, 0, 5, 0, 2, 1); // pivot = 5

    check_partition(X, indexes, 0, 5, 0, mid, 1);
}

static void test_all_equal() {
    double X[] = {7.0, 7.0, 7.0, 7.0};

    size_t indexes[] = {0,1,2,3};

    size_t mid = partition_quicksort(X, indexes, 0, 4, 0, 1, 1); // pivot = 7

    check_partition(X, indexes, 0, 4, 0, mid, 1);
}

static void test_pivot_smallest() {
    double X[] = {9.0, 8.0, 1.0, 7.0};

    size_t indexes[] = {0,1,2,3};

    size_t mid = partition_quicksort(X, indexes, 0, 4, 0, 2, 1); // pivot = 3

    ASSERT(mid == 0);

    check_partition(X, indexes, 0, 4, 0, mid, 1);
}

static void test_pivot_largest() {
    double X[] = {1.0, 2.0, 9.0, 3.0};

    size_t indexes[] = {0,1,2,3};

    size_t mid = partition_quicksort(X, indexes, 0, 4, 0, 2, 1); // pivot = 3

    ASSERT(mid == 3);

    check_partition(X, indexes, 0, 4, 0, mid, 1);
}

static void test_subrange() {
    double X[] = {10.0, 4.0, 2.0, 8.0, 99.0};

    size_t indexes[] = {0,1,2,3,4};

    size_t mid = partition_quicksort(X, indexes, 1, 4, 0, 1, 1); // pivot = 4

    // untouched
    ASSERT(indexes[0] == 0);
    ASSERT(indexes[4] == 4);

    check_partition(X, indexes, 1, 4, 0, mid, 1);
}

static void assert_sorted(const double* X, size_t feature, size_t* indexes, size_t start_idx, size_t end_idx, size_t p) {
    if (end_idx - start_idx < 2)
        return;
    for (size_t i = start_idx + 1; i < end_idx; ++i) {
        double prev = X[indexes[i-1] * p + feature];
        double curr = X[indexes[i] * p + feature];
        ASSERT(prev <= curr);
    }
}

static void assert_contains_indexes( size_t* indexes, size_t n) {
    int* seen = calloc(n, sizeof(int));
    for (size_t i = 0; i < n; ++i) {
        ASSERT(indexes[i] < n);
        ASSERT(!seen[indexes[i]]);
        seen[indexes[i]] = 1;
    }
    free(seen);
}

static void test_argsort_basic() {
    double X[] = {5.0, 1.0, 3.0};

    size_t indexes[] = {0,1,2};

    argsort(X, 0, indexes, 0, 3, 1);

    assert_sorted(X, 0, indexes, 0, 3, 1);

    ASSERT(indexes[0] == 1);
    ASSERT(indexes[1] == 2);
    ASSERT(indexes[2] == 0);
}

static void test_argsort_already_sorted() {
    double X[] = {1.0, 2.0, 3.0};

    size_t indexes[] = {0,1,2};

    argsort(X, 0, indexes, 0, 3, 1);

    assert_sorted(X, 0, indexes, 0, 3, 1);
}

static void test_argsort_duplicates() {
    double X[] = {2.0, 1.0, 2.0, 1.0};

    size_t indexes[] = {0,1,2,3};

    argsort(X, 0, indexes, 0, 4, 1);

    assert_sorted(X, 0, indexes, 0, 4, 1);

    assert_contains_indexes(indexes, 4);
}

static void test_argsort_subrange() {
    double X[] = {100.0, 3.0, 1.0, 2.0, 100.0};

    size_t indexes[] = {0,1,2,3,4};

    argsort(X, 0, indexes, 1, 4, 1);

    ASSERT(indexes[0] == 0);
    ASSERT(indexes[4] == 4);

    assert_sorted(X, 0, indexes, 1, 4, 1);
}

static void test_argsort_feature_selection() {
    double X[] = {10.0, 3.0, 1.0, 2.0, 5.0, 1.0};

    size_t indexes[] = {0,1,2};

    argsort(X, 1, indexes, 0, 3, 2);

    assert_sorted(X, 1, indexes, 0, 3, 2);

    ASSERT(indexes[0] == 2);
    ASSERT(indexes[1] == 1);
    ASSERT(indexes[2] == 0);
}

static void test_argsort_large(size_t n) {
    double** _X = generate_X(n, 1);
    double* X = c_contiguous_array(_X, n, 1);

    idx_array indexes = init_indexes(n);

    argsort(X, 0, indexes.data, 0, n, 1);
    assert_sorted(X, 0, indexes.data, 0, n, 1);

    free_X(_X, n);
    free(indexes.data);
    free(X);
}

static void tests() {
    // initializing indexes
    test_init_indexes(0);
    test_init_indexes(1);
    test_init_indexes(5);

    // detecting the number of classes
    test_get_classes();

    // initializing array with zeros
    test_zeros_array(0);
    test_zeros_array(1);
    test_zeros_array(5);

    // test probs calculation
    test_get_probs_basic();
    test_get_probs_subset_indexes();
    test_get_probs_subrange();
    test_get_probs_sum_to_one();

    // test gini
    test_gini_pure();
    test_gini_binary_balanced();
    test_gini_uniform_three_classes();

    // test entropy
    test_entropy_pure();
    test_entropy_binary_balanced();
    test_entropy_uniform_three_classes();

    // test partition
    test_partition_basic();
    test_partition_already_partitioned();
    test_partition_all_left();
    test_partition_all_right();
    test_partition_equal_goes_right();
    test_partition_subrange();

    printf("Partition tests passed\n");

    // test quicksort partition
    test_sorted();
    test_reverse();
    test_duplicates();
    test_all_equal();
    test_pivot_smallest();
    test_pivot_largest();
    test_subrange();

    printf("Quicksort partition tests passed\n");

    // test argort
    test_argsort_basic();
    test_argsort_already_sorted();
    test_argsort_duplicates();
    test_argsort_subrange();
    test_argsort_feature_selection();
    test_argsort_large(1);
    test_argsort_large(2);
    test_argsort_large(5);
    test_argsort_large(10);
    test_argsort_large(1000);
    test_argsort_large(1000000);

    printf("Argsort tests passed\n");

    printf("All tests passed\n");

}

// ----- DEBUG -----

static void print_indent(size_t depth) {
    for (size_t i = 0; i < depth; ++i)
        printf("  ");
}

// TODO maybe should be accessible from other files 
static void inspect_tree_node(const Node* node, size_t depth) {
    if (!node) {
        print_indent(depth);
        printf("NULL node\n");
        return;
    }

    print_indent(depth);

    // leaf detection
    int is_leaf = (node->left == NULL && node->right == NULL);

    if (is_leaf) {
        printf("[LEAF] impurity=%.6f probs=(", node->impurity);
        for (size_t i = 0; i < node->probs.size; ++i) {
            printf("%.3f", node->probs.data[i]);
            if (i + 1 < node->probs.size)
                printf(", ");
        }
        printf(")\n");
        return;
    }

    printf("[NODE] feature=%zu threshold=%.6f impurity=%.6f probs=(",
           node->feature,
           node->threshold,
           node->impurity);

    for (size_t i = 0; i < node->probs.size; ++i) {
        printf("%.3f", node->probs.data[i]);
        if (i + 1 < node->probs.size)
            printf(", ");
    }
    printf(")\n");

    inspect_tree_node(node->left, depth + 1);
    inspect_tree_node(node->right, depth + 1);
}


// TODO expose
// probably it should estimate the number of classes and set it as a parameter for the tree
void tree_fit(Node** root, const double* X, const size_t* y, size_t n, size_t p, size_t c, double (*impurity_func)(idx_array, size_t), size_t max_height, size_t min_samples_split) {
    // init indexes
    idx_array indexes = init_indexes(n); // maybe change the method to return just the array

    // initialize root
    idx_array counts_root = get_counts(c, y, indexes.data, 0, n);
    double_array probs_root = get_probs_from_counts(counts_root, n);
    double impurity_root = impurity_func(counts_root, n);
    free(counts_root.data);
    *root = init_node(0, &probs_root, impurity_root);

    // build the tree
    split(*root, X, y, p, c, indexes.data, 0, n, impurity_func, max_height, min_samples_split);

    // free memory
    free(indexes.data);
}

static void tree_test() {
    // random seed
    srand(time(NULL));

    // we have X nxp double matrix as input and y classes vector with c unique values
    size_t n = 10000;
    size_t p = 10;
    size_t max_c = 5;
    printf("Max number of classes: %zu\n", max_c);

    // initialize X
    double** _X = generate_X(n, p);
    double* X = c_contiguous_array(_X, n, p);

    // initialize y - c classes
    size_t* y = generate_y(_X, n, p, max_c);
    // size_t* y = generate_binary_y(X, n, p);

    // we should check if there are at least 2 classes
    // detecting actual number of classes
    size_t c = get_classes(y, n);
    printf("Detected number of classes: %zu\n", c);

    printf("Initializing indexes...\n");
    idx_array indexes = init_indexes(n); // maybe change the method to return just the array

    // initialize the tree
    printf("Initializing the tree...\n");

    // select impurity function
    double (*impurity_func)(idx_array, size_t) = gini_from_counts;
    // or entropy
    // double (*impurity_func)(idx_array, size_t) = entropy_from_counts;

    idx_array counts_root = get_counts(c, y, indexes.data, 0, n);
    double_array probs_root = get_probs_from_counts(counts_root, n);
    double impurity_root = impurity_func(counts_root, n);
    free(counts_root.data);

    Node* root = init_node(0, &probs_root, impurity_root);
    printf("Initial impurity: %f\n", root->impurity);

    printf("Setting hyperparameters...\n");
    size_t max_height = 5;
    size_t min_samples_split = 5;

    // build the tree
    printf("Fitting the tree...\n"); 
    split(root, X, y, p, c, indexes.data, 0, n, impurity_func, max_height, min_samples_split);
    printf("The tree is fitted\n");

    printf("Printing the tree\n");
    inspect_tree_node(root, max_height);

    // prediction
    printf("Prediction...\n");
    size_t* y_pred = tree_predict(root, X, n, p);
    double acc = accuracy(y, y_pred, n);
    printf("Accuracy: %f\n", acc);

    // free X, y, indexes and tree nodes
    printf("Clearing the memory...\n");
    free_X(_X, n);
    free(X);
    free(y);
    free(y_pred);
    free(indexes.data);
    free_tree(root);
}

int main() {

    printf("Running TESTS...\n");
    tests();

    printf("Running tree test...\n");
    tree_test();

    printf("Program finished\n");
    
    return 0;
}

