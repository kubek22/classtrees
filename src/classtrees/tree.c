#include "tree.h"
#include "assert.h"
#include "random.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>
#include <time.h>


// Convention
// LEFT : <
// RIGHT : >=


#define QUICKSORT_NMIN 40

#define MAX(A, B) (((A)>(B))?(A):(B))
#define MIN(A, B) (((A)<(B))?(A):(B))

// for testing
static void print_double_array(double_array x) {
    ASSERT(x.data);

    printf("[");
    for (size_t i = 0; i < x.size; i++) {
        printf("%f, ", x.data[i]);
    }
    printf("]\n");
}

// used only for tests
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
    // consider calloc - may be faster
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

static size_t partition(double* feat, size_t* indexes, size_t start_idx, size_t end_idx, double t) {
    // p is the number of features
    // modifies indexes within range [start_idx, end_idx) so that values X[:][feature] < t in the analyzed range will be located before elements >= t
    // returns the first index of an element >= t
    // t threshold does not need to be present in the data
    // the algorithm guarantees only to place elements <t first and >=t later
    // it does not guarantee shrinkage of the problem so cannot be applied for quicksort
    // it t is greater than all values, end_idx is returned

    ASSERT(feat);
    ASSERT(indexes);
    ASSERT(end_idx >= start_idx);

    // copying the initial indexes
    size_t i = start_idx;
    size_t j = end_idx;

    while (i < j) {
        // searching for missplaced elements
        while (i < j && feat[i - start_idx] < t)
            i++;
        // dangerous - it may underflow
        while (i < j && feat[j - 1 - start_idx] >= t)
            j--;
        // swapping the elements
        if (i < j) {
            size_t tmp = indexes[i];
            indexes[i] = indexes[j - 1];
            indexes[j - 1] = tmp;
            double tmp_val = feat[i - start_idx];
            feat[i - start_idx] = feat[j - 1 - start_idx];
            feat[j - 1 - start_idx] = tmp_val;
            // moving forward
            ++i;
            --j;
        }
    }
    return i;
}

static size_t partition_quicksort(double* feat, size_t* indexes, size_t start_idx, size_t end_idx, size_t p_index) {
    // Require non-empty range
    // Require start_idx <= p_index < end_idx
    ASSERT(feat);
    ASSERT(indexes);
    ASSERT(end_idx > start_idx);
    ASSERT(p_index >= start_idx && p_index < end_idx);

    double pivot = feat[p_index - start_idx];

    // Move pivot to the end
    size_t pivot_idx = indexes[p_index];
    indexes[p_index] = indexes[end_idx - 1];
    indexes[end_idx - 1] = pivot_idx;

    feat[p_index - start_idx] = feat[end_idx - 1 - start_idx];
    feat[end_idx - 1 - start_idx] = pivot;

    // Partition excluding pivot
    size_t mid = partition(feat, indexes, start_idx, end_idx - 1, pivot);

    // Move pivot into final place
    indexes[end_idx - 1] = indexes[mid];
    indexes[mid] = pivot_idx;

    feat[end_idx - 1- start_idx] = feat[mid - start_idx];
    feat[mid - start_idx] = pivot;

    return mid;
}

static size_t pivot_idx(const double* feat, size_t* indexes, size_t start_idx, size_t end_idx) {
    ASSERT(feat);
    ASSERT(indexes);
    ASSERT(start_idx < end_idx);

    // selects 3 elements and chooses a median from them
    size_t mid = start_idx + (end_idx - start_idx) / 2;

    double a = feat[start_idx - start_idx];
    double b = feat[mid - start_idx];
    double c = feat[end_idx - 1 - start_idx];

    if ((a <= b && b <= c) || (c <= b && b <= a))
        return mid;
    if ((b <= a && a <= c) || (c <= a && a <= b))
        return start_idx;
    return end_idx - 1;
}

static void insertion_argsort(double* feat, size_t* indexes, size_t start_idx, size_t end_idx) {
    ASSERT(feat);
    ASSERT(indexes);

    for (size_t i = start_idx + 1; i < end_idx; i++) {
        size_t idx = indexes[i];
        double t = feat[i - start_idx];
        size_t j = i;

        while (j > start_idx && feat[j - 1 - start_idx] > t) {
            indexes[j] = indexes[j - 1];
            feat[j - start_idx] = feat[j - 1 - start_idx];
            j--;
        }

        indexes[j] = idx;
        feat[j - start_idx] = t;
    }
}

static void argsort(double* feat, size_t* indexes, size_t start_idx, size_t end_idx) {
    // gets X matrix as input, considers only given features and applies argsort on indexes array within given range
    // modifies indexes inplace
    // quicksort + insertion sort

    ASSERT(feat);
    ASSERT(indexes);

    // sort until there is only one element
    size_t n = end_idx - start_idx;
    while (n > 1) {
        // run insertion sort for small arrays
        if (n < QUICKSORT_NMIN) return insertion_argsort(feat, indexes, start_idx, end_idx);
        size_t p_index = pivot_idx(feat, indexes, start_idx, end_idx);
        size_t mid_idx = partition_quicksort(feat, indexes, start_idx, end_idx, p_index);
        if (mid_idx - start_idx < n - mid_idx - 1) {
            argsort(feat, indexes, start_idx, mid_idx);
            start_idx = mid_idx + 1;
        }
        else {
            argsort(feat, indexes, mid_idx + 1, end_idx);
            end_idx = mid_idx;
        }
        // update size
        n = end_idx - start_idx;
    }
}

static void shuffle_limit(size_t* array, size_t n, size_t limit, pcg32_random_t* rng) {
    // receives an array of size n and shuffles only the first limit elements
    ASSERT(array);
    ASSERT(n > 0);
    ASSERT(limit <= n);
    for (size_t i = 0; i < limit; i++) {
        // possibly think about more uniform shuffling
        size_t j = i + pcg32_random_r(rng) % (n - i);
        size_t tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

static void split(Node* node, const double* X, const size_t* y, size_t p, size_t c,
    size_t* indexes, size_t start_idx, size_t end_idx,
    impurity_func_t impurity_func, size_t max_height, size_t min_samples_split, size_t min_samples_leaf,
    size_t max_features, pcg32_random_t* rng) {
    // p is the number of features
    // c is the number of classes
    // impurity_func is gini or entropy
    // for given data X and for considered observation within range [start_idx, end_idx), we want to make an optimal split

    ASSERT(node);
    ASSERT(X);
    ASSERT(y);
    ASSERT(start_idx < end_idx);
    ASSERT(min_samples_leaf > 0);
    ASSERT(max_features > 0);
    ASSERT(max_features <= p);

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
    // we need to have at least min_samples_leaf samples in each leaf - otherwise we cannot split
    if (2 * min_samples_leaf > end_idx - start_idx)
        return;

    // store initial best impurity in the node
    // we want to minimize impurities
    double best_score = INFINITY;
    size_t n = end_idx - start_idx; // number of observations in a node

    // caching of feature column - it will be used for partitioning and sorting
    double* feat = malloc(n * sizeof(double));

    // shuffle features
    size_t MAX_FEATURES = MAX(1, MIN(max_features, p));
    idx_array shuffled_features = init_indexes(p);
    shuffle_limit(shuffled_features.data, shuffled_features.size, MAX_FEATURES, rng);

    // for each feature
    // condifer only max features
    for (size_t feature_idx = 0; feature_idx < MAX_FEATURES; feature_idx++) {
        size_t feature = shuffled_features.data[feature_idx];

        // copying the feature column for the considered observations
        for (size_t i = 0; i < n; i++)
            feat[i] = X[indexes[start_idx + i] * p + feature];

        // sort values for the feature
        argsort(feat, indexes, start_idx, end_idx);
        
        // initially all elements are in the right subtree
        // possibly allocate this only once and just fill with zeros
        idx_array left_counts = zeros_idx_array(c);
        idx_array right_counts = get_counts(c, y, indexes, start_idx, end_idx);

        // iterate over middle points as thresholds
        for (size_t i = 0; i < MIN(n - min_samples_leaf, n - 1); i++) {
            // move 1 element to the left subtree
            size_t class = y[indexes[start_idx + i]];
            left_counts.data[class]++;
            right_counts.data[class]--;

            // skip if we dont have enough samples in the left subtree
            if (i + 1 < min_samples_leaf)
                continue;

            // skip if we have identical values
            if (fabs(feat[i + 1] - feat[i]) < 1e-12)
                continue;

            // compute the new threshold
            double threshold = (feat[i] + feat[i + 1]) * 0.5;

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
    // freeing cached feature column
    free(shuffled_features.data);

    // recomputing probs - partition will be faster than sorting
    for (size_t i = 0; i < n; i++)
        feat[i] = X[indexes[start_idx + i] * p + node->feature];
    size_t mid_idx = partition(feat, indexes, start_idx, end_idx, node->threshold);
    free(feat);

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
    split(node->left, X, y, p, c, indexes, start_idx, mid_idx, impurity_func,
        max_height, min_samples_split, min_samples_leaf, max_features, rng);
    split(node->right, X, y, p, c, indexes, mid_idx, end_idx, impurity_func,
        max_height, min_samples_split, min_samples_leaf, max_features, rng);
}

// check if it will be needed
void free_tree(Node* root) {
    if (!root) return;
    if (root->left) free_tree(root->left);
    if (root->right) free_tree(root->right);
    root->left = NULL;
    root->right = NULL;
    free(root->probs.data);
    free(root);
}

double* predict_proba_one(Node* root, const double* x) {
    // predicts a class for only one observation
    
    // sanity check
    ASSERT(root);
    ASSERT(x);
    
    // going down to a leaf
    while (root->left || root->right) {
        // it may be dangerous when we try to access non-existing feature -> should be checked at the beginning
        if (x[root->feature] < root->threshold)
            root = root->left;
        else
            root = root->right;
    }
    
    ASSERT(root->probs.data);
    ASSERT(root->probs.size > 0);

    return root->probs.data;
}

double* tree_predict_proba(Node* root, const double* X, size_t n, size_t p) {
    size_t c = root->probs.size;
    // initialize y_pred
    double* ret = (double*)malloc(n * c * sizeof(double));
    // iterate over samples
    for (size_t i = 0; i < n; i++) {
        double* probs = predict_proba_one(root, &X[i * p]);
        for (size_t j = 0; j < c; j++) {
            ret[i * c + j] = probs[j];
        }
    }
    return ret;
}

// it works only for C contiguous arrays
static int64_t predict_one(Node* root, const double* x) {
    // predicts a class for only one observation
    
    // sanity check
    ASSERT(root);
    ASSERT(x);

    size_t c = root->probs.size;
    double* probs_array = predict_proba_one(root, x);
    
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
        ret[i] = predict_one(root, &X[i * p]);
    }
    return ret;
}

// for tests
static double* c_contiguous_array(double * const * X, size_t n, size_t p) {
    double* ret = (double*)malloc(n * p * sizeof(double));
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < p; j++) {
            ret[i * p + j] = X[i][j];
        }
    }
    return ret;
}

// for tests
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



// ----- DEBUG -----

static void print_indent(size_t depth) {
    for (size_t i = 0; i < depth; ++i)
        printf("  ");
}

// for tests
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


void tree_fit(Node** root, const double* X, const size_t* y, size_t n, size_t p, size_t c,
    impurity_func_t impurity_func, size_t max_height, size_t min_samples_split, size_t min_samples_leaf,
    size_t max_features, pcg32_random_t* rng) {
    // init indexes
    idx_array indexes = init_indexes(n); // maybe change the method to return just the array

    // initialize root
    idx_array counts_root = get_counts(c, y, indexes.data, 0, n);
    double_array probs_root = get_probs_from_counts(counts_root, n);
    double impurity_root = impurity_func(counts_root, n);
    free(counts_root.data);
    *root = init_node(0, &probs_root, impurity_root);

    // build the tree (we pass a fixed number of classes - useful for Random Forest consistency)
    split(*root, X, y, p, c, indexes.data, 0, n, impurity_func, max_height,
        min_samples_split, min_samples_leaf, max_features, rng);

    // free memory
    free(indexes.data);
}

// for tests
static void tree_test() {
    // random seed
    // srand(time(NULL));

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
    impurity_func_t impurity_func = gini_from_counts;
    // or entropy
    // double (*impurity_func)(idx_array, size_t) = entropy_from_counts;

    // idx_array counts_root = get_counts(c, y, indexes.data, 0, n);
    // double_array probs_root = get_probs_from_counts(counts_root, n);
    // double impurity_root = impurity_func(counts_root, n);
    // free(counts_root.data);

    // Node* root = init_node(0, &probs_root, impurity_root);
    // printf("Initial impurity: %f\n", root->impurity);

    Node* root = NULL;

    printf("Setting hyperparameters...\n");
    size_t max_height = INT_MAX; // no limit on the height
    size_t min_samples_split = 10;
    size_t min_samples_leaf = 5;
    size_t max_features = 4;
    // build the tree
    printf("Fitting the tree...\n");
    int random_state = 44; // for reproducibility
    pcg32_random_t rng;
    rng.state = random_state;
    rng.inc = INC_DEFAULT;
    pcg32_random_r(&rng); // advance state to avoid initial low-quality output

    tree_fit(&root, X, y, n, p, c, impurity_func, max_height, min_samples_split, min_samples_leaf, max_features, &rng);
    // split(root, X, y, p, c, indexes.data, 0, n, impurity_func, max_height, min_samples_split, min_samples_leaf, max_features, &rng);
    printf("The tree is fitted\n");

    printf("Printing the tree\n");
    // inspect_tree_node(root, max_height);

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

    printf("Running tree test...\n");
    tree_test();

    printf("Program finished\n");
    
    return 0;
}

