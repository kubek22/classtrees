#include "tree_bootstrap.h"
#include "assert.h"
#include <math.h>
#include <stdlib.h>

#define QUICKSORT_NMIN 40

#define MAX(A, B) (((A)>(B))?(A):(B))
#define MIN(A, B) (((A)<(B))?(A):(B))

Node* init_node(size_t h, double_array* probs, double impurity);

static idx_array init_indexes(size_t size) {
    idx_array ret;
    ret.size = size;
    ret.data = (size_t*)malloc(size * sizeof(size_t));
    for (size_t i = 0; i < size; i++) {
        ret.data[i] = i;
    }
    return ret;
}

static idx_array zeros_idx_array(size_t size) {
    idx_array ret;
    ret.size = size;
    ret.data = (size_t*)malloc(size * sizeof(size_t));
    for (size_t i = 0; i < size; i++) {
        ret.data[i] = 0;
    }
    return ret;
}

static idx_array get_counts_bootstrap(size_t c, const size_t* y, const size_t* bootstrap_counts,
    size_t* indexes, size_t start_idx, size_t end_idx)
{
    ASSERT(y);
    ASSERT(bootstrap_counts);
    ASSERT(indexes);
    ASSERT(end_idx - start_idx >= 1);

    idx_array ret = zeros_idx_array(c);
    for (size_t i = start_idx; i < end_idx; i++) {
        size_t row = indexes[i];
        size_t class = y[row];
        ASSERT(class < c);
        ret.data[class] += bootstrap_counts[row];
    }
    return ret;
}

static size_t get_weight_bootstrap(const size_t* bootstrap_counts, size_t* indexes,
    size_t start_idx, size_t end_idx)
{
    ASSERT(bootstrap_counts);
    ASSERT(indexes);

    size_t ret = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        ret += bootstrap_counts[indexes[i]];
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
    for (size_t i = 0; i < ret.size - 1; i++) {
        ret.data[i] = (double)counts.data[i] / n;
        sum += ret.data[i];
    }
    ret.data[ret.size - 1] = 1.0 - sum;
    return ret;
}

static size_t partition(double* feat, size_t* indexes, size_t start_idx, size_t end_idx,
    double t, size_t feat_base_idx)
{
    ASSERT(feat);
    ASSERT(indexes);
    ASSERT(end_idx >= start_idx);

    size_t i = start_idx;
    size_t j = end_idx;

    while (i < j) {
        while (i < j && feat[i - feat_base_idx] < t)
            i++;
        while (i < j && feat[j - 1 - feat_base_idx] >= t)
            j--;
        if (i < j) {
            size_t tmp = indexes[i];
            indexes[i] = indexes[j - 1];
            indexes[j - 1] = tmp;

            double tmp_val = feat[i - feat_base_idx];
            feat[i - feat_base_idx] = feat[j - 1 - feat_base_idx];
            feat[j - 1 - feat_base_idx] = tmp_val;

            ++i;
            --j;
        }
    }
    return i;
}

static size_t partition_quicksort(double* feat, size_t* indexes, size_t start_idx,
    size_t end_idx, size_t p_index, size_t feat_base_idx)
{
    ASSERT(feat);
    ASSERT(indexes);
    ASSERT(end_idx > start_idx);
    ASSERT(p_index >= start_idx && p_index < end_idx);

    double pivot = feat[p_index - feat_base_idx];

    size_t pivot_idx = indexes[p_index];
    indexes[p_index] = indexes[end_idx - 1];
    indexes[end_idx - 1] = pivot_idx;

    feat[p_index - feat_base_idx] = feat[end_idx - 1 - feat_base_idx];
    feat[end_idx - 1 - feat_base_idx] = pivot;

    size_t mid = partition(feat, indexes, start_idx, end_idx - 1, pivot, feat_base_idx);

    indexes[end_idx - 1] = indexes[mid];
    indexes[mid] = pivot_idx;

    feat[end_idx - 1 - feat_base_idx] = feat[mid - feat_base_idx];
    feat[mid - feat_base_idx] = pivot;

    return mid;
}

static size_t pivot_idx(const double* feat, size_t* indexes, size_t start_idx,
    size_t end_idx, size_t feat_base_idx)
{
    ASSERT(feat);
    ASSERT(indexes);
    ASSERT(start_idx < end_idx);

    size_t mid = start_idx + (end_idx - start_idx) / 2;

    double a = feat[start_idx - feat_base_idx];
    double b = feat[mid - feat_base_idx];
    double c = feat[end_idx - 1 - feat_base_idx];

    if ((a <= b && b <= c) || (c <= b && b <= a))
        return mid;
    if ((b <= a && a <= c) || (c <= a && a <= b))
        return start_idx;
    return end_idx - 1;
}

static void insertion_argsort(double* feat, size_t* indexes, size_t start_idx,
    size_t end_idx, size_t feat_base_idx)
{
    ASSERT(feat);
    ASSERT(indexes);

    for (size_t i = start_idx + 1; i < end_idx; i++) {
        size_t idx = indexes[i];
        double t = feat[i - feat_base_idx];
        size_t j = i;

        while (j > start_idx && feat[j - 1 - feat_base_idx] > t) {
            indexes[j] = indexes[j - 1];
            feat[j - feat_base_idx] = feat[j - 1 - feat_base_idx];
            j--;
        }

        indexes[j] = idx;
        feat[j - feat_base_idx] = t;
    }
}

static void argsort(double* feat, size_t* indexes, size_t start_idx, size_t end_idx,
    size_t feat_base_idx)
{
    ASSERT(feat);
    ASSERT(indexes);

    size_t n = end_idx - start_idx;
    while (n > 1) {
        if (n < QUICKSORT_NMIN)
            return insertion_argsort(feat, indexes, start_idx, end_idx, feat_base_idx);

        size_t p_index = pivot_idx(feat, indexes, start_idx, end_idx, feat_base_idx);
        size_t mid_idx = partition_quicksort(feat, indexes, start_idx, end_idx,
            p_index, feat_base_idx);

        if (mid_idx - start_idx < n - mid_idx - 1) {
            argsort(feat, indexes, start_idx, mid_idx, feat_base_idx);
            start_idx = mid_idx + 1;
        }
        else {
            argsort(feat, indexes, mid_idx + 1, end_idx, feat_base_idx);
            end_idx = mid_idx;
        }
        n = end_idx - start_idx;
    }
}

static void shuffle_limit(size_t* array, size_t n, size_t limit, pcg32_random_t* rng) {
    ASSERT(array);
    ASSERT(n > 0);
    ASSERT(limit <= n);
    for (size_t i = 0; i < limit; i++) {
        size_t j = i + pcg32_random_r(rng) % (n - i);
        size_t tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

static void split_bootstrap(Node* node, const double* X, const size_t* y,
    const size_t* bootstrap_counts, size_t p, size_t c, size_t* indexes,
    size_t start_idx, size_t end_idx, impurity_func_t impurity_func,
    size_t max_height, size_t min_samples_split, size_t min_samples_leaf,
    size_t max_features, pcg32_random_t* rng)
{
    ASSERT(node);
    ASSERT(X);
    ASSERT(y);
    ASSERT(bootstrap_counts);
    ASSERT(start_idx < end_idx);
    ASSERT(min_samples_leaf > 0);
    ASSERT(max_features > 0);
    ASSERT(max_features <= p);

    if (end_idx - start_idx < 2)
        return;
    if (node->impurity < 1e-12)
        return;
    if (node->h >= max_height)
        return;

    size_t n = end_idx - start_idx;
    size_t weighted_n = get_weight_bootstrap(bootstrap_counts, indexes, start_idx, end_idx);

    if (weighted_n < min_samples_split)
        return;
    if (2 * min_samples_leaf > weighted_n)
        return;

    double best_score = INFINITY;

    double* feat = (double*)malloc(n * sizeof(double));

    size_t MAX_FEATURES = MAX(1, MIN(max_features, p));
    idx_array shuffled_features = init_indexes(p);
    shuffle_limit(shuffled_features.data, shuffled_features.size, MAX_FEATURES, rng);

    for (size_t feature_idx = 0; feature_idx < MAX_FEATURES; feature_idx++) {
        size_t feature = shuffled_features.data[feature_idx];

        for (size_t i = 0; i < n; i++) {
            size_t row = indexes[start_idx + i];
            feat[i] = X[row * p + feature];
        }

        argsort(feat, indexes, start_idx, end_idx, start_idx);

        idx_array left_counts = zeros_idx_array(c);
        idx_array right_counts = get_counts_bootstrap(c, y, bootstrap_counts, indexes, start_idx, end_idx);

        size_t n_left = 0;
        size_t n_right = weighted_n;

        for (size_t i = 0; i < n - 1; i++) {
            // accessing a row in the original data
            size_t row = indexes[start_idx + i];

            // accessing the count of this row in the bootstrap sample
            size_t count = bootstrap_counts[row];

            // skip if this row is not in the bootstrap sample
            if (count == 0)
                continue;

            size_t class = y[row];
            left_counts.data[class] += count;
            right_counts.data[class] -= count;

            n_left += count;
            n_right -= count;

            if (n_left < min_samples_leaf)
                continue;
            
            if (n_right < min_samples_leaf)
                break;

            size_t next_i = i + 1;
            while (next_i < n && bootstrap_counts[indexes[start_idx + next_i]] == 0)
                next_i++;

            if (next_i == n)
                break;

            if (fabs(feat[next_i] - feat[i]) < 1e-12)
                continue;

            double threshold = (feat[i] + feat[next_i]) * 0.5;

            double impurity_left = impurity_func(left_counts, n_left);
            double impurity_right = impurity_func(right_counts, n_right);

            double p_left = (double)(n_left) / weighted_n;
            double p_right = 1.0 - p_left;

            double score = p_left * impurity_left + p_right * impurity_right;

            if (best_score > score) {
                best_score = score;
                node->feature = feature;
                node->threshold = threshold;
            }
        }

        free(left_counts.data);
        free(right_counts.data);
    }

    free(shuffled_features.data);

    if (best_score == INFINITY) {
        free(feat);
        return;
    }

    for (size_t i = 0; i < n; i++) {
        size_t row = indexes[start_idx + i];
        feat[i] = X[row * p + node->feature];
    }
    size_t mid_idx = partition(feat, indexes, start_idx, end_idx, node->threshold, start_idx);
    free(feat);

    idx_array counts_left = get_counts_bootstrap(c, y, bootstrap_counts, indexes, start_idx, mid_idx);
    idx_array counts_right = get_counts_bootstrap(c, y, bootstrap_counts, indexes, mid_idx, end_idx);

    size_t weighted_n_left = get_weight_bootstrap(bootstrap_counts, indexes, start_idx, mid_idx);
    size_t weighted_n_right = weighted_n - weighted_n_left;

    double impurity_left = impurity_func(counts_left, weighted_n_left);
    double impurity_right = impurity_func(counts_right, weighted_n_right);

    double_array probs_left = get_probs_from_counts(counts_left, weighted_n_left);
    double_array probs_right = get_probs_from_counts(counts_right, weighted_n_right);

    free(counts_left.data);
    free(counts_right.data);

    node->left = init_node(node->h + 1, &probs_left, impurity_left);
    node->right = init_node(node->h + 1, &probs_right, impurity_right);

    split_bootstrap(node->left, X, y, bootstrap_counts, p, c, indexes, start_idx, mid_idx,
        impurity_func, max_height, min_samples_split, min_samples_leaf, max_features, rng);
    split_bootstrap(node->right, X, y, bootstrap_counts, p, c, indexes, mid_idx, end_idx,
        impurity_func, max_height, min_samples_split, min_samples_leaf, max_features, rng);
}

void tree_fit_bootstrap(Node** root, const double* X, const size_t* y, const size_t* bootstrap_counts,
    size_t n, size_t p, size_t c, impurity_func_t impurity_func, size_t max_height,
    size_t min_samples_split, size_t min_samples_leaf, size_t max_features, pcg32_random_t* rng)
{
    ASSERT(root);
    ASSERT(X);
    ASSERT(y);
    ASSERT(bootstrap_counts);
    ASSERT(n > 0);
    ASSERT(p > 0);
    ASSERT(c > 0);

    idx_array indexes = init_indexes(n);

    idx_array counts_root = get_counts_bootstrap(c, y, bootstrap_counts, indexes.data, 0, n);
    size_t weighted_n = get_weight_bootstrap(bootstrap_counts, indexes.data, 0, n);
    double_array probs_root = get_probs_from_counts(counts_root, weighted_n);
    double impurity_root = impurity_func(counts_root, weighted_n);
    free(counts_root.data);

    *root = init_node(0, &probs_root, impurity_root);

    split_bootstrap(*root, X, y, bootstrap_counts, p, c, indexes.data, 0, n,
        impurity_func, max_height, min_samples_split, min_samples_leaf, max_features, rng);

    free(indexes.data);
}
