#include "tree.h"
#include "assert.h"
#include "random.h"
#include <omp.h>


// TODO repeated with tree.c
#define MAX(A, B) (((A)>(B))?(A):(B))
#define MIN(A, B) (((A)<(B))?(A):(B))


static int get_num_threads(int n_jobs) {
    if (n_jobs == -1) return omp_get_max_threads();
    else return MAX(1, MIN(n_jobs, omp_get_max_threads()));
}

static void bootstrap_sample(const double* X, const int64_t* y, size_t n, size_t p,
    pcg32_random_t* rng, double** new_X, size_t** new_y)
{
    ASSERT(X);
    ASSERT(y);
    ASSERT(n > 0);
    ASSERT(p > 0);
    ASSERT(rng);

    *new_X = (double*)malloc(n * p * sizeof(double));
    *new_y = (size_t*)malloc(n * sizeof(size_t));

    ASSERT(*new_X);
    ASSERT(*new_y);

    for (size_t i = 0; i < n; i++) {
        size_t j = pcg32_random_r(rng) % n;

        for (size_t k = 0; k < p; k++) {
            (*new_X)[i * p + k] = X[j * p + k];
        }

        (*new_y)[i] = (size_t)y[j];
    }
}

void rf_fit(Node** roots, size_t n_estimators, const double* X, const int64_t* y,
    size_t n, size_t p, size_t c, impurity_func_t impurity_func, size_t max_height,
    size_t min_samples_split, size_t min_samples_leaf, size_t max_features,
    pcg32_random_t* rngs, int n_jobs)
{
    // we assume that rngs[i] belongs to i-th tree and is already initialzed
    ASSERT(roots);
    ASSERT(n_estimators > 0);
    ASSERT(X);
    ASSERT(y);
    ASSERT(n > 0);
    ASSERT(p > 0);
    ASSERT(c > 0);
    ASSERT(impurity_func);
    ASSERT(max_height >= 0);
    ASSERT(min_samples_split >= 2);
    ASSERT(min_samples_leaf >= 1);
    ASSERT(0 < max_features && max_features <= p);
    ASSERT(rngs);
    ASSERT(n_jobs == -1 || n_jobs > 0);


    int threads = get_num_threads(n_jobs);

    // iterate over estimators
    // TODO use n_jobs
    #pragma omp parallel for num_threads(threads)
    for (size_t i = 0; i < n_estimators; i++) {
        // generate bootstrap sample
        double* X_boot;
        size_t* y_boot;
        bootstrap_sample(X, y, n, p, &rngs[i], &X_boot, &y_boot);

        // fit a tree (roots will be initialized here)
        tree_fit(&roots[i], X_boot, y_boot, n, p, c, impurity_func,
            max_height, min_samples_split, min_samples_leaf, max_features, &rngs[i]);

        // free the memory
        free(X_boot);
        free(y_boot);
    }
}

double* rf_predict_proba(Node** roots, size_t n_estimators, const double* X,
    size_t n, size_t p, size_t c, int n_jobs)
{
    double* ret = (double*)malloc(n * c * sizeof(double));
    // if (!ret) return NULL;
    // for (size_t i = 0; i < n * c; i++) ret[i] = 0.0;

    // for (size_t i = 0; i < n; i++) {
    //     for (size_t j = 0; j < n_estimators; j++) {
    //         // it returnes only a pointer to leaf probs, it should not be freed
    //         double* x_probs = predict_proba_one(roots[j], &X[i * p]);
    //         if (!x_probs) {
    //             free(ret);
    //             return NULL;
    //         }
    //         for (size_t k = 0; k < c; k++) {
    //             ret[i * c + k] += x_probs[k];
    //         }
    //     }
    //     for (size_t k = 0; k < c; k++) {
    //         ret[i * c + k] /= (double)n_estimators;
    //     }
    // }
    return ret;
}

int64_t* rf_predict(Node** roots, size_t n_estimators, const double* X,
                    size_t n, size_t p, size_t c, int n_jobs)
{
    // double* probs = rf_predict_proba(roots, n_estimators, X, n, p, c, n_jobs);
    // if (!probs)
    //     return NULL;

    int64_t* ret = (int64_t*)malloc(n * sizeof(int64_t));
    if (!ret) {
        // free(probs);
        return NULL;
    }

    // int threads = get_num_threads(n_jobs);

    // #pragma omp parallel for num_threads(threads)
    // for (size_t i = 0; i < n; i++) {
    //     ret[i] = 0;
    //     double max_prob = probs[i * c];
    //     for (size_t j = 1; j < c; j++) {
    //         if (max_prob < probs[i * c + j]) {
    //             ret[i] = (int64_t)j;
    //             max_prob = probs[i * c + j];
    //         }
    //     }
    // }

    // free(probs);
    return ret;
}

