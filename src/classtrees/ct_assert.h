#ifndef CT_ASSERT_H
#define CT_ASSERT_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef NDEBUG

#define ASSERT(EXPR) ((void)0)
#define ASSERT_DOUBLE_EQ(A, B, EPS) ((void)0)

#else

#define ASSERT(EXPR) do { \
    if (!(EXPR)) { \
        fprintf(stderr, "Assertion failed: %s (%s:%d)\n", \
                #EXPR, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define ASSERT_DOUBLE_EQ(A, B, EPS) do { \
    double a_ = (A); \
    double b_ = (B); \
    if (fabs(a_ - b_) > (EPS)) { \
        fprintf(stderr, \
                "Assertion failed: %s == %s (got %.12f vs %.12f) (%s:%d)\n", \
                #A, #B, a_, b_, __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#endif

#endif