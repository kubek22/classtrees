import numpy as np
import pytest
from classtrees import ClassTree


# -------------------------
# Helpers
# -------------------------

def make_simple_dataset(n=20):
    """
    Binary classification dataset:
    y = 1 if sum(x) > threshold else 0
    """
    X = np.random.randn(n, 3).astype(np.float64)
    y = (X.sum(axis=1) > 0).astype(np.uint64)
    return X, y


# -------------------------
# Constructor tests
# -------------------------

def test_tree_initialization_defaults():
    tree = ClassTree()
    assert tree is not None


def test_tree_initialization_params():
    tree = ClassTree(
        impurity="gini",
        max_height=5,
        min_samples_split=3
    )
    assert tree is not None


def test_tree_invalid_impurity_should_fail():
    with pytest.raises(Exception):
        ClassTree(impurity="unknown")


# -------------------------
# Fit tests
# -------------------------

def test_fit_runs_without_crash():
    X, y = make_simple_dataset()

    tree = ClassTree()
    tree.fit(X, y)  # should not crash


def test_fit_accepts_numpy_contiguous():
    X = np.ascontiguousarray(np.random.randn(10, 4))
    y = np.ascontiguousarray(np.random.randint(0, 2, size=10)).astype(np.uint64)

    tree = ClassTree()
    tree.fit(X, y)


# -------------------------
# Predict tests
# -------------------------

def test_predict_returns_array():
    X, y = make_simple_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    preds = tree.predict(X)

    assert isinstance(preds, np.ndarray)


def test_predict_shape_matches_input():
    X, y = make_simple_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    preds = tree.predict(X)

    assert preds.shape[0] == X.shape[0]


# -------------------------
# Stability / regression-style test
# -------------------------

def test_deterministic_behavior():
    np.random.seed(42)
    X, y = make_simple_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    p1 = tree.predict(X)
    p2 = tree.predict(X)

    np.testing.assert_array_equal(p1, p2)
