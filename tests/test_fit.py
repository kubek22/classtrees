import numpy as np
import pytest
from classtrees import ClassTree


def test_fit_runs():
    X = np.random.randn(20, 3)
    y = (X.sum(axis=1) > 0).astype(np.int64)

    tree = ClassTree()
    tree.fit(X, y)


def test_fit_empty_X():
    tree = ClassTree()

    X = np.empty((0, 3))
    y = np.array([])

    with pytest.raises(Exception):
        tree.fit(X, y)


def test_fit_mismatched_shapes():
    tree = ClassTree()

    X = np.random.randn(10, 3)
    y = np.array([0, 1])  # wrong

    with pytest.raises(Exception):
        tree.fit(X, y)


def test_fit_single_class():
    X = np.random.randn(10, 3)
    y = np.zeros(10, dtype=np.int64)

    tree = ClassTree()
    tree.fit(X, y)  # should not crash