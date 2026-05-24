import numpy as np
import pytest
from classtrees import ClassTree


def test_wrong_dtype_X():
    tree = ClassTree()

    X = np.array([["a", "b"], ["c", "d"]])
    y = np.array([0, 1])

    with pytest.raises(Exception):
        tree.fit(X, y)


def test_wrong_dtype_y():
    tree = ClassTree()

    X = np.random.randn(10, 3)
    y = np.array(["a", "b", "c", "d", "e"])

    with pytest.raises(Exception):
        tree.fit(X, y)


def test_none_input():
    tree = ClassTree()

    with pytest.raises(Exception):
        tree.fit(None, None)


def test_nan_input():
    X = np.array([[np.nan, 1, 2]])
    y = np.array([0])

    tree = ClassTree()

    with pytest.raises(Exception):
        tree.fit(X, y)
