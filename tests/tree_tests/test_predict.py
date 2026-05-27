import numpy as np
from classtrees import ClassTree
import pytest

def test_predict_before_fit():
    tree = ClassTree()
    with pytest.raises(Exception):
        tree.predict(np.random.randn(5, 3))


def test_predict_invalid_input():
    tree = ClassTree()
    X = np.random.randn(10, 3)
    y = (X[:, 0] > 0).astype(int)

    tree.fit(X, y)

    with pytest.raises(Exception):
        tree.predict("invalid")

    with pytest.raises(Exception):
        tree.predict([[1, 2]])  # wrong shape

def test_predict_wrong_feature_dimension():
    tree = ClassTree()

    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)

    tree.fit(X, y)

    X_bad = np.random.randn(10, 5)  # wrong feature count

    with pytest.raises(Exception):
        tree.predict(X_bad)

def test_predict_empty_input():
    tree = ClassTree()

    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)

    tree.fit(X, y)

    X_empty = np.empty((0, 3))

    with pytest.raises(Exception):
        preds = tree.predict(X_empty)

def test_predict_float64_input():
    tree = ClassTree()

    X = np.random.randn(30, 3).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)

    tree.fit(X, y)

    preds = tree.predict(X)

    assert isinstance(preds, np.ndarray)

def test_predict_object_dtype():
    tree = ClassTree()

    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)

    tree.fit(X, y)

    X_obj = X.astype(object)

    with pytest.raises(Exception):
        tree.predict(X_obj)
    
def test_predict_deterministic():
    tree = ClassTree()

    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)

    tree.fit(X, y)

    p1 = tree.predict(X)
    p2 = tree.predict(X)

    np.testing.assert_array_equal(p1, p2)

def test_predict_only_seen_labels():
    tree = ClassTree()

    X = np.random.randn(50, 3)
    y = np.array([0, 1] * 25)

    tree.fit(X, y)

    preds = tree.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})

def test_predict_before_fit():
    tree = ClassTree()

    X = np.random.randn(5, 3)

    with pytest.raises(Exception):
        tree.predict(X)

