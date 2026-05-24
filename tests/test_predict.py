import numpy as np
import pytest
from classtrees import ClassTree


def make_dataset():
    X = np.random.randn(30, 3)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def test_predict_before_fit():
    tree = ClassTree()
    X = np.random.randn(5, 3)

    with pytest.raises(Exception):
        tree.predict(X)


def test_predict_shape():
    X, y = make_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    preds = tree.predict(X)

    assert preds.shape == (X.shape[0],)


def test_predict_dtype():
    X, y = make_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    preds = tree.predict(X)

    assert preds.dtype == np.int64


def test_predict_invalid_input():
    tree = ClassTree()
    tree.fit(*make_dataset())

    with pytest.raises(Exception):
        tree.predict("not an array")

    with pytest.raises(Exception):
        tree.predict([[1, 2, 3]])