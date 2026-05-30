import numpy as np
from classtrees import RandomForest
import pytest


def test_proba_properties():
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)

    rf = RandomForest(n_estimators=3)
    rf.fit(X, y)

    proba = rf.predict_proba(X)

    assert np.all(proba >= 0)
    assert np.all(proba <= 1)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_proba_shape():
    X = np.random.randn(30, 3)
    y = (X[:, 0] > 0).astype(int)

    rf = RandomForest(n_estimators=3)
    rf.fit(X, y)

    proba = rf.predict_proba(X)
    assert proba.shape == (len(X), len(np.unique(y)))
    
def test_proba_class_order_stability():
    X = np.random.randn(100, 3)
    y = np.array([1, 0] * 50, dtype=np.int64)  # deliberately unordered

    rf = RandomForest(n_estimators=3)
    rf.fit(X, y)

    proba = rf.predict_proba(X)

    # probabilities must align with sorted unique classes (assumption)
    classes = np.unique(y)
    assert proba.shape[1] == len(classes)

def test_proba_single_class():
    X = np.random.randn(50, 3)
    y = np.zeros(50, dtype=int)

    rf = RandomForest(n_estimators=3)
    rf.fit(X, y)

    proba = rf.predict_proba(X)

    assert proba.shape == (50, 1)
    np.testing.assert_allclose(proba[:, 0], 1.0, atol=1e-6)


def test_proba_row_normalization_stability():
    X = np.random.randn(60, 3)
    y = (X[:, 0] > 0).astype(int)

    rf = RandomForest(n_estimators=3)
    rf.fit(X, y)

    proba = rf.predict_proba(X)

    row_sums = proba.sum(axis=1)

    np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

def test_proba_invalid_input():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)

    rf.fit(X, y)

    with pytest.raises(Exception):
        rf.predict_proba("invalid")

    with pytest.raises(Exception):
        rf.predict_proba([[1, 2]])
    
def test_proba_empty_input():
    X = np.random.randn(30, 3)
    y = (X[:, 0] > 0).astype(int)

    rf = RandomForest(n_estimators=3)
    rf.fit(X, y)

    X_empty = np.empty((0, 3))

    with pytest.raises(Exception):
        rf.predict_proba(X_empty)

def test_predict_proba_before_fit():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(5, 3)

    with pytest.raises(Exception):
        rf.predict_proba(X)
