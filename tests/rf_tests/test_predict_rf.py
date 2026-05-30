import numpy as np
from classtrees import RandomForest
import pytest

def test_predict_before_fit():
    rf = RandomForest(n_estimators=3)
    with pytest.raises(Exception):
        rf.predict(np.random.randn(5, 3))


def test_predict_invalid_input():
    rf = RandomForest(n_estimators=3)
    X = np.random.randn(10, 3)
    y = (X[:, 0] > 0).astype(int)

    rf.fit(X, y)

    with pytest.raises(Exception):
        rf.predict("invalid")

    with pytest.raises(Exception):
        rf.predict([[1, 2]])  # wrong shape

def test_predict_wrong_feature_dimension():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)

    rf.fit(X, y)

    X_bad = np.random.randn(10, 5)  # wrong feature count

    with pytest.raises(Exception):
        rf.predict(X_bad)

def test_predict_empty_input():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)

    rf.fit(X, y)

    X_empty = np.empty((0, 3))

    with pytest.raises(Exception):
        preds = rf.predict(X_empty)

def test_predict_float64_input():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(30, 3).astype(np.float64)
    y = (X[:, 0] > 0).astype(int)

    rf.fit(X, y)

    preds = rf.predict(X)

    assert isinstance(preds, np.ndarray)

def test_predict_object_dtype():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)

    rf.fit(X, y)

    X_obj = X.astype(object)

    with pytest.raises(Exception):
        rf.predict(X_obj)
    
def test_predict_deterministic():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)

    rf.fit(X, y)

    p1 = rf.predict(X)
    p2 = rf.predict(X)

    np.testing.assert_array_equal(p1, p2)

def test_predict_only_seen_labels():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(50, 3)
    y = np.array([0, 1] * 25, dtype=np.int64)

    rf.fit(X, y)

    preds = rf.predict(X)

    assert set(np.unique(preds)).issubset({0, 1})

def test_predict_before_fit():
    rf = RandomForest(n_estimators=3)

    X = np.random.randn(5, 3)

    with pytest.raises(Exception):
        rf.predict(X)

