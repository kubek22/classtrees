import numpy as np
import pytest
from classtrees import RandomForest


def test_fit_none_input():
    rf = RandomForest()
    with pytest.raises(Exception):
        rf.fit(None, None)

def test_fit_empty():
    rf = RandomForest()
    X = np.empty((0, 3))
    y = np.array([], dtype=np.int64)
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_fit_shape_mismatch():
    rf = RandomForest()
    X = np.random.randn(10, 3)
    y = np.array([0, 1], dtype=np.int64)
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_fit_nan_inf():
    rf = RandomForest()
    X = np.array([[np.nan, 1, 2]])
    y = np.array([0], dtype=np.int64)
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_single_class():
    X = np.random.randn(20, 3)
    y = np.zeros(20, dtype=np.int64)
    rf = RandomForest()
    rf.fit(X, y)

def test_fit_float_labels_rejected():
    X = np.random.randn(10, 3)
    y = np.zeros(10, dtype=np.float64)
    rf = RandomForest()
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_fit_mixed_dtype_X():
    X = np.array([[1, 2, 3],
                  [4.0, 5, 6]], dtype=object)
    y = np.array([0, 1])
    rf = RandomForest()
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_fit_inf():
    X = np.array([[np.inf, 1, 2]])
    y = np.array([0], dtype=np.int64)
    rf = RandomForest()
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_fit_negative_inf():
    X = np.array([[-np.inf, 1, 2]])
    y = np.array([0], dtype=np.int64)
    rf = RandomForest()
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_fit_nan_in_multiple_rows():
    X = np.random.randn(10, 3)
    X[3, 1] = np.nan
    y = np.random.randint(0, 2, size=10)
    rf = RandomForest()
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_fit_single_sample():
    X = np.random.randn(1, 3)
    y = np.array([1], dtype=np.int64)
    rf = RandomForest()
    rf.fit(X, y)

def test_fit_single_feature():
    X = np.random.randn(20, 1)
    y = np.random.randint(0, 2, size=20)
    rf = RandomForest()
    rf.fit(X, y)

def test_double_fit():
    X = np.random.randn(20, 1)
    y = np.random.randint(0, 2, size=20)
    rf = RandomForest()
    rf.fit(X, y)
    with pytest.raises(Exception):
        rf.fit(X, y)

def test_fit_all_same_features():
    X = np.ones((20, 3))
    y = np.random.randint(0, 2, size=20)
    rf = RandomForest(max_height=5)
    rf.fit(X, y)  # must not crash / infinite loop

def test_fit_binary_labels():
    X = np.random.randn(20, 3)
    y = np.random.choice([0, 1], size=20)
    rf = RandomForest()
    rf.fit(X, y)

def test_fit_non_contiguous_labels():
    X = np.random.randn(20, 3)
    y = np.array([0, 1] * 10, dtype=np.int64)[::1]
    rf = RandomForest()
    rf.fit(X, y)

def test_fit_produces_valid_model():
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, size=20)
    rf = RandomForest()
    rf.fit(X, y)
    preds = rf.predict(X)
    assert len(preds) == len(y)

def test_fit_non_contiguous_arrays():
    # Create contiguous base array
    X_base = np.random.randn(20, 5)
    y_base = np.random.randint(0, 2, size=20)

    # Make non-contiguous views
    X = X_base[:, ::2]   # stride view (non-contiguous)
    y = y_base[::2]      # also non-contiguous

    assert not X.flags["C_CONTIGUOUS"]
    assert not y.flags["C_CONTIGUOUS"]

    rf = RandomForest()

    with pytest.raises(Exception):
        rf.fit(X, y)
