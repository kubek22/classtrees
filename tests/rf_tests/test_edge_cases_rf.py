import numpy as np
import time
from classtrees import RandomForest


def test_no_infinite_loop_and_finite_structure():
    X = np.random.randn(2000, 10)
    y = (X[:, 0] > 0).astype(int)

    rf = RandomForest(n_estimators=10, max_height=10)

    start = time.time()
    rf.fit(X, y)
    duration = time.time() - start

    # must finish quickly
    assert duration < 3.5

def test_prediction_variance_exists():
    X = np.random.randn(1000, 5)
    y = (X[:, 0] > 0).astype(int)

    rf = RandomForest(n_estimators=10, max_height=10)
    rf.fit(X, y)

    preds = rf.predict(X)

    # not all identical (would indicate broken rf)
    assert len(np.unique(preds)) > 1