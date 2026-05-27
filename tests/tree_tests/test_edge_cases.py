import numpy as np
import time
from classtrees import ClassTree


def test_no_infinite_loop_and_finite_structure():
    X = np.random.randn(2000, 10)
    y = (X[:, 0] > 0).astype(int)

    tree = ClassTree(max_height=10)

    start = time.time()
    tree.fit(X, y)
    duration = time.time() - start

    # must finish quickly
    assert duration < 1.5

def test_prediction_variance_exists():
    X = np.random.randn(1000, 5)
    y = (X[:, 0] > 0).astype(int)

    tree = ClassTree(max_height=10)
    tree.fit(X, y)

    preds = tree.predict(X)

    # not all identical (would indicate broken tree)
    assert len(np.unique(preds)) > 1