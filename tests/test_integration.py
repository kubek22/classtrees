import numpy as np
from classtrees import ClassTree


def test_accuracy_above_random():
    X = np.random.randn(200, 3)
    y = (X[:, 0] > 0).astype(int)

    tree = ClassTree(max_height=5)
    tree.fit(X, y)

    preds = tree.predict(X)

    acc = (preds == y).mean()

    assert acc > 0.6

