import numpy as np
from classtrees import ClassTree


def test_single_sample():
    X = np.random.randn(1, 3)
    y = np.array([1])

    tree = ClassTree()
    tree.fit(X, y)
    assert tree.predict(X).shape == (1,)


def test_two_class_gap_labels():
    X = np.random.randn(50, 3)
    y = np.random.choice([0, 5], size=50)

    tree = ClassTree()
    tree.fit(X, y)

    preds = tree.predict(X)

    assert set(np.unique(preds)).issubset({0, 5})


def test_large_feature_gap():
    X = np.random.randn(20, 10)
    y = (X[:, 0] > 0).astype(int)

    tree = ClassTree()
    tree.fit(X, y)

    tree.predict(X)