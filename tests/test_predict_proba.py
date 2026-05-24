import numpy as np
from classtrees import ClassTree


def make_dataset():
    X = np.random.randn(20, 3)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def test_proba_shape():
    X, y = make_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    proba = tree.predict_proba(X)

    assert proba.shape == (X.shape[0], len(np.unique(y)))


def test_proba_sums_to_one():
    X, y = make_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    proba = tree.predict_proba(X)

    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_proba_non_negative():
    X, y = make_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    proba = tree.predict_proba(X)

    assert (proba >= 0).all()

def test_proba_less_than_one():
    X, y = make_dataset()

    tree = ClassTree()
    tree.fit(X, y)

    proba = tree.predict_proba(X)

    assert (proba <= 1).all()