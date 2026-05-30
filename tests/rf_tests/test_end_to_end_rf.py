import numpy as np
import pytest
from classtrees import RandomForest


@pytest.mark.parametrize(
    "n_estimators,impurity,max_height,min_samples_split,min_samples_leaf,max_features,random_state,n_jobs",
    [
        (10, "gini", None, 2, 1, None, None, 1),
        (1, "gini", 5, 2, 1, None, 42, 100),
        (5, "entropy", 5, 2, 1, 3, 123, 10),
        (100, "gini", 10, 5, 2, 4, 0, -1),
    ],
)
def test_fit_param_combinations_sanity(
    n_estimators,
    impurity,
    max_height,
    min_samples_split,
    min_samples_leaf,
    max_features,
    random_state,
    n_jobs
):
    X = np.random.randn(200, 6)
    y = (X[:, 0] > 0).astype(np.int64)

    rf = RandomForest(
        n_estimators=n_estimators,
        impurity=impurity,
        max_height=max_height,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs
    )

    rf.fit(X, y)

    preds = rf.predict(X)

    # ---- type checks
    assert isinstance(preds, np.ndarray)
    assert preds.dtype == np.int64
    assert preds.shape == (X.shape[0],)

    # ---- validity checks
    assert np.all(np.isin(preds, [0, 1]))

    # ---- sanity accuracy (not random)
    acc = (preds == y).mean()
    assert acc >= 0.55
    