import pytest
import numpy as np
from classtrees import RandomForest

def test_valid_init():
    rf = RandomForest(
        impurity="gini",
        max_height=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=0
    )
    assert rf is not None

@pytest.mark.parametrize("value", [1, 2, 5, 10, 100, 1000])
def test_correct_n_estimators(value):
    RandomForest(n_estimators=value)

@pytest.mark.parametrize("value", [0.5, -1, "ten", -10, -0.3, np.array([10]), np.inf, True, False])
def test_wrong_n_estimators(value):
    with pytest.raises(Exception):
        RandomForest(n_estimators=value)

@pytest.mark.parametrize("impurity", ["gini", "entropy"])
def test_valid_impurity(impurity):
    RandomForest(impurity=impurity)

@pytest.mark.parametrize("impurity", [None, -1, "", "invalid", 123, np.array(["gini"]), True, False])
def test_invalid_impurity(impurity):
    with pytest.raises(Exception):
        RandomForest(impurity=impurity)

@pytest.mark.parametrize("value", [0, 1, 10, None])
def test_correct_height(value):
    RandomForest(max_height=value)

@pytest.mark.parametrize("value", [0.5, -1, "ten", -10, -0.3, np.array([10]), np.inf, True, False])
def test_wrong_height(value):
    with pytest.raises(Exception):
        RandomForest(max_height=value)

@pytest.mark.parametrize("value", [2, 3, 5, 10, 100, 2000])
def test_correct_min_samples_split(value):
    RandomForest(min_samples_split=value)

@pytest.mark.parametrize("value", [0.5, 1, "two", -10, None, np.array([2]), np.inf, 1.9999, 0.0001, True, False, 2.2, 4.7, 43.346, 2.0001])
def test_wrong_min_samples_split(value):
    with pytest.raises(Exception):
        RandomForest(min_samples_split=value)

@pytest.mark.parametrize("value", [1, 2, 3, 5, 10, 100, 2000])
def test_correct_min_samples_leaf(value):
    RandomForest(min_samples_leaf=value)

@pytest.mark.parametrize("value", [0.5, -7.6, "one", -10, None, np.array([1]), np.inf, 0.9999, 0.0001, 1.5, 3.7, 4.2, 1.0001, True, False])
def test_wrong_min_samples_leaf(value):
    with pytest.raises(Exception):
        RandomForest(min_samples_leaf=value)
    
@pytest.mark.parametrize("value", [1, 2, 3, 5, 10, 100, 2000, None])
def test_correct_max_features(value):
    RandomForest(max_features=value)

@pytest.mark.parametrize("value", [-1, -5, "a", np.inf, np.array([1]), 0.5, 0.9999, 1.5, 3.7, 1.0001, True, False])
def test_wrong_max_features(value):
    with pytest.raises(Exception):
        RandomForest(max_features=value)

@pytest.mark.parametrize("value", [None, 0, 1, 42, 123456789])
def test_correct_random_state(value):
    RandomForest(random_state=value)

@pytest.mark.parametrize("value", [-1, 2.5, "seed", np.array([0]), np.inf])
def test_wrong_random_state(value):
    with pytest.raises(Exception):
        RandomForest(random_state=value)

def test_unknown_param():
    with pytest.raises(Exception):
        RandomForest(unknown=0)

@pytest.mark.parametrize("value", [-1, 1, 2, 5, 10, 100, 1000])
def test_correct_n_jobs(value):
    RandomForest(n_jobs=value)

@pytest.mark.parametrize("value", [0.5, -15, "ten", -10, -0.3, np.array([10]), np.inf, True, False, 5.76])
def test_wrong_n_jobs(value):
    with pytest.raises(Exception):
        RandomForest(n_jobs=value)
