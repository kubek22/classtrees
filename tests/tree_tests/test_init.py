import pytest
import numpy as np
from classtrees import ClassTree

def test_valid_init():
    tree = ClassTree(
        impurity="gini",
        max_height=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=0
    )
    assert tree is not None


@pytest.mark.parametrize("impurity", ["gini", "entropy"])
def test_valid_impurity(impurity):
    ClassTree(impurity=impurity)

@pytest.mark.parametrize("impurity", [None, -1, "", "invalid", 123, np.array(["gini"])])
def test_invalid_impurity(impurity):
    with pytest.raises(Exception):
        ClassTree(impurity=impurity)

@pytest.mark.parametrize("value", [0, 1, 10, None])
def test_correct_height(value):
    ClassTree(max_height=value)

@pytest.mark.parametrize("value", [0.5, -1, "ten", -10, -0.3, np.array([10]), np.inf])
def test_wrong_height(value):
    with pytest.raises(Exception):
        ClassTree(max_height=value)

@pytest.mark.parametrize("value", [2, 3, 5, 10, 100, 2000, 2.2, 4.7, 43.346, 2.0001])
def test_correct_min_samples_split(value):
    ClassTree(min_samples_split=value)

@pytest.mark.parametrize("value", [0.5, 1, "two", -10, None, np.array([2]), np.inf, 1.9999, 0.0001])
def test_wrong_min_samples_split(value):
    with pytest.raises(Exception):
        ClassTree(min_samples_split=value)

@pytest.mark.parametrize("value", [1, 2, 3, 5, 10, 100, 2000, 1.5, 3.7, 4.2, 1.0001])
def test_correct_min_samples_leaf(value):
    ClassTree(min_samples_leaf=value)

@pytest.mark.parametrize("value", [0.5, -7.6, "one", -10, None, np.array([1]), np.inf, 0.9999, 0.0001])
def test_wrong_min_samples_leaf(value):
    with pytest.raises(Exception):
        ClassTree(min_samples_leaf=value)
    
@pytest.mark.parametrize("value", [1, 2, 3, 5, 10, 100, 2000, None, 1.5, 3.7, 1.0001])
def test_correct_max_features(value):
    ClassTree(max_features=value)

@pytest.mark.parametrize("value", [-1, -5, "a", np.inf, np.array([1]), 0.5, 0.9999])
def test_wrong_max_features(value):
    with pytest.raises(Exception):
        ClassTree(max_features=value)

@pytest.mark.parametrize("value", [None, 0, 1, 42, 123456789])
def test_correct_random_state(value):
    ClassTree(random_state=value)

@pytest.mark.parametrize("value", [-1, 2.5, "seed", np.array([0]), np.inf])
def test_wrong_random_state(value):
    with pytest.raises(Exception):
        ClassTree(random_state=value)

def test_unknown_param():
    with pytest.raises(Exception):
        ClassTree(unknown=0)

