import pytest
from classtrees import ClassTree


def test_valid_initialization():
    tree = ClassTree(impurity="gini", max_height=10, min_samples_split=2)
    assert tree is not None


def test_invalid_impurity():
    with pytest.raises(Exception):
        ClassTree(impurity="unknown")


def test_invalid_types():
    with pytest.raises(TypeError):
        ClassTree(max_height="ten")

    with pytest.raises(TypeError):
        ClassTree(min_samples_split=2.5)


def test_negative_values():
    with pytest.raises(Exception):
        ClassTree(max_height=-1)

    with pytest.raises(Exception):
        ClassTree(min_samples_split=0)