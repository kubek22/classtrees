from classtrees.tree_module import PyTree

# python class that wraps the PyTree class and provides a scikit-learn-like interface

class ClassTree:
    def __init__(self, impurity="gini", max_height=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None):
        """
        max_height: int or None, default=None
        min_samples_split: int or float, default=2
        min_samples_leaf: int or float, default=1
        max_features: int or float or None, default=None
        random_state: int or None, default=None
        """
        self._tree = PyTree(
            impurity=impurity,
            max_height=max_height,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

    def fit(self, X, y):
        return self._tree.fit(X, y)

    def predict(self, X):
        return self._tree.predict(X)

    def predict_proba(self, X):
        return self._tree.predict_proba(X)
