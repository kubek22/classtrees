from classtrees.tree_module import PyTree

# python class that wraps the PyTree class and provides a scikit-learn-like interface

class ClassTree:
    def __init__(self, impurity="gini", max_height=10, min_samples_split=2):
        self._tree = PyTree(
            impurity=impurity,
            max_height=max_height,
            min_samples_split=min_samples_split
        )

    def fit(self, X, y):
        return self._tree.fit(X, y)

    def predict(self, X):
        return self._tree.predict(X)

    # def predict_proba(self, X):
    #     return self._tree.predict_proba(X)  # placeholder
