from classtrees.tree_module import PyForest


class RandomForest:
    """
    Random forest classifier.

    A random forest is an ensemble of decision trees trained on bootstrap
    samples of the training data. Predictions are obtained by aggregating
    the predictions of individual trees, which typically improves
    generalization performance and reduces overfitting compared to a
    single decision tree.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest. Must be at least 1.

    impurity : {'gini', 'entropy'}, default='gini'
        Impurity criterion used to evaluate candidate splits.

        - ``'gini'``: Gini impurity.
        - ``'entropy'``: entropy.

    max_height : int or None, default=None
        Maximum height of the tree. The minimal value is 0, then only a root note is created.
        If ``None``, nodes are expanded until no further valid splits can be made.

    min_samples_split : int, default=2
        Minimum number of samples required to split a node.
        Must be at least 2.

    min_samples_leaf : int, default=1
        Minimum number of samples required in each leaf node.

    max_features : int or None, default=None
        Number of features considered when searching for the best split
        in each tree. If ``None``, all features are considered.

    random_state : int or None, default=None
        Seed used by the random number generator. If ``None``, an
        seed is set based on current timestamp.

    n_jobs : int, default=1
        Number of worker threads used during training and prediction.

        - ``1``: Use a single thread.
        - ``-1``: Use all available CPU cores.
        - ``n >= 1``: Use exactly ``n`` threads.

    Notes
    -----
    Input features are expected to be provided as a two-dimensional
    ``numpy.ndarray`` of floating-point values. Target labels should be
    provided as a one-dimensional integer array.

    Examples
    --------
    >>> clf = RandomForest(
    ...     n_estimators=200,
    ...     max_height=10,
    ...     random_state=42
    ... )
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(self, n_estimators=100, impurity="gini", max_height=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, random_state=None, n_jobs=1):
        self._forest = PyForest(
            n_estimators=n_estimators,
            impurity=impurity,
            max_height=max_height,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs
        )

    def fit(self, X, y):
        """
        Build a random forest classifier from training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.

        y : ndarray of shape (n_samples,)
            Target class labels encoded as integers.

        Returns
        -------
        RandomForest
            Fitted classifier.

        Notes
        -----
        The forest consists of ``n_estimators`` independently trained
        decision trees.

        Examples
        --------
        >>> clf = RandomForest()
        >>> clf.fit(X, y)
        """

        return self._forest.rffit(X, y)

    def predict(self, X):
        """
        Predict class labels for samples in ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels obtained by soft voting among the
            trees in the forest.

        Examples
        --------
        >>> y_pred = clf.predict(X_test)
        """
        
        return self._forest.rfpredict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities for samples in ``X``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Predicted class probabilities. Probabilities are computed by
            averaging the class probability estimates of all trees in the
            forest.

        Examples
        --------
        >>> proba = clf.predict_proba(X_test)
        """
        
        return self._forest.rfpredict_proba(X)
