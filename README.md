# classtrees

`classtrees` is a lightweight Python package that implements fast classification trees and random forest. It combines a NumPy-friendly Python API with a compiled C backend to deliver efficient training and prediction for binary and multiclass classification.

## Key Features

- Decision tree classifier with configurable tree height, split criteria, and feature subsampling
- Random forest ensemble with bootstrap aggregation and support for parallel training
- Predict class labels and class probabilities
- Built as a C extension for performance
- Designed for NumPy arrays and scientific workflows

## Installation

### Requirements

- Python 3.10 or newer
- NumPy
- C compiler with OpenMP support (required to build the C extension)

### Install from PyPI

```bash
python -m pip install classtrees
```

### Install from source

```bash
python -m pip install .
```

## Quick Start

```python
import numpy as np
from classtrees import ClassTree, RandomForest

# training data
X = np.random.randn(100, 5)
y = (X[:, 0] > 0).astype(np.int64)

# single decision tree
tree = ClassTree(max_height=5, random_state=42)
tree.fit(X, y)
y_pred = tree.predict(X)
proba = tree.predict_proba(X)

# random forest
forest = RandomForest(n_estimators=50, max_height=8, random_state=42, n_jobs=1)
forest.fit(X, y)
forest_pred = forest.predict(X)
forest_proba = forest.predict_proba(X)
```

## API

### `ClassTree`

A single decision tree classifier.

Constructor arguments:

- `impurity`: `'gini'` or `'entropy'` (default: `'gini'`)
- `max_height`: maximum tree height, or `None` to grow until no valid split remains
- `min_samples_split`: minimum samples required to split a node (default: `2`)
- `min_samples_leaf`: minimum samples required in each leaf node (default: `1`)
- `max_features`: number of features considered for splits, or `None` to use all features
- `random_state`: random seed for reproducible training

Methods:

- `fit(X, y)`: train the classifier on feature matrix `X` and labels `y`
- `predict(X)`: return predicted class labels for input samples
- `predict_proba(X)`: return predicted class probabilities

### `RandomForest`

An ensemble of decision trees with bootstrap sampling.

Constructor arguments:

- `n_estimators`: number of trees in the forest (default: `100`)
- `impurity`, `max_height`, `min_samples_split`, `min_samples_leaf`, `max_features`, `random_state`: same as `ClassTree`
- `n_jobs`: number of worker threads for training and prediction (default: `1`). Use `-1` to enable all available cores.

Methods:

- `fit(X, y)`: train the forest ensemble
- `predict(X)`: return ensemble class predictions
- `predict_proba(X)`: return averaged class probabilities from all trees

## Documentation

Full documentation is available in the project `docs/` folder and online at:

- https://kubek22.github.io/classtrees/

## License

This project is licensed under the BSD-3-Clause license. See `LICENSE` for details.

## Project Links

- Homepage: https://github.com/kubek22/classtrees
- Repository: https://github.com/kubek22/classtrees
