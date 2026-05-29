import numpy as np
import time

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from classtrees import ClassTree


# -----------------------------
# CONFIG
# -----------------------------
N_SAMPLES = 50000
N_FEATURES = 50
N_CLASSES = 4
N_RUNS = 10
RANDOM_STATE = 42
CRITERION = "gini"
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 2 #5
MIN_SAMPLES_LEAF = 1 #2


# -----------------------------
# DATASET
# -----------------------------
def generate_data():
    return make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=30,
        n_redundant=10,
        n_classes=N_CLASSES,
        random_state=RANDOM_STATE
    )


# -----------------------------
# TIMING UTIL
# -----------------------------
def time_fn(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    end = time.perf_counter()
    return result, end - start


def mean_std(values):
    arr = np.array(values)
    return arr.mean(), arr.std()


# -----------------------------
# BENCHMARK CORE
# -----------------------------
def benchmark_model(name, model_cls, X, y):
    fit_times = []
    pred_times = []
    proba_times = []
    accuracies = []

    for i in range(N_RUNS):
        model = model_cls()

        # FIT
        _, t_fit = time_fn(model.fit, X, y)

        # PREDICT
        y_pred, t_pred = time_fn(model.predict, X)

        # PROBA
        y_proba, t_proba = time_fn(model.predict_proba, X)

        acc = accuracy_score(y, y_pred)

        fit_times.append(t_fit)
        pred_times.append(t_pred)
        proba_times.append(t_proba)
        accuracies.append(acc)

    print(f"\n=== {name} ===")
    print(f"Fit time:     {mean_std(fit_times)[0]:.6f} ± {mean_std(fit_times)[1]:.6f}s")
    print(f"Predict time: {mean_std(pred_times)[0]:.6f} ± {mean_std(pred_times)[1]:.6f}s")
    print(f"Proba time:   {mean_std(proba_times)[0]:.6f} ± {mean_std(proba_times)[1]:.6f}s")
    print(f"Accuracy:     {mean_std(accuracies)[0]:.4f}")


# -----------------------------
# WRAPPERS
# -----------------------------
class SklearnTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            criterion=CRITERION,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=N_FEATURES,
            random_state=None
        )

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MyTree:
    def __init__(self):
        self.model = ClassTree(max_height=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
                               min_samples_leaf=MIN_SAMPLES_LEAF, max_features=N_FEATURES,
                               random_state=None, impurity=CRITERION)

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    X, y = generate_data()

    benchmark_model("Custom C Tree", MyTree, X, y)
    benchmark_model("sklearn DecisionTree", SklearnTree, X, y)
