import numpy as np
import time
import csv

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from classtrees import ClassTree


# -----------------------------
# CONFIG
# -----------------------------
BASE_SAMPLES = 5_000
MAX_SAMPLES = 320_000
GROWTH_FACTOR = 2

N_FEATURES = 50
N_CLASSES = 4
N_RUNS = 5

RANDOM_STATE = 42

CRITERION = "gini"
MAX_DEPTH = None
MIN_SAMPLES_SPLIT = 5

CSV_OUTPUT = "benchmark_scaling.csv"


# -----------------------------
# DATA GENERATION
# -----------------------------
def generate_data(n_samples):
    return make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=30,
        n_redundant=10,
        n_classes=N_CLASSES,
        random_state=RANDOM_STATE
    )


# -----------------------------
# UTIL
# -----------------------------
def time_fn(fn, *args):
    start = time.perf_counter()
    res = fn(*args)
    end = time.perf_counter()
    return res, end - start


def mean_std(x):
    x = np.array(x)
    return float(x.mean()), float(x.std())


# -----------------------------
# WRAPPERS
# -----------------------------
class SklearnTree:
    def __init__(self):
        self.model = DecisionTreeClassifier(
            criterion=CRITERION,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
        )

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MyTree:
    def __init__(self):
        self.model = ClassTree(
            max_height=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            impurity=CRITERION
        )

    def fit(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# -----------------------------
# CORE BENCHMARK
# -----------------------------
def benchmark_one(model_cls, X, y):
    fit_times, pred_times, proba_times, accs = [], [], [], []

    for _ in range(N_RUNS):
        model = model_cls()

        _, t_fit = time_fn(model.fit, X, y)
        _, t_pred = time_fn(model.predict, X)
        _, t_proba = time_fn(model.predict_proba, X)

        y_pred = model.predict(X)
        acc = accuracy_score(y, y_pred)

        fit_times.append(t_fit)
        pred_times.append(t_pred)
        proba_times.append(t_proba)
        accs.append(acc)

    return {
        "fit_mean": mean_std(fit_times)[0],
        "fit_std": mean_std(fit_times)[1],
        "pred_mean": mean_std(pred_times)[0],
        "pred_std": mean_std(pred_times)[1],
        "proba_mean": mean_std(proba_times)[0],
        "proba_std": mean_std(proba_times)[1],
        "acc": mean_std(accs)[0],
    }


# -----------------------------
# MAIN SCALING LOOP
# -----------------------------
def run_scaling():
    sizes = []
    n = BASE_SAMPLES

    while n <= MAX_SAMPLES:
        sizes.append(n)
        n *= GROWTH_FACTOR

    results = []

    print("\n===== SCALING BENCHMARK =====\n")

    for n in sizes:
        print(f"\n--- N_SAMPLES = {n} ---")

        X, y = generate_data(n)

        my_stats = benchmark_one(MyTree, X, y)
        sk_stats = benchmark_one(SklearnTree, X, y)

        print("Custom:")
        print(my_stats)

        print("sklearn:")
        print(sk_stats)

        results.append({
            "n": n,

            "my_fit": my_stats["fit_mean"],
            "my_pred": my_stats["pred_mean"],
            "my_proba": my_stats["proba_mean"],
            "my_acc": my_stats["acc"],

            "sk_fit": sk_stats["fit_mean"],
            "sk_pred": sk_stats["pred_mean"],
            "sk_proba": sk_stats["proba_mean"],
            "sk_acc": sk_stats["acc"],
        })

    # save CSV
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to {CSV_OUTPUT}")


# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    run_scaling()