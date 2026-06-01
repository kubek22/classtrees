import time
import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from classtrees import ClassTree, RandomForest


# =====================================================
# CONFIG
# =====================================================

N_SAMPLES = 100_000
N_FEATURES = 50
N_CLASSES = 4

TEST_SIZE = 0.2

N_RUNS = 10

RANDOM_STATE = 42

MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2

MAX_FEATURES_TREE = N_FEATURES
MAX_FEATURES_RF = int(np.sqrt(N_FEATURES))

CRITERION = "gini"

N_JOBS = -1

RF_ESTIMATORS = [10, 25, 50, 100]


# =====================================================
# DATA
# =====================================================

def generate_data():
    return make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=30,
        n_redundant=10,
        n_classes=N_CLASSES,
        random_state=RANDOM_STATE
    )


# =====================================================
# UTILS
# =====================================================

def mean_std(values):
    arr = np.asarray(values)
    return arr.mean(), arr.std()


def time_fn(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    end = time.perf_counter()
    return result, end - start


# =====================================================
# BENCHMARK
# =====================================================

def benchmark_model(name, model_factory,
                    X_train, X_test,
                    y_train, y_test):

    fit_times = []
    pred_times = []
    proba_times = []

    train_accs = []
    test_accs = []

    for _ in range(N_RUNS):

        model = model_factory()

        _, fit_time = time_fn(
            model.fit,
            X_train,
            y_train
        )

        y_pred_train, train_pred_time = time_fn(
            model.predict,
            X_train
        )

        y_pred_test, test_pred_time = time_fn(
            model.predict,
            X_test
        )

        _, proba_time = time_fn(
            model.predict_proba,
            X_test
        )

        train_acc = accuracy_score(
            y_train,
            y_pred_train
        )

        test_acc = accuracy_score(
            y_test,
            y_pred_test
        )

        fit_times.append(fit_time)

        pred_times.append(
            train_pred_time + test_pred_time
        )

        proba_times.append(proba_time)

        train_accs.append(train_acc)
        test_accs.append(test_acc)

    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Fit time:       "
        f"{mean_std(fit_times)[0]:.4f} ± "
        f"{mean_std(fit_times)[1]:.4f}s"
    )

    print(
        f"Predict time:   "
        f"{mean_std(pred_times)[0]:.4f} ± "
        f"{mean_std(pred_times)[1]:.4f}s"
    )

    print(
        f"Predict_proba:  "
        f"{mean_std(proba_times)[0]:.4f} ± "
        f"{mean_std(proba_times)[1]:.4f}s"
    )

    print(
        f"Train accuracy: "
        f"{mean_std(train_accs)[0]:.4f}"
    )

    print(
        f"Test accuracy:  "
        f"{mean_std(test_accs)[0]:.4f}"
    )


# =====================================================
# MAIN
# =====================================================

def main():

    X, y = generate_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # -------------------------------------------------
    # DECISION TREES
    # -------------------------------------------------

    benchmark_model(
        "Custom ClassTree",
        lambda: ClassTree(
            max_height=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES_TREE,
            impurity=CRITERION,
            random_state=None
        ),
        X_train,
        X_test,
        y_train,
        y_test
    )

    benchmark_model(
        "sklearn DecisionTree",
        lambda: DecisionTreeClassifier(
            criterion=CRITERION,
            max_depth=MAX_DEPTH,
            min_samples_split=MIN_SAMPLES_SPLIT,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_features=MAX_FEATURES_TREE,
            random_state=None
        ),
        X_train,
        X_test,
        y_train,
        y_test
    )

    # -------------------------------------------------
    # RANDOM FORESTS
    # -------------------------------------------------

    for n_estimators in RF_ESTIMATORS:

        benchmark_model(
            f"Custom RandomForest ({n_estimators} trees)",
            lambda n=n_estimators: RandomForest(
                n_estimators=n,
                max_features=MAX_FEATURES_RF,
                random_state=None,
                impurity=CRITERION,
                max_height=MAX_DEPTH,
                min_samples_split=MIN_SAMPLES_SPLIT,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                n_jobs=-1
            ),
            X_train,
            X_test,
            y_train,
            y_test
        )

        benchmark_model(
            f"sklearn RandomForest ({n_estimators} trees)",
            lambda n=n_estimators: RandomForestClassifier(
                n_estimators=n,
                criterion=CRITERION,
                max_depth=MAX_DEPTH,
                min_samples_split=MIN_SAMPLES_SPLIT,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                max_features=MAX_FEATURES_RF,
                n_jobs=-1,
                random_state=None
            ),
            X_train,
            X_test,
            y_train,
            y_test
        )


if __name__ == "__main__":
    main()
