import numpy as np
import time

from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from classtrees import RandomForest


N_SAMPLES = 50000
N_FEATURES = 50
MAX_FEATURES = int(np.sqrt(50))
N_CLASSES = 4
N_RUNS = 10
N_ESTIMATORS = 10
RANDOM_STATE = 42
CRITERION = "gini"
MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
N_JOBS = -1

def generate_data():
    return make_classification(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_informative=30,
        n_redundant=10,
        n_classes=N_CLASSES,
        random_state=RANDOM_STATE
    )

def main():
    X, y = generate_data()
    rf = RandomForest(n_estimators=N_ESTIMATORS, max_features=MAX_FEATURES, random_state=RANDOM_STATE, impurity=CRITERION, max_height=MAX_DEPTH,
                      min_samples_split=MIN_SAMPLES_SPLIT, min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=N_JOBS)
    rf.fit(X, y)
    y_pred = rf.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
