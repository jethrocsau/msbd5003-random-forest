import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import load, dump
from time import time
import pandas as pd
import json
from tqdm import tqdm

np.random.seed(42)


def printls(*s):
    s = " ".join([str(i) for i in s])
    print(f"{'-'*10}{s}{'-'*10}")


class RandomForest:
    def __init__(self, n_estimators=100, criterion="gini"):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.estimators = []
        self.feature_subsets = []
        self.data_subsets = []
        self.random_state = np.random.RandomState(42)

    def fit(self, X, y):
        start_time = time()
        n_samples, n_features = X.shape

        for _ in tqdm(range(self.n_estimators)):
            subset_indices = self.random_state.choice(
                n_samples, n_samples, replace=True
            )
            feature_indices = self.random_state.choice(
                n_features, int(np.sqrt(n_features)), replace=False
            )

            subset_X = X.iloc[subset_indices, feature_indices]
            subset_y = y[subset_indices]

            tree = DecisionTreeClassifier(
                criterion=self.criterion, random_state=self.random_state  # type: ignore
            )
            tree.fit(subset_X, subset_y)

            self.estimators.append(tree)
            self.feature_subsets.append(feature_indices)
            self.data_subsets.append(subset_indices)

        return time() - start_time

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.estimators)))

        for i, tree in enumerate(self.estimators):
            subset_X = X.iloc[:, self.feature_subsets[i]]
            predictions[:, i] = tree.predict(subset_X)

        return np.mean(predictions, axis=1).round().astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    # Set the seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    training_configs = [
        {
            "X": load("data/all_categorical_arr.joblib"),
            "y": load("data/y_categorical.joblib"),
            "n_estimators": 100,
            "name": "categorical_100_estimators",
        },
        {
            "X": load("data/all_numerical_arr.joblib"),
            "y": load("data/y_numerical.joblib"),
            "n_estimators": 100,
            "name": "numerical_100_estimators",
        },
        {
            "X": load("data/balanced_arr.joblib"),
            "y": load("data/y_balanced.joblib"),
            "n_estimators": 100,
            "name": "balanced_100_estimators",
        },
        {
            "X": load("data/balanced_arr.joblib"),
            "y": load("data/y_balanced.joblib"),
            "n_estimators": 1_000,
            "name": "balanced_1000_estimators",
        },
    ]

    runtime_dict = {}
    for training_config in training_configs:
        for criterion in ["gini"]:
            training_config["X"] = pd.DataFrame(training_config["X"])
            training_config["X"].columns = training_config["X"].columns.map(str)

            X = training_config["X"]
            y = training_config["y"]
            n_estimators = training_config["n_estimators"]

            # Call the random_forest_classification function with the seed
            rf_clf = RandomForest(n_estimators=n_estimators)
            elapsed_time = rf_clf.fit(X, y)

            accuracy = rf_clf.score(X, y)
            feature_subsets = rf_clf.feature_subsets
            data_subsets = rf_clf.data_subsets

            printls(training_config["name"], criterion)
            print("Accuracy:", accuracy)
            print("Elapsed Time:", elapsed_time, "seconds")

            runtime_dict[f"{training_config['name']}_{criterion}"] = elapsed_time

            dump(
                feature_subsets,
                f"data/{training_config['name']}_feature_subsets.joblib",
            )
            dump(data_subsets, f"data/{training_config['name']}_data_subsets.joblib")

    json.dump(runtime_dict, open("data/runtime_result.json", "w"))
