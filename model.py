import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from joblib import load
import pandas as pd


def random_forest_classification(X, y, n_estimators=100, random_state=42):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Create a Random Forest classifier with 100 trees
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state  # type: ignore
    )

    # Start the timer
    start_time = time.time()

    # Train the classifier
    rf_classifier.fit(X_train, y_train)

    # Stop the timer and compute the elapsed time
    elapsed_time = time.time() - start_time

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)

    return (
        accuracy,
        elapsed_time,
        rf_classifier.estimators_samples_,  # type: ignore
        [
            np.where(estimator.feature_importances_ > 0)[0].shape
            for estimator in rf_classifier.estimators_
        ],
    )


if __name__ == "__main__":
    # Set the seed for reproducibility
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)

    training_configs = [
        {
            "X": load("data/all_categorical_arr.joblib"),
            "y": load("data/y_categorical.joblib"),
            "n_estimators": 100,
        },
        {
            "X": load("data/all_numerical_arr.joblib"),
            "y": load("data/y_numerical.joblib"),
            "n_estimators": 100,
        },
        {
            "X": load("data/balanced_arr.joblib"),
            "y": load("data/y_balanced.joblib"),
            "n_estimators": 100,
        },
        {
            "X": load("data/balanced_arr.joblib"),
            "y": load("data/y_balanced.joblib"),
            "n_estimators": 10_000,
        },
    ]

    for training_config in training_configs:
        training_config["X"] = pd.DataFrame(training_config["X"])
        training_config["X"].columns = training_config["X"].columns.map(str)

        X = training_config["X"]
        y = training_config["y"]
        n_estimators = training_config["n_estimators"]

        # Call the random_forest_classification function with the seed
        accuracy, elapsed_time, estimators_samples, feature_names = (
            random_forest_classification(
                X, y, random_state=RANDOM_SEED, n_estimators=n_estimators
            )
        )

        print("Accuracy:", accuracy)
        print("Elapsed Time:", elapsed_time, "seconds")
        print("Feature names:", feature_names)
