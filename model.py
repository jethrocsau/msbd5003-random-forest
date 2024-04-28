import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


def random_forest_classification(X, y, random_state):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Create a Random Forest classifier with 100 trees
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)

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

    return accuracy, elapsed_time


if __name__ == "__main__":
    # Set the seed for reproducibility
    RANDOM_SEED = 42
    TRAIN_SIZE = 100_000
    NUM_FEATURES = 50
    np.random.seed(RANDOM_SEED)

    # Generate a NumPy array with random values of shape (100000, 100)
    X = np.random.rand(TRAIN_SIZE, NUM_FEATURES)

    # Generate random target variable
    y = np.random.randint(0, 2, size=TRAIN_SIZE)

    # Call the random_forest_classification function with the seed
    accuracy, elapsed_time = random_forest_classification(X, y, RANDOM_SEED)

    print("Accuracy:", accuracy)
    print("Elapsed Time:", elapsed_time, "seconds")
