"""
Model training module for the AI Cancer Prediction Project.

Provides functions to train Logistic Regression, Random Forest, and
Support Vector Machine classifiers on preprocessed data.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def train_logistic_regression(X_train, y_train, random_state=42):
    """
    Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Scaled training features.
    y_train : array-like of shape (n_samples,)
        Training labels.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    LogisticRegression
        Fitted Logistic Regression model.
    """
    model = LogisticRegression(max_iter=10000, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Scaled training features.
    y_train : array-like of shape (n_samples,)
        Training labels.
    n_estimators : int
        Number of trees in the forest.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    RandomForestClassifier
        Fitted Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, kernel="rbf", random_state=42):
    """
    Train a Support Vector Machine classifier.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Scaled training features.
    y_train : array-like of shape (n_samples,)
        Training labels.
    kernel : str
        Kernel type to be used in the SVM algorithm.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    SVC
        Fitted SVM model.
    """
    model = SVC(kernel=kernel, probability=True, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train, random_state=42):
    """
    Train all three classifiers and return them in a dictionary.

    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Scaled training features.
    y_train : array-like of shape (n_samples,)
        Training labels.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary mapping model name to fitted model instance.
    """
    return {
        "Logistic Regression": train_logistic_regression(
            X_train, y_train, random_state=random_state
        ),
        "Random Forest": train_random_forest(
            X_train, y_train, random_state=random_state
        ),
        "SVM": train_svm(X_train, y_train, random_state=random_state),
    }
