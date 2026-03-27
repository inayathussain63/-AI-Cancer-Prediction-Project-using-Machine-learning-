"""
Data preprocessing module for the AI Cancer Prediction Project.

Loads the Wisconsin Breast Cancer dataset, performs feature scaling,
and splits data into training and testing sets.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Load the Wisconsin Breast Cancer dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing features and a 'target' column where
        0 = malignant and 1 = benign.
    """
    dataset = load_breast_cancer()
    df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target
    return df


def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the DataFrame by splitting into train/test sets and
    applying StandardScaler to the features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with features and a 'target' column.
    test_size : float
        Proportion of the dataset to include in the test split.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scaler)
        X_train and X_test are scaled numpy arrays.
        scaler is the fitted StandardScaler instance.
    """
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def get_feature_names():
    """
    Return the feature names from the Wisconsin Breast Cancer dataset.

    Returns
    -------
    list of str
    """
    return list(load_breast_cancer().feature_names)
