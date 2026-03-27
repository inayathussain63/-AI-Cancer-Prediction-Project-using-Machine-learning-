"""
Prediction interface for the AI Cancer Prediction Project.

Loads a saved model + scaler and provides a simple predict() function
that accepts a list of clinical feature values.
"""

import numpy as np
import joblib


def predict(feature_values, model_path, scaler_path):
    """
    Predict whether a tumor is malignant or benign.

    Parameters
    ----------
    feature_values : list or array-like of shape (n_features,)
        Raw (unscaled) clinical feature values in the same order as the
        Wisconsin Breast Cancer dataset features.
    model_path : str
        Path to a joblib-serialised trained model.
    scaler_path : str
        Path to a joblib-serialised StandardScaler.

    Returns
    -------
    dict
        {
            'prediction': 'Malignant' | 'Benign',
            'probability_malignant': float,
            'probability_benign': float,
        }
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X = np.array(feature_values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    label = model.predict(X_scaled)[0]
    prediction = "Benign" if label == 1 else "Malignant"

    result = {"prediction": prediction}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        result["probability_malignant"] = round(float(probs[0]), 4)
        result["probability_benign"] = round(float(probs[1]), 4)

    return result
