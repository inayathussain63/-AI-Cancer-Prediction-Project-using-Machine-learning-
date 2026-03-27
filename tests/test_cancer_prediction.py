"""
Unit tests for the AI Cancer Prediction Project.
"""

import os
import numpy as np
import pytest
import joblib

from src.data_preprocessing import load_data, preprocess_data, get_feature_names
from src.model_training import (
    train_logistic_regression,
    train_random_forest,
    train_svm,
    train_all_models,
)
from src.model_evaluation import evaluate_model, evaluate_all_models
from src.predict import predict


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def prepared_data():
    df = load_data()
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    return X_train, X_test, y_train, y_test, scaler


@pytest.fixture(scope="module")
def trained_models(prepared_data):
    X_train, _, y_train, _, _ = prepared_data
    return train_all_models(X_train, y_train)


# ---------------------------------------------------------------------------
# Data preprocessing tests
# ---------------------------------------------------------------------------

class TestDataPreprocessing:
    def test_load_data_shape(self):
        df = load_data()
        assert df.shape == (569, 31), "Dataset should have 569 rows and 31 columns"

    def test_load_data_has_target(self):
        df = load_data()
        assert "target" in df.columns

    def test_load_data_binary_target(self):
        df = load_data()
        assert set(df["target"].unique()) == {0, 1}

    def test_preprocess_split_sizes(self, prepared_data):
        X_train, X_test, y_train, y_test, _ = prepared_data
        total = len(X_train) + len(X_test)
        assert total == 569
        assert abs(len(X_test) / total - 0.2) < 0.02

    def test_preprocess_scaled_range(self, prepared_data):
        X_train, _, _, _, _ = prepared_data
        # After StandardScaler the mean of each feature should be close to 0
        assert np.allclose(X_train.mean(axis=0), 0, atol=1e-6)

    def test_get_feature_names_count(self):
        names = get_feature_names()
        assert len(names) == 30


# ---------------------------------------------------------------------------
# Model training tests
# ---------------------------------------------------------------------------

class TestModelTraining:
    def test_logistic_regression_trains(self, prepared_data):
        X_train, _, y_train, _, _ = prepared_data
        model = train_logistic_regression(X_train, y_train)
        assert hasattr(model, "predict")

    def test_random_forest_trains(self, prepared_data):
        X_train, _, y_train, _, _ = prepared_data
        model = train_random_forest(X_train, y_train)
        assert hasattr(model, "predict")

    def test_svm_trains(self, prepared_data):
        X_train, _, y_train, _, _ = prepared_data
        model = train_svm(X_train, y_train)
        assert hasattr(model, "predict")

    def test_train_all_models_keys(self, trained_models):
        expected = {"Logistic Regression", "Random Forest", "SVM"}
        assert set(trained_models.keys()) == expected

    def test_models_predict_correct_classes(self, trained_models, prepared_data):
        _, X_test, _, y_test, _ = prepared_data
        for name, model in trained_models.items():
            preds = model.predict(X_test)
            assert set(preds).issubset({0, 1}), f"{name} predicted unexpected labels"


# ---------------------------------------------------------------------------
# Model evaluation tests
# ---------------------------------------------------------------------------

class TestModelEvaluation:
    def test_evaluate_model_keys(self, trained_models, prepared_data):
        _, X_test, _, y_test, _ = prepared_data
        model = trained_models["Logistic Regression"]
        result = evaluate_model(model, X_test, y_test)
        assert "accuracy" in result
        assert "roc_auc" in result
        assert "report" in result

    def test_evaluate_model_accuracy_range(self, trained_models, prepared_data):
        _, X_test, _, y_test, _ = prepared_data
        for name, model in trained_models.items():
            result = evaluate_model(model, X_test, y_test)
            acc = result["accuracy"]
            assert 0.8 <= acc <= 1.0, f"{name} accuracy {acc:.4f} below 80%"

    def test_evaluate_model_roc_auc_range(self, trained_models, prepared_data):
        _, X_test, _, y_test, _ = prepared_data
        for name, model in trained_models.items():
            result = evaluate_model(model, X_test, y_test)
            if result["roc_auc"] is not None:
                assert 0.8 <= result["roc_auc"] <= 1.0

    def test_evaluate_all_models_returns_all(self, trained_models, prepared_data):
        _, X_test, _, y_test, _ = prepared_data
        results = evaluate_all_models(trained_models, X_test, y_test)
        assert set(results.keys()) == set(trained_models.keys())


# ---------------------------------------------------------------------------
# Prediction interface tests
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_valid_label(self, trained_models, prepared_data, tmp_path):
        _, X_test, _, _, scaler = prepared_data
        model = trained_models["Logistic Regression"]

        model_path = str(tmp_path / "model.pkl")
        scaler_path = str(tmp_path / "scaler.pkl")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Use the first test sample (raw values recovered by inverse_transform)
        raw_sample = scaler.inverse_transform(X_test[:1])[0].tolist()
        result = predict(raw_sample, model_path, scaler_path)

        assert result["prediction"] in ("Malignant", "Benign")

    def test_predict_probabilities_sum_to_one(self, trained_models, prepared_data, tmp_path):
        _, X_test, _, _, scaler = prepared_data
        model = trained_models["Random Forest"]

        model_path = str(tmp_path / "rf_model.pkl")
        scaler_path = str(tmp_path / "rf_scaler.pkl")
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        raw_sample = scaler.inverse_transform(X_test[:1])[0].tolist()
        result = predict(raw_sample, model_path, scaler_path)

        total = result["probability_malignant"] + result["probability_benign"]
        assert abs(total - 1.0) < 1e-3
