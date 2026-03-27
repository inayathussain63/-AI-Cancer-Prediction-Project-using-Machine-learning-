"""
Main pipeline script for the AI Cancer Prediction Project.

Runs the full ML cycle:
  1. Load and preprocess data
  2. Train all classifiers
  3. Evaluate and compare models
  4. Save the best model and the scaler
  5. Generate visualisations
"""

import os
import joblib

from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_all_models
from src.model_evaluation import (
    evaluate_all_models,
    plot_confusion_matrix,
    plot_roc_curves,
    print_summary,
)

MODELS_DIR = "models"


def main():
    print("=" * 60)
    print("  AI Cancer Prediction – Machine Learning Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load & preprocess data                                           #
    # ------------------------------------------------------------------ #
    print("\n[1/5] Loading Wisconsin Breast Cancer dataset …")
    df = load_data()
    print(f"      Dataset shape : {df.shape}")
    print(f"      Class balance : {df['target'].value_counts().to_dict()}")

    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"      Train samples : {len(X_train)}")
    print(f"      Test  samples : {len(X_test)}")

    # ------------------------------------------------------------------ #
    # 2. Train models                                                     #
    # ------------------------------------------------------------------ #
    print("\n[2/5] Training classifiers …")
    models = train_all_models(X_train, y_train)
    print(f"      Trained : {list(models.keys())}")

    # ------------------------------------------------------------------ #
    # 3. Evaluate models                                                  #
    # ------------------------------------------------------------------ #
    print("\n[3/5] Evaluating models on the test set …\n")
    results = evaluate_all_models(models, X_test, y_test)
    print_summary(results)

    # ------------------------------------------------------------------ #
    # 4. Save the best model (highest accuracy) and the scaler           #
    # ------------------------------------------------------------------ #
    print("[4/5] Saving the best model …")
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = models[best_name]
    print(f"      Best model : {best_name}  "
          f"(accuracy={results[best_name]['accuracy']:.4f})")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"      Model  saved → {model_path}")
    print(f"      Scaler saved → {scaler_path}")

    # ------------------------------------------------------------------ #
    # 5. Generate plots                                                   #
    # ------------------------------------------------------------------ #
    print("\n[5/5] Generating visualisations …")
    for name, model in models.items():
        path = plot_confusion_matrix(model, X_test, y_test, name, MODELS_DIR)
        print(f"      Confusion matrix → {path}")

    roc_path = plot_roc_curves(models, X_test, y_test, MODELS_DIR)
    print(f"      ROC curves       → {roc_path}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
